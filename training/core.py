"""The core LightningModule parent class which will be inherited by the 
subclasses in modules.py """

from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class LoaderModule(LightningDataModule):
    """
    The LoaderModule is responsible for passing batches of data to the
    UpdateModule subclass during a TrainingModule train run.

    Each of the train_data, val_data and test_data arguments need to be
    specified with lists of tensors that have equal sized first dimensions.
    For example an image dataset for supervised training of a classifier might
    have 100 paired (X, Y) training samples which would be two tensors
    in a list:

    train_data = [image tensor of dimension [100, 1, 8, 8],
                label tensor of dimension [100, 1],
        ]

    train_data and val_data must be specified (val_data is necessary to
    prevent overfitting). test_data is optional, but highly recommended.

    All the dataloaders will pass a batch size as specified in the
    batch_size argument.
    """

    def __init__(
        self,
        train_data: Sequence[torch.Tensor],
        val_data: Sequence[torch.Tensor],
        test_data: Optional[Sequence[torch.Tensor]] = None,
        batch_size: int = 64,
    ):
        super().__init__()

        self.train_dataset = TensorDataset(*train_data)
        self.val_dataset = TensorDataset(*val_data)

        if test_data is not None:
            self.test_dataset = TensorDataset(*test_data)
            self.test_data_present = True
        else:
            self.test_data_present = False

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )


class TrainingModule:
    """
    Matches an UpdateModule subclass (see modules.py) with a LoaderModule
    subclass and then runs a pytorch lightning Trainer upon calls to the
    train method.

    Some of the more important lightning Trainer arguments have been pulled out,
    but additional kwargs can be added using trainer_kwargs.

    Arguments:

    -update_module
        An instance of an UpdateModule subclass
    -loader_module
        An instance of a LoaderModule
    -accelerator
        A string specifying "cpu" or "gpu" use. Defaults to "gpu"
    -devices
        Specifies number of devices to use e.g. 1 GPU. Defaults to 1
    -max_epochs
        Maximum epochs before training is stopped (if early stopping is not
        triggering). Defaults to 500.
    -log_every_n_steps
        How often to log results. Default is every step.
    -trainer_kwargs:
        Option to add additional kwargs to trainer.
        See pytorch_lightning.Trainer for full list
    """

    def __init__(
        self,
        update_module,
        loader_module,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 500,
        log_every_n_steps: int = 1,
        trainer_kwargs: Optional[Dict] = None,
    ):

        kwargs = {} if trainer_kwargs is None else trainer_kwargs
        kwargs["accelerator"] = accelerator
        kwargs["devices"] = devices
        kwargs["max_epochs"] = max_epochs
        kwargs["log_every_n_steps"] = log_every_n_steps

        self.update_module = update_module
        self.loader_module = loader_module
        self.trainer = Trainer(**kwargs)

    def train(self):
        self.trainer.fit(self.update_module, datamodule=self.loader_module)
        if self.loader_module.test_data_present:
            self.trainer.test(
                self.update_module, dataloaders=self.loader_module
            )


class UpdateModule(LightningModule):
    """
    Wraps the pytorch lightning module, which is designed to handle network
    training in conjunction with the Trainer lightning class. For specific
    network training, this should be used as the parent class.

    The init method of the subclass using this parent class needs to include
    the following super call in its __init__ method:

        super().__init__(
            network,
            dirpath,
            filename,
            maximize_val_target,
            optimizer,
            optimizer_kwargs,
            lr_scheduler_kwargs,
            early_stopping_kwargs,
        )

    The subclass at minimum needs to have a training_step and validation_step
    (and a test_step if using Trainer to test). All of these must have
    (self, batch, batch_idx) as their arguments to be compatible with the
    Trainer. The batch variable passed from the loader is usually a list
    containing [X, Y] (although this will obviously change for custom loaders).

    Arguments:

    -network
        A pytorch network (torch.nn.Module)
    -dirpath
        The directory path string to save the model file.
        If left empty will create a lightning_log directory.
    -filename
        The filename string to save the model file.
        If left empty will save in the dirpath location as {epoch}-{step}
    -maximize_val_target
        If set to True, scheduler, early stopper and checkpoint will seek
        to maximize val_target in the logger rather than minimize. Use for
        example if val_target is an accuracy.
        Default False.
    -optimizer
        A torch.optim optimizer function. Defaults to adam with weight decay.
    -optimizer_kwargs
        Optional keyword arguments dict for the optimizer function.
    -lr_scheduler_kwargs
        Optional keyword arguments dict for the learning rate scheduler,
        which is torch.optim.lr_scheduler.ReduceLROnPlateau.
        The "mode" key will be overwritten to be maximize_val_target if set
        The "patience" key value must be less than that of early stopping to
        avoid the lr scheduler being interrupted by early stopping.
    -early_stopping_kwargs
        Optional keyword arguments dict for the early stopping monitor,
        which is pytorch_lightning.callbacks.EarlyStopping
        The "mode" key will be overwritten to be maximize_val_target if set
        The "monitor" key will be overwritten to be val_target if set
    """

    def __init__(
        self,
        network,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        maximize_val_target: bool = False,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
    ):
        super().__init__()

        # Lightning uses max and min strings for selecting best models
        mode = "max" if maximize_val_target else "min"

        # Convert any None kwargs to empty dicts
        optimizer_kwargs, lr_scheduler_kwargs, early_stopping_kwargs = [
            {} if kwargs is None else kwargs
            for kwargs in [
                optimizer_kwargs,
                lr_scheduler_kwargs,
                early_stopping_kwargs,
            ]
        ]

        # Assemble checkpointing instructions dict
        checkpoint_kwargs = {
            "dirpath": dirpath,
            "filename": filename,
            "monitor": "val_target",
            "mode": mode,
        }

        # Force scheduler and early stopper to be in correct mode
        lr_scheduler_kwargs["mode"], early_stopping_kwargs["mode"] = mode, mode

        # Force early stopper to track correct metric
        early_stopping_kwargs["monitor"] = "val_target"

        # If not defined, set patience on scheduler and/or early stopper
        if "patience" not in lr_scheduler_kwargs:
            lr_scheduler_kwargs["patience"] = 10
        if "patience" not in early_stopping_kwargs:
            early_stopping_kwargs["patience"] = 20

        # Check to make sure early stop will not interrupt scheduler
        assert (
            early_stopping_kwargs["patience"] > lr_scheduler_kwargs["patience"]
        ), "Patience of early_stopping_kwargs must be >= lr_scheduler_kwargs."

        # Move checked arguments to class scope
        self.network = network
        self.checkpoint_kwargs = checkpoint_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    **self.lr_scheduler_kwargs,
                ),
                "monitor": "val_target",
            },
        }

    def configure_callbacks(self):
        return [
            EarlyStopping(**self.early_stopping_kwargs),
            ModelCheckpoint(**self.checkpoint_kwargs),
        ]

    def training_step(self, batch, batch_idx):
        """
        The training method in py-lightning. The training step takes a batch
        from the loader and calculates a loss through the process specified
        here.

        Once the loss is found, it needs to be logged and passed out in the
        return, meaning the last two lines of the method will be:

        self.log("train_loss", loss)
        return loss
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """
        The validation method in py-lightning. The validation step takes a batch
        from the loader and calculates a validation target through the process
        specified here.

        Once the target is found, it needs to be logged for early stopping and
        model saving, meaning the last line of the method must be:

        self.log("val_target", target)

        Other validation values can also be logged, but there must always
        be a "val_target" in the logs.

        Remember to set the maximize_val_target argument to True
        if you want the target to be maximized,
        e.g. if you are tracking accuracy
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        """
        The test method in py-lightning. Only necessary if testing using
        the inbuilt Trainer test method (this is encouraged)

        Once the loss is found, it needs to be logged, meaning the last line
        of the method must be:

        self.log("some_test_value", some_test_value)
        """
        raise NotImplementedError
