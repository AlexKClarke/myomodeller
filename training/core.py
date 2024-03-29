"""The core LightningModule parent classes which will be inherited by the 
subclasses in loader and update subclasses"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from training.callbacks import (
    TrainingEpochEnd,
    ValidationEpochEnd,
    TestEpochEnd,
)


class LoaderModule(LightningDataModule):
    """
    The LoaderModule is responsible for passing batches of data to the
    UpdateModule subclass during a TrainingModule train run. This should be
    used as the parent class.

    Each of the train_data, val_data and test_data arguments need to be
    specified with lists of tensors that have equal sized first dimensions.
    For example an image dataset for supervised training of a classifier might
    have 100 paired (X, Y) training samples which would be two tensors
    in a list:

    train_data = [image tensor of dimension [100, 1, 8, 8],
                label tensor of dimension [100, 1],
        ]

    The first item in the list should always be the tensor being input into
    the network

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
        weighted_sampler: bool = False,
        train_sample_weighting: float = 1.0,
        test_sample_weighting: float = 1.0,
        input_shape: Optional[Sequence[int]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        [
            self.train_data_present,
            self.val_data_present,
            self.test_data_present,
        ] = [data is not None for data in [train_data, val_data, test_data]]

        if self.train_data_present:
            self.train_dataset = TensorDataset(*train_data)
            self.train_data_present = True

        if self.val_data_present:
            self.val_dataset = TensorDataset(*val_data)
            self.val_data_present = True

        if self.test_data_present:
            self.test_dataset = TensorDataset(*test_data)
            self.test_data_present = True

        self.batch_size = batch_size
        self.weighted_sampler = weighted_sampler
        if weighted_sampler:
            self.train_sampler = self._generate_weighted_sampler(
                train_data[1], batch_size, train_sample_weighting
            )
            self.val_sampler = self._generate_weighted_sampler(
                val_data[1], batch_size, test_sample_weighting
            )
            self.test_sampler = self._generate_weighted_sampler(
                val_data[1], batch_size, test_sample_weighting
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

        self.input_shape = input_shape
        self.output_shape = output_shape

    def _generate_weighted_sampler(
        self,
        labels: torch.Tensor,
        batch_size: int,
        rare_sample_weighting: float = 1.0,
    ):
        assert (
            labels.dtype == torch.int64
        ), "If using weighted sampler the labels must be torch.int64"

        if len(labels.shape) == 2:
            labels = labels.argmax(1)

        weights = torch.ones(labels.shape[0])
        for c in labels.unique():
            locs = torch.where(labels == c)[0]
            occurrence = labels.shape[0] / locs.shape[0]
            weights[locs] = rare_sample_weighting * occurrence + (
                1 - rare_sample_weighting
            )

        return WeightedRandomSampler(weights, batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
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

    Usually you will also want to add training_epoch_end and
    validation_epoch_end for the purpose of tensorboard logging.

    Arguments:

    -network
        A pytorch network (torch.nn.Module)
    -hpo_mode
        A bool flag which adds a ray tune callback to the callbacks list
        for hyperparameter optimisation
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
        hpo_mode: bool,
        maximize_val_target: bool,
        optimizer: str = "Adam",
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
            "monitor": "val_target",
            "mode": mode,
        }

        # Force scheduler and early stopper to be in correct mode
        lr_scheduler_kwargs["mode"], early_stopping_kwargs["mode"] = mode, mode

        # Force early stopper to track correct metric
        early_stopping_kwargs["monitor"] = "val_target"

        # If not defined, set patience on scheduler and/or early stopper
        if "patience" not in lr_scheduler_kwargs:
            lr_scheduler_kwargs["patience"] = 100
        if "patience" not in early_stopping_kwargs:
            early_stopping_kwargs["patience"] = 200

        # Check to make sure early stop will not interrupt scheduler
        assert (
            early_stopping_kwargs["patience"] > lr_scheduler_kwargs["patience"]
        ), "Patience of early_stopping_kwargs must be >= lr_scheduler_kwargs."

        # Switch off automatic optimisation if multiple optimizers
        if optimizer == "multimodel":
            self.automatic_optimization = False

        # Move checked arguments to class scope
        self.network = network
        self.hpo_mode = hpo_mode
        self.checkpoint_kwargs = checkpoint_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs

        # Add epoch level accumulators for step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if isinstance(self.network, nn.ModuleList):
            return self.network[0](x)
        else:
            return self.network(x)

    def configure_optimizers(self):
        """
        Configure the optimizers to use a reducing learning rate scheduler
        based on the val_target monitor
        """

        networks = (
            self.network if isinstance(self.network, nn.ModuleList) else [self.network]
        )

        if self.optimizer == "multimodel":
            if "optimizer" in self.optimizer_kwargs:
                optimizers = len(networks) * [getattr(torch.optim, self.optimizer)]
                optimizer_kwargs = len(networks) * [self.optimizer_kwargs]
            else:
                assert len(self.optimizer_kwargs) == len(
                    self.network
                ), "Different number of optimizer kwargs to networks."

                optimizers = [
                    getattr(torch.optim, self.optimizer_kwargs[idx]["optimizer"])
                    for idx in range(len(networks))
                ]
                optimizer_kwargs = [
                    self.optimizer_kwargs[idx]["optimizer_kwargs"]
                    for idx in range(len(networks))
                ]
        else:
            optimizers = [getattr(torch.optim, self.optimizer)]
            optimizer_kwargs = [self.optimizer_kwargs]

        opt_configuration = []
        for idx, [net, opt, opt_kwargs] in enumerate(
            zip(networks, optimizers, optimizer_kwargs)
        ):
            optimizer = opt(net.parameters(), **opt_kwargs)
            if idx == 0:
                opt_configuration.append(
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": ReduceLROnPlateau(
                                optimizer,
                                **self.lr_scheduler_kwargs,
                            ),
                            "monitor": "val_target",
                        },
                    }
                )
            else:
                opt_configuration.append(
                    {
                        "optimizer": optimizer,
                    }
                )

        return opt_configuration

    def configure_callbacks(self):
        """
        Configure the early stopping and checkpointing callbacks
        """
        callbacks = [EarlyStopping(**self.early_stopping_kwargs)]
        if self.hpo_mode:
            callbacks.append(TuneReportCallback("val_target", on="validation_end"))
        callbacks.append(ModelCheckpoint(**self.checkpoint_kwargs))
        callbacks.append(TrainingEpochEnd())
        callbacks.append(ValidationEpochEnd())
        callbacks.append(TestEpochEnd())

        return callbacks

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
