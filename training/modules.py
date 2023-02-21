from typing import Dict, Optional

import torch
from training.core import UpdateModule


class SupervisedClassifier(UpdateModule):
    """A module that trains a network to classify using supervised learning,
    with accuracy as the validation target."""

    def __init__(
        self,
        network,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
    ):
        maximize_val_target = True

        super().__init__(
            network,
            maximize_val_target,
            optimizer,
            optimizer_kwargs,
            lr_scheduler_kwargs,
            early_stopping_kwargs,
        )

        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        return loss

    def training_epoch_end(self, outputs):
        average_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "training_loss", average_loss, self.current_epoch
        )

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        self.log("val_target", accuracy)
        return accuracy

    def validation_epoch_end(self, outputs):
        average_accuracy = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar(
            "validation_accuracy", average_accuracy, self.current_epoch
        )

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        return accuracy

    def test_epoch_end(self, outputs):
        average_accuracy = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar(
            "test_accuracy", average_accuracy, self.current_epoch
        )
