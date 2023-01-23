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
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
    ):
        maximize_val_target = True

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

        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        self.log("val_target", accuracy)

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        self.log("test_accuracy", accuracy)
