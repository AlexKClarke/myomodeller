from typing import Dict, Optional

import torch
from training import UpdateModule


class SupervisedClassifier(UpdateModule):
    """A module that trains a network to classify using supervised learning,
    with accuracy as the validation target."""

    def __init__(
        self,
        network,
        hpo_mode: bool = False,
        maximize_val_target: bool = True,
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            network,
            hpo_mode,
            maximize_val_target,
            optimizer,
            optimizer_kwargs,
            lr_scheduler_kwargs,
            early_stopping_kwargs,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        self.training_step_outputs.append({"training_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        self.log("val_target", accuracy)
        self.validation_step_outputs.append({"val_accuracy": accuracy})
        return accuracy

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        accuracy = (y_predict.argmax(dim=-1) == y_target).float().mean()
        self.test_step_outputs.append({"test_accuracy": accuracy})
        return accuracy
