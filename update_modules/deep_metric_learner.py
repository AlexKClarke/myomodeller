from typing import Dict, Optional

import torch
from training import UpdateModule


class DeepMetricLearner(UpdateModule):
    """A module that trains a network to perform deep metric learning,
    with the loss as the validation target."""

    def __init__(
        self,
        network,
        num_classes: int,
        hpo_mode: bool = False,
        maximize_val_target: bool = True,
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
        margin: float = 0.1,
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

        self.loss_fn = torch.nn.MSELoss()
        self.num_classes = num_classes
        self.margin = margin

    def _calculate_loss(self, x, y_target):
        embedding = self(x)
        l2_embedding = torch.divide(
            embedding, embedding.square().sum(1, keepdim=True).sqrt()
        )
        y_target_one_hot = torch.nn.functional.one_hot(
            y_target, self.num_classes
        ).type_as(x)

        pairwise_distance = 1 - torch.matmul(l2_embedding, l2_embedding.t())

        eye = torch.eye(y_target_one_hot.shape[0]).type_as(y_target_one_hot)
        neg_labels = (
            1 - torch.matmul(y_target_one_hot, y_target_one_hot.t()) + eye
        )

        easy_positives = (pairwise_distance + 2 * neg_labels).min(
            1, keepdim=True
        )[0]
        combi = (
            easy_positives.tile([1, pairwise_distance.shape[1]])
            - pairwise_distance
            + self.margin
        )

        loss_mat = combi * neg_labels * (1 - eye)
        hinge = torch.maximum(loss_mat, torch.zeros_like(loss_mat))

        return hinge.sum() / max(1, hinge.count_nonzero())

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        return loss

    def training_epoch_end(self, outputs):
        average_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "training_loss", average_loss, self.current_epoch
        )

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        self.log("val_target", loss)
        return loss

    def validation_epoch_end(self, outputs):
        average_loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar(
            "validation_loss", average_loss, self.current_epoch
        )

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        return loss

    def test_epoch_end(self, outputs):
        average_loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar(
            "test_loss", average_loss, self.current_epoch
        )
