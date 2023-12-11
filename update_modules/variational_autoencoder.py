from typing import Dict, Optional
import torch
from training import UpdateModule


class VariationalAutoencoder(UpdateModule):
    """A module that trains a network to autoencode. The network must have
    a "return_sparse_weights" method which returns the weights on the
    bottleneck of the sparse autoencoder."""

    def __init__(
        self,
        network,
        hpo_mode: bool = False,
        maximize_val_target: bool = True,
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
        l1_loss_coeff: float = 0.1,
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
        self.vae_rec_loss = torch.nn.BCELoss()
        self.lamb = l1_loss_coeff

    def _calculate_loss(self, x, y_target):
        # y_target = x
        y_predict, mu, std = self(x)

        recon_loss = self.vae_rec_loss(y_predict, y_target, reduction='sum')
        KL_div = -0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2)

        '''sparse_weights = self.network.return_sparse_weights()
        l1_loss = sparse_weights.abs().mean()'''

        #return recon_loss + self.lamb * l1_loss
        return recon_loss + KL_div

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        self.training_step_outputs.append({"training_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        self.log("val_target", loss)
        self.validation_step_outputs.append({"validation_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        loss = self._calculate_loss(x, y_target)
        self.test_step_outputs.append({"test_loss": loss})
        return loss
