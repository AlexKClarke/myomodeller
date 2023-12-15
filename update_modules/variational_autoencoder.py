from typing import Dict, Optional
import torch
from training import UpdateModule
from torch.distributions import kl_divergence, Normal


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

        # 1. Encode the input
        z_mean, z_log_exp_cov = self.network.encode(x)
        # 2. Generate posterior distribution
        posterior_distribution = Normal(z_mean, z_log_exp_cov)

        # 3. Create prior normal distribution (0 mean unit var)
        prior_normal_distribution = Normal(
            torch.zeros_like(z_mean), torch.ones_like(z_log_exp_cov)
        )

        # 4. Forward pass to obtain reconstructed output
        y_predict = self.network.forward(x)

        # 5. Compute loss terms
        recon_loss = self.loss_fn(y_predict, y_target) #todo Change with BCELoss
        kl_div = kl_divergence(posterior_distribution, prior_normal_distribution)

        return recon_loss + kl_div.mean() #todo is taking the mean appropriate? i get error that loss is not scalar otherwise (grad computation impossible)

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
