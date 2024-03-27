from typing import Dict, Optional
import torch
import torch.distributions as td

from training import UpdateModule


class VariationalAutoencoder(UpdateModule):
    """A module that trains a variational autoencoder using a beta ELBO"""

    def __init__(
        self,
        network,
        hpo_mode: bool = False,
        maximize_val_target: bool = True,
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping_kwargs: Optional[Dict] = None,
        starting_beta: float = 0.0,
        beta_step: float = 1e-2,
        max_beta: float = 1.0,
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

        self.starting_beta = starting_beta
        self.beta_step = beta_step
        self.max_beta = max_beta

    def _calculate_elbo_terms(self, x: torch.Tensor):
        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network.encode(x)
        z = self.network.sample_posterior(z_mean, z_var)
        recon_mean, recon_var = self.network.decode(z)

        # Create distributions
        posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
        eye = (
            torch.eye(z_var.shape[1], dtype=z_var.dtype, device=z_var.device)
            .unsqueeze(0)
            .repeat(z_var.shape[0], 1, 1)
        )
        prior_dist = td.MultivariateNormal(torch.zeros_like(z_mean), eye)
        recon_dist = td.Normal(recon_mean, recon_var)

        # Get the KL divergence between posterior and prior
        kld_loss = td.kl_divergence(posterior_dist, prior_dist).mean()

        # Get the log probability of x given the reconstruction distribution
        recon_loss = recon_dist.log_prob(x).sum(1).mean()

        return -recon_loss, kld_loss

    def _calculate_recon_r2(self, x: torch.Tensor):
        x_recon = self.network.forward(x)

        ssr = (x - x_recon).square().sum()
        sst = (x - x.mean()).square().sum()

        r2 = 1 - ssr.divide(sst)

        return r2.mean()

    def training_step(self, batch, batch_idx):
        recon_loss, kld_loss = self._calculate_elbo_terms(batch[0])
        elbo = recon_loss + self.starting_beta * kld_loss

        self.training_step_outputs.append(
            {
                "training_elbo": elbo,
                "training_recon": recon_loss,
                "training_kld": kld_loss,
                "beta": torch.tensor(self.starting_beta),
            }
        )
        return elbo

    def on_train_epoch_end(self):
        if self.starting_beta < self.max_beta:
            self.starting_beta += self.beta_step
        else:
            self.starting_beta = self.max_beta

    def validation_step(self, batch, batch_idx):
        r2 = self._calculate_recon_r2(batch[0])
        self.log("val_target", r2)
        self.validation_step_outputs.append({"validation_r2": r2})
        return r2

    def test_step(self, batch, batch_idx):
        r2 = self._calculate_recon_r2(batch[0])
        self.test_step_outputs.append({"test_r2": r2})
        return r2
