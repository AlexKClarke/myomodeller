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

        self.beta = 0.5 # beta initialised
        self.beta_step = beta_step # added at the end of each epoch
        self.max_beta = max_beta

    def _calculate_elbo_terms(self, x: torch.Tensor):

        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network.encode(x)

        #todo: DOES THIS MAKE SENSE?
        # it adds a small epsilon to any element that is not greater than 0
        # issue: the error is not always solved
        '''min_value = z_var.min().item()
        epsilon = min_value
        if epsilon > 1e-18:
            epsilon = 1e-18
        # Add 1/10 of the smallest element to any element not greater than 0
        z_var = torch.where(z_var > 0, z_var, z_var + epsilon)'''
        ######################

        z_var = z_var.clamp(min=1e-36)

        z = self.network.sample_posterior(z_mean, z_var)
        recon_mean, recon_var = self.network.decode(z)

        # Create distributions
        posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
        prior_dist = td.MultivariateNormal(
            torch.zeros_like(z_mean), torch.ones_like(z_var).diag_embed()
        )
        recon_dist = td.Normal(recon_mean, recon_var)

        # Get the KL divergence between posterior and prior
        kld_loss = td.kl_divergence(posterior_dist, prior_dist).mean()

        # Get the log probability of x given the reconstruction distribution
        recon_loss = recon_dist.log_prob(x).sum(1).mean() # mean represents the expectation over the samples

        return -recon_loss, kld_loss

    def _calculate_recon_r2(self, x: torch.Tensor):
        x_recon = self.network.forward(x)

        ssr = (x - x_recon).square().sum()
        sst = (x - x.mean()).square().sum()

        r2 = 1 - ssr.divide(sst)

        return r2.mean()

    def training_step(self, batch, batch_idx):
        recon_loss, kld_loss = self._calculate_elbo_terms(batch[0])
        elbo = recon_loss + self.beta * kld_loss
        self.training_step_outputs.append(
            {
                "training_elbo": elbo,
                "training_recon": recon_loss,
                "training_kld": kld_loss,
                "beta": torch.tensor(self.beta),
            }
        )
        return elbo

    def on_train_epoch_end(self):
        if self.beta < self.max_beta:
            self.beta += self.beta_step
        else:
            self.beta = self.max_beta



    def validation_step(self, batch, batch_idx):
        r2 = self._calculate_recon_r2(batch[0])
        self.log("val_target", r2)
        self.validation_step_outputs.append({"validation_r2": r2})
        return r2

    def test_step(self, batch, batch_idx):
        r2 = self._calculate_recon_r2(batch[0])
        self.test_step_outputs.append({"test_r2": r2})
        return r2
