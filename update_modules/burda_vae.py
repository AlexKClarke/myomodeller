from typing import Dict, Optional
import torch
import torch.distributions as td

from training import UpdateModule


class BurdaVariationalAutoencoder(UpdateModule):
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
        burda_samples: int = 5,
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

        self.beta = 0.0
        self.beta_step = beta_step
        self.max_beta = max_beta
        self.burda_samples = burda_samples


    def _calculate_burda_likelihood(self, x: torch.Tensor, burda_samples: int):
        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network.encode(x)
        z = self.network.sample_posterior(z_mean, z_var, burda_samples)
        recon_mean, recon_var = self.network.decode(z) # returns (batch_size, burda_samples, 1, heigth, width) for both mean and variance

        # Expand matrices for the burda reps
        z_mean = z_mean.unsqueeze(1).repeat(1, burda_samples, 1)
        z_var = z_var.unsqueeze(1).repeat(1, burda_samples, 1)
        x = x.repeat(1, burda_samples, 1, 1)

        # Create distributions
        posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
        prior_dist = td.MultivariateNormal(torch.zeros_like(z_mean), torch.ones_like(z_var).diag_embed())
        recon_dist = td.Normal(recon_mean.squeeze(), recon_var.squeeze())

        # Calculate log probabilities
        p_hi = prior_dist.log_prob(z)
        p_xGhi = recon_dist.log_prob(x).sum((-1, -2))  # Sum along the image dimensions
        q_hiGx = posterior_dist.log_prob(z)

        # Calculate the ratio
        ratio = torch.logsumexp(p_xGhi + p_hi - q_hiGx, dim=-1) + torch.log(torch.tensor(1/burda_samples)) # addition to compute the mean (log of product rule)
        loss = ratio.mean()

        return loss

    def _calculate_recon_r2(self, x: torch.Tensor):
        x_recon = self.network.forward(x)

        ssr = (x - x_recon).square().sum()
        sst = (x - x.mean()).square().sum()

        r2 = 1 - ssr.divide(sst)

        return r2.mean()

    def training_step(self, batch, batch_idx):
        #recon_loss, kld_loss = self._calculate_elbo_terms(batch[0])
        loss = self._calculate_burda_likelihood(batch[0], burda_samples=self.burda_samples)

        self.training_step_outputs.append(
            {
                "training_loss": loss,
                "beta": torch.tensor(self.beta),
            }
        )
        return loss

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
