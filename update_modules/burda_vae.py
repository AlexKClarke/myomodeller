from typing import Dict, Optional
import torch
import torch.distributions as td

from update_modules import VariationalAutoencoder


class BurdaVariationalAutoencoder(VariationalAutoencoder):
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
        burda_samples: int = 64,
    ):
        super().__init__(
            network,
            hpo_mode,
            maximize_val_target,
            optimizer,
            optimizer_kwargs,
            lr_scheduler_kwargs,
            early_stopping_kwargs,
            starting_beta,
            beta_step,
            max_beta,
        )

        self.burda_samples = burda_samples

    def _calculate_burda_likelihood(self, x: torch.Tensor, burda_samples: int):

        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network.encode(x)
        repeat_shape = [burda_samples] + [1] * (len(z_mean.shape) - 1)
        z_mean_expanded = z_mean.repeat(repeat_shape)
        z_var_expanded = z_var.repeat(repeat_shape)

        # Sample from the posterior a number of times per data sample
        z = self.network.sample_posterior(z_mean, z_var, burda_samples)

        # Get reconstruction params for each z [batch, posterior_samples, ...]
        recon_mean, recon_var = self.network.decode(z)

        # Create distributions
        posterior_dist = td.MultivariateNormal(
            z_mean_expanded, z_var_expanded.diag_embed()
        )
        prior_dist = td.MultivariateNormal(
            torch.zeros_like(z_mean_expanded),
            torch.ones_like(z_var_expanded).diag_embed(),
        )
        recon_dist = td.Normal(recon_mean, recon_var)

        # Find the log probabilities on the latents
        posterior_log_prob = posterior_dist.log_prob(z.flatten(0, 1)).unflatten(
            0, [z.shape[0], z.shape[1]]
        )
        prior_log_prob = prior_dist.log_prob(z.flatten(0, 1)).unflatten(
            0, [z.shape[0], z.shape[1]]
        )

        # Get the log probs of the reconstructions
        repeat_shape = [1] + [burda_samples] + [1] * (len(x.shape) - 1)
        sum_dims = list(range(2, 2 + (len(x.shape) - 1)))
        recon_log_prob = recon_dist.log_prob(x.unsqueeze(1).repeat(repeat_shape)).sum(
            sum_dims
        )

        # Calculate the ratio
        ratio = (
            torch.logsumexp(recon_log_prob + prior_log_prob - posterior_log_prob, -1)
            + (torch.ones(1).type_as(x) / burda_samples).log()
        )
        loss = -ratio.mean()

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._calculate_burda_likelihood(
            batch[0], burda_samples=self.burda_samples
        )

        self.training_step_outputs.append(
            {
                "training_loss": loss,
                "beta": torch.tensor(self.starting_beta),
            }
        )
        return loss
