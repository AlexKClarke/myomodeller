from typing import Dict, Optional
import torch
import torch.distributions as td

from training import UpdateModule


class IOVariationalAutoencoder(UpdateModule):
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
        n_steps_per_switch: int = 5,
        n_samples_in_aux: int = 16,
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
        self.n_steps_per_switch = n_steps_per_switch
        self.n_samples_in_aux = n_samples_in_aux

    def _calculate_elbo_terms(self, x: torch.Tensor):
        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network[0].encode(x)
        z = self.network[0].sample_posterior(z_mean, z_var)
        recon_mean, recon_var = self.network[0].decode(z)

        # Create distributions
        posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
        prior_dist = td.MultivariateNormal(
            torch.zeros_like(z_mean), torch.ones_like(z_var).diag_embed()
        )
        recon_dist = td.Normal(recon_mean, recon_var)

        # Get the KL divergence between posterior and prior
        kld_loss = td.kl_divergence(posterior_dist, prior_dist).mean()

        # Get the log probability of x given the reconstruction distribution
        # sum along pixel axis, mean across batch axis
        recon_loss = recon_dist.log_prob(x).sum(1).mean()

        return -recon_loss, kld_loss

    def _calculate_elbo_terms_io(self, x: torch.Tensor):
        # Get the posterior and reconstuction parameters
        z_mean, z_var = self.network[0].encode(x)
        z = self.network[0].sample_posterior(z_mean, z_var)
        recon_mean, recon_var = self.network[0].decode(z)

        # Create distributions
        posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
        prior_dist = td.MultivariateNormal(
            torch.zeros_like(z_mean), torch.ones_like(z_var).diag_embed()
        )
        recon_dist = td.Normal(recon_mean, recon_var)

        # Get the KL divergence between posterior and prior
        kld_loss = td.kl_divergence(posterior_dist, prior_dist).mean()

        # Get the log probability of x given the reconstruction distribution
        # 0. compute log probabilities
        # 1. sum along the 'pixel axis'
        # 2. add the discriminator output
        # 3. take the mean across the batch axis
        regularised_recon_loss = (recon_dist.log_prob(x).sum(1) + self.network[1](z.reshape([-1, z_mean.shape[-1]]))).mean()



        return -regularised_recon_loss, kld_loss

    def _calculate_auxiliary_loss(self, x: torch.Tensor):
        with torch.no_grad():
            z_mean, z_var = self.network[0].encode(x)

            # Create normal distribution with 0-mean and unit-variance and posterior distribution with mean and variance resulting from the encoder
            posterior_dist = td.MultivariateNormal(z_mean, z_var.diag_embed())
            prior_dist = td.MultivariateNormal(
                torch.zeros_like(z_mean), torch.ones_like(z_var).diag_embed()
            )

            # sample N samples from both distributions
            post_sample = posterior_dist.sample([self.n_samples_in_aux]).reshape(
                [-1, z_mean.shape[-1]]
            )
            prior_sample = prior_dist.sample([self.n_samples_in_aux]).reshape(
                [-1, z_mean.shape[-1]]
            )

        # make predictions using the discriminator network (network[1])
        post_pred = self.network[1](post_sample)
        prior_pred = self.network[1](prior_sample)

        # return loss (binary cross entropy) between the predictions, and the true values (labelled 0 and 1 depending on the distribution)
        return torch.nn.functional.binary_cross_entropy(
            torch.concat([post_pred, prior_pred]),
            torch.concat([torch.ones_like(post_pred), torch.zeros_like(prior_pred)]),
        )

    def _calculate_recon_r2(self, x: torch.Tensor):
        x_recon = self.network[0].forward(x)

        ssr = (x - x_recon).square().sum()
        sst = (x - x.mean()).square().sum()

        r2 = 1 - ssr.divide(sst)

        return r2.mean()

    def training_step(self, batch, batch_idx):

        # n_steps_per_switch determines how many times the discriminator network is updates wrt. the vae network
        if self.trainer.global_step % self.n_steps_per_switch != 0:
            #--> here, the main vae is updated

            opt = self.optimizers()
            opt[0].zero_grad()

            #recon_loss, kld_loss = self._calculate_elbo_terms(batch[0])
            recon_loss, kld_loss = self._calculate_elbo_terms(batch[0])
            elbo = recon_loss + self.beta * kld_loss

            self.manual_backward(elbo)
            opt[0].step()

            self.training_step_outputs.append(
                {
                    "training_elbo": elbo,
                    "training_recon": recon_loss,
                    "training_kld": kld_loss,
                    "beta": torch.tensor(self.beta),
                }
            )
        else:
            #--> here, the discriminator network is updated

            opt = self.optimizers()
            opt[1].zero_grad()

            aux_loss = self._calculate_auxiliary_loss(batch[0])

            self.manual_backward(aux_loss)
            opt[1].step()

            self.training_step_outputs.append(
                {
                    "aux_loss": aux_loss,
                }
            )

    def on_train_epoch_end(self):
        # gradually increase beta until max_beta

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
