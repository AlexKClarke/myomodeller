"""Deterministic autencoders which use a structural bottleneck

test git changes
test git changes 2
"""
from typing import Sequence, Union, Optional, List, Type, Tuple
import torch
import torch.nn as nn
import torch.distributions as td

from networks.blocks import (
    Conv1dBlock,
    Conv2dBlock,
    MLPBlock,
)
from networks.transpose import (
    ConvTranspose1dBlock,
    ConvTranspose2dBlock,
)

# print(torch.cuda.is_available())
"""if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")"""


class MLPVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        num_sampling_draws: int,
        use_batch_norm: bool = True,
        fix_recon_cov: bool = False,
    ):
        """An MLP-based sparse autoencoder

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [..., Feature]
                (Linear layers operate on the final dimension)
            output_shape (Sequence[int]):
                The output shape of the data - [..., Feature]
            latent_dim (int):
                The dimension of the sparse layer
            num_sampling_draws (int):
                Number of draws performed during sampling (reparametrization trick), for each data sample
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [64, 32] would give two layers of hidden channel 64 then 32
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            fix_recon_cov (bool, optional):
                Whether to fix reconstruction to a unit covariance
                Defaults to False.
        """
        super().__init__()

        self.prior = td.MultivariateNormal(
            torch.zeros(latent_dim), torch.eye(latent_dim)
        )
        self.posterior = td.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.recon = td.Normal(
            torch.zeros(output_shape[-1]), torch.ones(output_shape[-1])
        )

        if fix_recon_cov:
            self.recon_cov = torch.ones(output_shape[-1])
        else:
            self.recon_cov = nn.Parameter(torch.ones(output_shape[-1]))

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = MLPBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=None,
            use_output_bias=False,
        )

        out_chans_per_layer = out_chans_per_layer[::-1]

        self.decoder = MLPBlock(
            input_shape=[
                num_sampling_draws * latent_dim
            ],  # 1D, afte flattening the 2D output of the sampling function
            output_shape=output_shape,
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=None,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a posterior mean and covariance for a given input"""

        params = self.encoder(x)
        z_mean, z_log_cov = params.split(params.shape[-1] // 2, dim=-1)
        return z_mean, z_log_cov.exp()

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a reconstruction mean and covariance for a given input"""

        x_mean = self.decoder(z)
        return x_mean, self.recon_cov

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input for testing"""

        z = td.Normal(*self.encode(x)).sample()
        return self.decode(z)[0]

    def sample_posterior(
        self, z_mean: torch.Tensor, z_cov: torch.Tensor, num_draws: int = 1
    ) -> torch.Tensor:
        z = td.Normal(z_mean, z_cov).rsample((num_draws,))
        return z

    def sampling(self, latent_dim, num_sampling_draws, z: torch.Tensor) -> torch.Tensor:
        # mean and std of new sample
        self.distribution_mean = z[:latent_dim]
        self.covariance_matrix = torch.diag(z[latent_dim:])  # diagonal cov matrix
        # update distribution
        self.multivariate_normal = MultivariateNormal(
            self.distribution_mean, self.covariance_matrix
        )

        # Returns a 2D tensor, of size (num_draws, latent_dim)
        return (
            self.multivariate_normal.rsample((num_sampling_draws,)),
            self.distribution_mean,
            self.covariance_matrix,
        )


class Conv1dVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        num_sampling_draws: int,
        kernel_size_per_layer: Union[int, List[int]] = 3,
        stride_per_layer: Union[int, List[int]] = 1,
        use_batch_norm: bool = True,
        encoder_output_activation: Optional[Type[nn.Module]] = None,
    ):
        """A Conv1d-based sparse autoencoder

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Time]
            output_shape (Sequence[int]):
                The output shape of the data - [Channel, Time]
            latent_dim (int):
                The dimension of the sparse layer
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [64, 32] would give two layers of hidden channel 64 then 32
            kernel_size_per_layer (Union[int, List[int]], optional):
                Size of the conv1d kernel
                Can be set as a list of same length as kernel_size_per_layer
                or as an int (in which case the same value will be used for each layer)
                Defaults to 5.
            stride_per_layer (Union[int, List[int]], optional):
                Size of the conv1d stride
                Can be set as a list of same length as kernel_size_per_layer
                or as an int (in which case the same value will be used for each layer)
                Defaults to 1.
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to identity.
        """
        super().__init__()

        # CLASS INSTANCES for mean, cov matrix and poserior normal distribution
        self.distribution_mean = torch.zeros(latent_dim)
        self.covariance_matrix = torch.eye(latent_dim)
        self.multivariate_normal = MultivariateNormal(
            self.distribution_mean, self.covariance_matrix
        )

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv1dBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=encoder_output_activation,
            use_output_bias=False,
        )

        linear_out_chan = out_chans_per_layer[-1]
        out_chans_per_layer = out_chans_per_layer[::-1][1:] + [input_shape[0]]
        if type(kernel_size_per_layer) == list:
            kernel_size_per_layer = kernel_size_per_layer[::-1]
        if type(stride_per_layer) == list:
            stride_per_layer = stride_per_layer[::-1]

        self.decoder = ConvTranspose1dBlock(
            input_shape=[num_sampling_draws * latent_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def sampling(self, latent_dim, num_sampling_draws, z: torch.Tensor) -> torch.Tensor:
        # mean and std of new sample
        self.distribution_mean = z[:latent_dim]
        self.covariance_matrix = torch.diag(z[latent_dim:])  # diagonal cov matrix
        # update distribution
        self.multivariate_normal = MultivariateNormal(
            self.distribution_mean, self.covariance_matrix
        )

        # Returns a 2D tensor, of size (num_draws, latent_dim)
        return (
            self.multivariate_normal.rsample((num_sampling_draws,)),
            self.distribution_mean,
            self.covariance_matrix,
        )

    def forward(self, latent_dim, num_sampling_draws, x: torch.Tensor) -> torch.Tensor:
        z, mu, std = self.sampling(latent_dim, num_sampling_draws, self.encoder(x))
        return self.decoder(z.flatten()), mu, std

    def return_sparse_weights(self) -> torch.Tensor:
        return self.encoder.block[-3].weight


class Conv2dVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        num_sampling_draws: int,
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        use_batch_norm: bool = True,
        encoder_output_activation: Optional[Type[nn.Module]] = None,
    ):
        """A Conv2d-based sparse autoencoder

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Height, Width]
            output_shape (Sequence[int]):
                The output shape of the data - [Channel, Height, Width]
            latent_dim (int):
                The dimension of the sparse layer
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [64, 32] would give two layers of hidden channel 64 then 32
            kernel_size_per_layer (Union[int, List[int]], optional):
                Size of the conv2d kernel (height and width)
                Can be set as a list of (H, W) tuples of same length as kernel_size_per_layer
                or as an int (in which case the same value will be used for each layer)
                Defaults to 5.
            stride_per_layer (Union[int, List[int]], optional):
                Size of the conv2d stride (height and width)
                Can be set as a list of (H, W) tuples of same length as kernel_size_per_layer
                or as an int (in which case the same value will be used for each layer)
                Defaults to 1.
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to identity
        """
        super().__init__()

        # CLASS INSTANCES for mean, cov matrix and poserior normal distribution
        self.distribution_mean = torch.zeros(latent_dim)
        self.covariance_matrix = torch.eye(latent_dim)
        self.multivariate_normal = MultivariateNormal(
            self.distribution_mean, self.covariance_matrix
        )

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv2dBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=encoder_output_activation,
            use_output_bias=False,
        )

        linear_out_chan = out_chans_per_layer[-1]
        out_chans_per_layer = out_chans_per_layer[::-1][1:] + [input_shape[0]]
        if type(kernel_size_per_layer) == list:
            kernel_size_per_layer = kernel_size_per_layer[::-1]
        if type(stride_per_layer) == list:
            stride_per_layer = stride_per_layer[::-1]

        self.decoder = ConvTranspose2dBlock(
            input_shape=[num_sampling_draws * latent_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def sampling(self, latent_dim, num_sampling_draws, z: torch.Tensor) -> torch.Tensor:
        # mean and std of new sample
        self.distribution_mean = z[:latent_dim]
        self.covariance_matrix = torch.diag(z[latent_dim:])  # diagonal cov matrix
        # update distribution
        self.multivariate_normal = MultivariateNormal(
            self.distribution_mean, self.covariance_matrix
        )

        # Returns a 2D tensor, of size (num_draws, latent_dim)
        return (
            self.multivariate_normal.rsample((num_sampling_draws,)),
            self.distribution_mean,
            self.covariance_matrix,
        )

    def forward(self, latent_dim, num_sampling_draws, x: torch.Tensor) -> torch.Tensor:
        z, mu, std = self.sampling(latent_dim, num_sampling_draws, self.encoder(x))
        return self.decoder(z.flatten()), mu, std

    def return_sparse_weights(self) -> torch.Tensor:
        return self.encoder.block[-3].weight
