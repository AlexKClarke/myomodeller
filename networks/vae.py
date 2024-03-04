"""Variational autencoders architectures
"""

from typing import Sequence, Union, List, Tuple
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


class MLPVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        use_batch_norm: bool = True,
        fix_recon_var: bool = False,
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
            fix_recon_var (bool, optional):
                Whether to fix reconstruction to a unit variance
                Defaults to False.
        """
        super().__init__()

        self.output_shape = output_shape

        if fix_recon_var:
            self.recon_log_var = torch.zeros(output_shape)
        else:
            self.recon_log_var = nn.Parameter(torch.zeros(output_shape))

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = MLPBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * latent_dim * 2], # changed to account for multivariate mean and variance
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
        )

        self.decoder = MLPBlock(
            input_shape=[latent_dim * latent_dim],
            output_shape=output_shape,
            out_chans_per_layer=out_chans_per_layer[::-1],
            use_batch_norm=use_batch_norm,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a posterior mean and variance for a given input"""

        params = self.encoder(x)
        # Extract mean and log variance
        z_mean, z_log_cov = params.split(params.shape[-1] // 2, dim=-1)

        return z_mean, z_log_cov.exp()

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a reconstruction mean and variance for a given input

        Assumes input shape of [batch, channels] or
        [batch, num posterior draws, channels]
        """

        # Add a num post draws dim if missing, retain z shape and flatten
        # the batch and draws dim for the encoder
        squeeze_z = False
        if z.dim() == 2:
            squeeze_z = True
            z = z.unsqueeze(1)
        z_shape = z.shape
        z = z.reshape((z_shape[0] * z_shape[1], z_shape[2]))

        # Get the reconstruction parameters
        recon_mean = self.decoder(z)
        recon_var = (
            self.recon_log_var.exp().unsqueeze(0).tile((recon_mean.shape[0], 1, 1, 1))
        )

        # Need to unflatten the batch and draws and then remove draws if 2D
        recon_mean = recon_mean.reshape((z_shape[0], z_shape[1]) + self.output_shape)
        recon_var = recon_var.reshape((z_shape[0], z_shape[1]) + self.output_shape)

        recon_mean = recon_mean.squeeze(1) if squeeze_z else recon_mean
        recon_var = recon_var.squeeze(1) if squeeze_z else recon_var

        return recon_mean, recon_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input for testing using mean of posterior and recon"""

        z = self.encode(x)[0]
        return self.decode(z)[0]

    def sample_posterior(
        self, z_mean: torch.Tensor, z_var: torch.Tensor, num_draws: int = 1
    ) -> torch.Tensor:
        z_old = td.Normal(z_mean, z_var).rsample((num_draws,)).transpose(0, 1)
        z = td.MultivariateNormal(z_mean, z_var).rsample((num_draws,)).transpose(0, 1)
        return z.squeeze(1) if z.shape[1] == 1 else z


class Conv1dVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[int]] = 3,
        stride_per_layer: Union[int, List[int]] = 1,
        use_batch_norm: bool = True,
        fix_recon_var: bool = False,
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
            fix_recon_var (bool, optional):
                Whether to fix reconstruction to a unit variance
                Defaults to False.
        """
        super().__init__()

        self.output_shape = output_shape

        if fix_recon_var:
            self.recon_log_var = torch.zeros(output_shape)
        else:
            self.recon_log_var = nn.Parameter(torch.zeros(output_shape))

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv1dBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

        linear_out_chan = out_chans_per_layer[-1]
        out_chans_per_layer = out_chans_per_layer[::-1][1:] + [input_shape[0]]
        if type(kernel_size_per_layer) == list:
            kernel_size_per_layer = kernel_size_per_layer[::-1]
        if type(stride_per_layer) == list:
            stride_per_layer = stride_per_layer[::-1]

        self.decoder = ConvTranspose1dBlock(
            input_shape=[latent_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a posterior mean and variance for a given input"""

        params = self.encoder(x)
        z_mean, z_log_cov = params.split(params.shape[-1] // 2, dim=-1)
        return z_mean, z_log_cov.exp()

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a reconstruction mean and variance for a given input

        Assumes input shape of [batch, channels] or
        [batch, num posterior draws, channels]
        """

        # Add a num post draws dim if missing, retain z shape and flatten
        # the batch and draws dim for the encoder
        squeeze_z = False
        if z.dim() == 2:
            squeeze_z = True
            z = z.unsqueeze(1)
        z_shape = z.shape
        z = z.reshape((z_shape[0] * z_shape[1], z_shape[2]))

        # Get the reconstruction parameters
        recon_mean = self.decoder(z)
        recon_var = (
            self.recon_log_var.exp().unsqueeze(0).tile((recon_mean.shape[0], 1, 1, 1))
        )

        # Need to unflatten the batch and draws and then remove draws if 2D
        recon_mean = recon_mean.reshape((z_shape[0], z_shape[1]) + self.output_shape)
        recon_var = recon_var.reshape((z_shape[0], z_shape[1]) + self.output_shape)

        recon_mean = recon_mean.squeeze(1) if squeeze_z else recon_mean
        recon_var = recon_var.squeeze(1) if squeeze_z else recon_var

        return recon_mean, recon_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input for testing using mean of posterior and recon"""

        z = self.encode(x)[0]
        return self.decode(z)[0]

    def sample_posterior(
        self, z_mean: torch.Tensor, z_var: torch.Tensor, num_draws: int = 1
    ) -> torch.Tensor:
        z = td.Normal(z_mean, z_var).rsample((num_draws,)).transpose(0, 1)
        return z.squeeze(1) if z.shape[1] == 1 else z


class Conv2dVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        use_batch_norm: bool = True,
        fix_recon_var: bool = False,
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
            fix_recon_var (bool, optional):
                Whether to fix reconstruction to a unit variance
                Defaults to False.
        """
        super().__init__()

        self.output_shape = output_shape

        if fix_recon_var:
            self.recon_log_var = torch.zeros(output_shape)
        else:
            self.recon_log_var = nn.Parameter(torch.zeros(output_shape))

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv2dBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

        linear_out_chan = out_chans_per_layer[-1]
        out_chans_per_layer = out_chans_per_layer[::-1][1:] + [input_shape[0]]
        if type(kernel_size_per_layer) == list:
            kernel_size_per_layer = kernel_size_per_layer[::-1]
        if type(stride_per_layer) == list:
            stride_per_layer = stride_per_layer[::-1]

        self.decoder = ConvTranspose2dBlock(
            input_shape=[latent_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a posterior mean and variance for a given input"""

        params = self.encoder(x)
        z_mean, z_log_cov = params.split(params.shape[-1] // 2, dim=-1)
        return z_mean, z_log_cov.exp()

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a reconstruction mean and variance for a given input

        Assumes input shape of [batch, channels] or
        [batch, num posterior draws, channels]
        """

        # Add a num post draws dim if missing, retain z shape and flatten
        # the batch and draws dim for the encoder
        squeeze_z = False
        if z.dim() == 2:
            squeeze_z = True
            z = z.unsqueeze(1)
        z_shape = z.shape
        z = z.reshape((z_shape[0] * z_shape[1], z_shape[2]))

        # Get the reconstruction parameters
        recon_mean = self.decoder(z)
        recon_var = (
            self.recon_log_var.exp().unsqueeze(0).tile((recon_mean.shape[0], 1, 1, 1))
        )

        # Need to unflatten the batch and draws and then remove draws if 2D
        recon_mean = recon_mean.reshape((z_shape[0], z_shape[1]) + self.output_shape)
        recon_var = recon_var.reshape((z_shape[0], z_shape[1]) + self.output_shape)

        recon_mean = recon_mean.squeeze(1) if squeeze_z else recon_mean
        recon_var = recon_var.squeeze(1) if squeeze_z else recon_var

        return recon_mean, recon_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs input for testing using mean of posterior and recon"""

        z = self.encode(x)[0]
        return self.decode(z)[0]

    def sample_posterior(
        self, z_mean: torch.Tensor, z_var: torch.Tensor, num_draws: int = 1
    ) -> torch.Tensor:
        z = td.Normal(z_mean, z_var).rsample((num_draws,)).transpose(0, 1)
        return z.squeeze(1) if z.shape[1] == 1 else z
