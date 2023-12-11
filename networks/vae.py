"""Deterministic autencoders which use a structural bottleneck

test git changes
test git changes 2
"""
from typing import Sequence, Union, Optional, List, Type, Tuple
import torch
import torch.nn as nn
from networks.blocks import (
    Conv1dBlock,
    Conv2dBlock,
    MLPBlock,
)
from networks.transpose import (
    ConvTranspose1dBlock,
    ConvTranspose2dBlock,
)


class MLPVariaionalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
        num_sampling_draws: int,
        use_batch_norm: bool = True,
        encoder_output_activation: Optional[Type[nn.Module]] = None,
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
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to Identity.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = MLPBlock(
            input_shape=input_shape,
            output_shape=[latent_dim * 2],
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=encoder_output_activation,
            use_output_bias=False,
        )

        out_chans_per_layer = out_chans_per_layer[::-1]

        self.decoder = MLPBlock(
            input_shape=[num_sampling_draws * latent_dim], # 1D, afe flattening the 2D output of the sampling function
            output_shape=output_shape,
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=encoder_output_activation,
        )

    def sampling(self, z: torch.Tensor) -> torch.Tensor:
        # retrieve mean and std
        z_mu = z[:self.latent_dim]
        z_std = z[self.latent_dim:]

        # sampling from normal distribution
        eps = torch.randn(self.num_sampling_draws, self.latent_dim)
        z_mu_extended = z_mu.repeat(self.num_sampling_draws, 1)
        z_std_extended = z_std.repeat(self.num_sampling_draws, 1)

        # Returns a 2D tensor, of size (num_draws, latent_dim)
        return eps * z_std_extended + z_mu_extended

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.sampling(self.encoder(x))

        # z, initially 2D, is flattened in order to be given as input to the decoder
        return self.decoder(z.flatten())

    def return_sparse_weights(self) -> torch.Tensor:
        return self.encoder.block[-3].weight


class Conv1dVariaionalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        latent_dim: int,
        out_chans_per_layer: List[int],
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

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv1dBlock(
            input_shape=input_shape,
            output_shape=[latent_dim],
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
            input_shape=[latent_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def return_sparse_weights(self) -> torch.Tensor:
        return self.encoder.block[-3].weight


class Conv2dVariaionalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        sparse_dim: int,
        out_chans_per_layer: List[int],
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
            sparse_dim (int):
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

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.encoder = Conv2dBlock(
            input_shape=input_shape,
            output_shape=[sparse_dim],
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
            input_shape=[sparse_dim],
            output_shape=output_shape,
            linear_out_chan=linear_out_chan,
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def return_sparse_weights(self) -> torch.Tensor:
        return self.encoder.block[-3].weight
