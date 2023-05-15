"""Blocks of neural network architectures.
"""
from typing import Sequence, Union, Optional, List, Type, Tuple, Iterable, cast
import torch
import torch.nn as nn
from networks.utils import Reshape


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        out_chans_per_layer: List[int],
        use_batch_norm: bool = True,
        output_activation: Optional[str] = None,
        use_output_bias: bool = True,
    ):
        """Builds a number of linear layers

        Args:
        input_shape (Sequence[int]):
            The input shape of a data sample - [..., Feature]
            (Linear layers operate on the final dimension)
        output_shape (Sequence[int]):
            The desired output shape of the data
        out_chans_per_layer (List[int]):
            Number of hidden channels per layer, for example
            [64, 32] would give two layers of hidden channel 64 then 32
        use_batch_norm (bool, optional):
            Whether to batch norm after each linear layer
            Defaults to True.
        output_activation (Optional[str], optional):
            Allows an nn activation to be added to end of block
            Defaults to Identity.
        use_output_bias: bool = True
            Whether to use bias on output layer
        """
        super().__init__()

        self.block = nn.ModuleList()

        in_features = input_shape[-1]
        for out_features in out_chans_per_layer:
            self.block.append(nn.Linear(in_features, out_features))
            self.block.append(nn.ReLU())
            if use_batch_norm:
                self.block.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        flat_output_dim = int(torch.tensor(output_shape).prod().numpy())

        if len(out_chans_per_layer) == 0:
            out_features = in_features
        self.block.append(
            nn.Linear(out_features, flat_output_dim, bias=use_output_bias)
        )
        self.block.append(Reshape(output_shape))
        self.block.append(
            getattr(torch.nn, output_activation)()
            if output_activation
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[int]] = 3,
        stride_per_layer: Union[int, List[int]] = 1,
        use_batch_norm: bool = True,
        output_activation: Optional[str] = None,
        use_output_bias: bool = True,
    ):
        """Builds a number of conv1d layers and then flattens with linear layer
        to desired output shape

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Time]
            output_shape (Sequence[int]):
                The desired output shape of the data
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
                Whether to batch norm after each conv1d
                Defaults to True.
            output_activation: Optional[str] = None,
                Allows an nn activation to be added to end of block
                Defaults to None.
            use_output_bias: bool = True
                Whether to use bias on output layer
        """
        super().__init__()

        num_layers = len(out_chans_per_layer)

        if type(kernel_size_per_layer) == int:
            kernel_size_per_layer = [kernel_size_per_layer] * num_layers

        if type(stride_per_layer) == int:
            stride_per_layer = [stride_per_layer] * num_layers

        params_per_layer = list(
            zip(
                out_chans_per_layer,
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
            )
        )

        time_dim = input_shape[1]
        for _, kernel_size, stride in params_per_layer:
            time_dim = ((time_dim - kernel_size) // stride) + 1
        flat_conv_dim = time_dim * out_chans_per_layer[-1]

        self.block = nn.ModuleList()

        in_channel = input_shape[0]
        for out_channel, kernel_size, stride in params_per_layer:
            self.block.append(
                nn.Conv1d(in_channel, out_channel, kernel_size, stride)
            )
            self.block.append(nn.ReLU())
            if use_batch_norm:
                self.block.append(nn.BatchNorm1d(out_channel))
            in_channel = out_channel

        flat_output_dim = int(torch.tensor(output_shape).prod().numpy())

        self.block.append(nn.Flatten())
        self.block.append(
            nn.Linear(flat_conv_dim, flat_output_dim, bias=use_output_bias)
        )
        self.block.append(Reshape(output_shape))
        self.block.append(
            getattr(torch.nn, output_activation)()
            if output_activation
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        use_batch_norm: bool = True,
        output_activation: Optional[Type[nn.Module]] = None,
        use_output_bias: bool = True,
    ):
        """Builds a number of conv2d layers and then flattens with linear layer
        to desired output shape

        Args:
        input_shape (Sequence[int]):
            The input shape of the data - [Channel, Height, Width]
        output_shape (Sequence[int]):
            The desired output shape of the data
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
            Whether to batch norm after each conv2d
            Defaults to True.
        output_activation: Optional[str] = None,
            Allows an nn activation to be added to end of block
            Defaults to None.
        use_output_bias: bool = True
            Whether to use bias on output layer
        """

        super().__init__()

        num_layers = len(out_chans_per_layer)

        if type(kernel_size_per_layer) == int:
            kernel_size_per_layer = [
                (kernel_size_per_layer, kernel_size_per_layer)
            ] * num_layers

        if type(stride_per_layer) == int:
            stride_per_layer = [
                (stride_per_layer, stride_per_layer)
            ] * num_layers

        params_per_layer = list(
            zip(
                out_chans_per_layer,
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
            )
        )

        h_dim, w_dim = input_shape[1], input_shape[2]
        for _, kernel_size, stride in params_per_layer:
            h_dim = ((h_dim - kernel_size[0]) // stride[0]) + 1
            w_dim = ((w_dim - kernel_size[1]) // stride[1]) + 1
        flat_conv_dim = h_dim * w_dim * out_chans_per_layer[-1]

        self.block = nn.ModuleList()

        in_channel = input_shape[0]
        for out_channel, kernel_size, stride in params_per_layer:
            self.block.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride)
            )
            self.block.append(nn.ReLU())
            if use_batch_norm:
                self.block.append(nn.BatchNorm2d(out_channel))
            in_channel = out_channel

        flat_output_dim = int(torch.tensor(output_shape).prod().numpy())

        self.block.append(nn.Flatten())
        self.block.append(
            nn.Linear(flat_conv_dim, flat_output_dim, bias=use_output_bias)
        )
        self.block.append(Reshape(output_shape))
        self.block.append(
            getattr(torch.nn, output_activation)()
            if output_activation
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x
