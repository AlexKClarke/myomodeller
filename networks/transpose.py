from typing import Sequence, Union, List, Tuple, Iterable, cast
import torch
import torch.nn as nn
from networks.utils import Reshape


class ConvTranspose1dBlock(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        linear_out_chan: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[int]] = 3,
        stride_per_layer: Union[int, List[int]] = 1,
        use_batch_norm: bool = True,
    ):
        """Complement class to Conv1d block that reverses its
        operations with transpose layers.

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Time]
            output_shape (Sequence[int]):
                The output shape of the data
            linear_out_chan (int)
                The output channel size of the linear layer. As the Conv1DBlock
                has a linear layer at the end, we need the output
                shape if the time dimension of the input is to match that of
                the output. This is usually the size of the final conv1d out chan
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [32, 64] would give two layers of hidden channel 32 then 64
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
            output_activation (Optional[Type[nn.Module]], optional):
                Allows an nn activation to be added to end of block
                Defaults to None.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        num_layers = len(out_chans_per_layer)

        if type(kernel_size_per_layer) == int:
            kernel_size_per_layer = [kernel_size_per_layer] * num_layers

        if type(stride_per_layer) == int:
            stride_per_layer = [stride_per_layer] * num_layers

        kern_stride_per_layer = list(
            zip(
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
            )
        )

        time_dim = [output_shape[1]]
        for kernel_size, stride in reversed(kern_stride_per_layer):
            time_dim.append(((time_dim[-1] - kernel_size) // stride) + 1)

        time_dim = time_dim[::-1]
        output_padding_per_layer = []
        for i, [kernel_size, stride] in enumerate(kern_stride_per_layer):
            current_time_dim = ((time_dim[i] - 1) * stride) + kernel_size
            target_time_dim = time_dim[i + 1]
            output_padding_per_layer.append(target_time_dim - current_time_dim)

        params_per_layer = list(
            zip(
                out_chans_per_layer,
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
                cast(Iterable, output_padding_per_layer),
            )
        )

        flat_linear_out_dim = time_dim[0] * linear_out_chan
        flat_output_dim = int(torch.tensor(input_shape).prod().numpy())

        self.block = nn.ModuleList()

        self.block.append(nn.Flatten())
        self.block.append(nn.Linear(flat_output_dim, flat_linear_out_dim))
        self.block.append(nn.ReLU())
        self.block.append(Reshape((linear_out_chan, time_dim[0])))

        in_channel = linear_out_chan
        for out_channel, kernel_size, stride, out_padding in params_per_layer:
            if use_batch_norm:
                self.block.append(nn.BatchNorm1d(in_channel))
            self.block.append(
                nn.ConvTranspose1d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    output_padding=out_padding,
                )
            )
            self.block.append(nn.ReLU())
            in_channel = out_channel

        self.block = self.block[:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        linear_out_chan: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        use_batch_norm: bool = True,
    ):
        """Complement class to Conv2d block that reverses its
        operations with transpose layers.

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Height, Width]
            output_shape (Sequence[int]):
                The output shape of the data - [Channel, Height, Width]
            linear_out_chan (int)
                The output channel size of the linear layer. As the Conv2DBlock
                has a linear layer at the end, we need the output
                shape if the time dimension of the input is to match that of
                the output. This is usually the size of the final conv2d out chan
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [32, 64] would give two layers of hidden channel 32 then 64
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
            output_activation (Optional[Type[nn.Module]], optional):
                Allows an nn activation to be added to end of block
                Defaults to None.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        num_layers = len(out_chans_per_layer)

        if type(kernel_size_per_layer) == int:
            kernel_size_per_layer = [
                (kernel_size_per_layer, kernel_size_per_layer)
            ] * num_layers

        if type(stride_per_layer) == int:
            stride_per_layer = [
                (stride_per_layer, stride_per_layer)
            ] * num_layers

        kern_stride_per_layer = list(
            zip(
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
            )
        )

        h_dim, w_dim = [output_shape[1]], [output_shape[2]]
        for kernel_size, stride in reversed(kern_stride_per_layer):
            h_dim.append(((h_dim[-1] - kernel_size[0]) // stride[0]) + 1)
            w_dim.append(((w_dim[-1] - kernel_size[1]) // stride[1]) + 1)

        h_dim, w_dim = h_dim[::-1], w_dim[::-1]
        output_padding_per_layer = []
        for i, [kernel_size, stride] in enumerate(kern_stride_per_layer):
            current_h_dim = ((h_dim[i] - 1) * stride[0]) + kernel_size[0]
            target_h_dim = h_dim[i + 1]
            current_w_dim = ((w_dim[i] - 1) * stride[1]) + kernel_size[1]
            target_w_dim = w_dim[i + 1]
            output_padding_per_layer.append(
                [
                    (target_h_dim - current_h_dim),
                    (target_w_dim - current_w_dim),
                ]
            )

        params_per_layer = list(
            zip(
                out_chans_per_layer,
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
                cast(Iterable, output_padding_per_layer),
            )
        )

        flat_linear_out_dim = h_dim[0] * w_dim[0] * linear_out_chan
        flat_output_dim = int(torch.tensor(input_shape).prod().numpy())

        self.block = nn.ModuleList()

        self.block.append(nn.Flatten())
        self.block.append(nn.Linear(flat_output_dim, flat_linear_out_dim))
        self.block.append(nn.ReLU())
        self.block.append(Reshape((linear_out_chan, h_dim[0], w_dim[0])))

        in_channel = linear_out_chan
        for out_channel, kernel_size, stride, out_padding in params_per_layer:
            if use_batch_norm:
                self.block.append(nn.BatchNorm2d(in_channel))
            self.block.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    output_padding=out_padding,
                )
            )
            self.block.append(nn.ReLU())
            in_channel = out_channel

        self.block = self.block[:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class ConvTranspose2d_MLP_Block(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        linear_out_chan: int,
        out_chans_per_layer: List[int],
        out_chans_per_layer_MLP: List[int],
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        use_batch_norm: bool = True,
        zero_weights: bool = False,
    ):
        """Complement class to Conv2d block that reverses its
        operations with transpose layers.

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [Channel, Height, Width]
            output_shape (Sequence[int]):
                The output shape of the data - [Channel, Height, Width]
            linear_out_chan (int)
                The output channel size of the linear layer. As the Conv2DBlock
                has a linear layer at the end, we need the output
                shape if the time dimension of the input is to match that of
                the output. This is usually the size of the final conv2d out chan
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [32, 64] would give two layers of hidden channel 32 then 64
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
            output_activation (Optional[Type[nn.Module]], optional):
                Allows an nn activation to be added to end of block
                Defaults to None.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]
        out_chans_per_layer_MLP = [c for c in out_chans_per_layer_MLP if c]

        num_layers = len(out_chans_per_layer)

        if type(kernel_size_per_layer) == int:
            kernel_size_per_layer = [
                (kernel_size_per_layer, kernel_size_per_layer)
            ] * num_layers

        if type(stride_per_layer) == int:
            stride_per_layer = [
                (stride_per_layer, stride_per_layer)
            ] * num_layers

        kern_stride_per_layer = list(
            zip(
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
            )
        )

        h_dim, w_dim = [output_shape[1]], [output_shape[2]]
        for kernel_size, stride in reversed(kern_stride_per_layer):
            h_dim.append(((h_dim[-1] - kernel_size[0]) // stride[0]) + 1)
            w_dim.append(((w_dim[-1] - kernel_size[1]) // stride[1]) + 1)

        h_dim, w_dim = h_dim[::-1], w_dim[::-1]
        output_padding_per_layer = []
        for i, [kernel_size, stride] in enumerate(kern_stride_per_layer):
            current_h_dim = ((h_dim[i] - 1) * stride[0]) + kernel_size[0]
            target_h_dim = h_dim[i + 1]
            current_w_dim = ((w_dim[i] - 1) * stride[1]) + kernel_size[1]
            target_w_dim = w_dim[i + 1]
            output_padding_per_layer.append(
                [
                    (target_h_dim - current_h_dim),
                    (target_w_dim - current_w_dim),
                ]
            )

        params_per_layer = list(
            zip(
                out_chans_per_layer,
                cast(Iterable, kernel_size_per_layer),
                cast(Iterable, stride_per_layer),
                cast(Iterable, output_padding_per_layer),
            )
        )

        flat_linear_out_dim = h_dim[0] * w_dim[0] * linear_out_chan
        flat_output_dim = int(torch.tensor(input_shape).prod().numpy())

        out_chans_per_layer_MLP.insert(0, flat_output_dim)
        out_chans_per_layer_MLP.append(flat_linear_out_dim)

        self.block = nn.ModuleList()

        self.block.append(nn.Flatten())

        # MLP LAYERS
        for i in range(len(out_chans_per_layer_MLP)-1):
            if use_batch_norm:
                self.block.append(nn.BatchNorm1d(out_chans_per_layer_MLP[i]))
            self.block.append(
                nn.Linear(out_chans_per_layer_MLP[i], out_chans_per_layer_MLP[i+1])
            )
            self.block.append(nn.ReLU())

        self.block.append(Reshape((linear_out_chan, h_dim[0], w_dim[0])))

        in_channel = linear_out_chan
        for out_channel, kernel_size, stride, out_padding in params_per_layer:
            if use_batch_norm:
                self.block.append(nn.BatchNorm2d(in_channel))
            self.block.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    output_padding=out_padding,
                )
            )
            self.block.append(nn.ReLU())
            in_channel = out_channel

        self.block = self.block[:-1]

        # Initialise weights of linear layers to zero
        if zero_weights == True:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear or nn.ConvTranspose2d)):
            #module.weight.data.normal_(mean=0.0, std=1.0)
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x