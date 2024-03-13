"""Networks which learn embeddings 
"""
from typing import Sequence, Union, Optional, List, Type, Tuple
import torch
import torch.nn as nn
from networks.blocks import (
    Conv1dBlock,
    Conv2dBlock,
    MLPBlock,
)
from networks.utils import Norm


class MLPEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        embedding_dim: int,
        out_chans_per_layer: List[int],
        embedding_norm_ord: Optional[Union[str, int, float]] = 2,
        output_shape: Optional[Sequence[int]] = None,
        use_batch_norm: bool = True,
    ):
        """An MLP-based embedding layer with optional linear classifier layer

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [..., Feature]
                (Linear layers operate on the final dimension)
            embedding_dim (int):
                The dimension of the embedding layer
            out_chans_per_layer (List[int]):
                Number of hidden channels per layer, for example
                [64, 32] would give two layers of hidden channel 64 then 32
            embedding_norm_ord:
                The order of the norm applied to the embedding layer
                Set to None if no norm wanted
                See torch.linalg.norm for ord options
            output_shape Optional(Sequence[int]):
                If used, will apply a final linear layer to the embedding dim
                for an auxiliary classification loss
                The output shape of the data - [..., Feature]
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to Identity.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.embedding = MLPBlock(
            input_shape=input_shape,
            output_shape=[embedding_dim],
            out_chans_per_layer=out_chans_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=None,
            use_output_bias=False,
        )

        if embedding_norm_ord is not None:
            self.norm = Norm(ord=embedding_norm_ord)
        else:
            self.norm = nn.Identity()

        if output_shape:
            self.classifier = MLPBlock(
                input_shape=[embedding_dim],
                output_shape=output_shape,
                out_chans_per_layer=[],
                use_batch_norm=False,
                output_activation=None,
                use_output_bias=False,
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = self.norm(self.embedding(x))
        return self.x

    def classifier_output(self) -> torch.Tensor:
        return self.classifier(self.x)


class Conv1dEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        embedding_dim: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[int]] = 3,
        stride_per_layer: Union[int, List[int]] = 1,
        embedding_norm_ord: Optional[Union[str, int, float]] = 2,
        output_shape: Optional[Sequence[int]] = None,
        use_batch_norm: bool = True,
    ):
        """An Conv1d-based embedding layer with optional linear classifier layer

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [..., Feature]
                (Linear layers operate on the final dimension)
            embedding_dim (int):
                The dimension of the embedding layer
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
            embedding_norm_ord:
                The order of the norm applied to the embedding layer
                Set to None if no norm wanted
                See torch.linalg.norm for ord options
            output_shape Optional(Sequence[int]):
                If used, will apply a final linear layer to the embedding dim
                for an auxiliary classification loss
                The output shape of the data - [..., Feature]
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to Identity.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.embedding = Conv1dBlock(
            input_shape=input_shape,
            output_shape=[embedding_dim],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=None,
            use_output_bias=False,
        )

        if embedding_norm_ord is not None:
            self.norm = Norm(ord=embedding_norm_ord)
        else:
            self.norm = nn.Identity()

        if output_shape:
            self.classifier = MLPBlock(
                input_shape=[embedding_dim],
                output_shape=output_shape,
                out_chans_per_layer=[],
                use_batch_norm=False,
                output_activation=None,
                use_output_bias=False,
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = self.norm(self.embedding(x))
        return self.x

    def classifier_output(self) -> torch.Tensor:
        return self.classifier(self.x)


class Conv2dEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        embedding_dim: int,
        out_chans_per_layer: List[int],
        kernel_size_per_layer: Union[int, List[Tuple[int, int]]] = 3,
        stride_per_layer: Union[int, List[Tuple[int, int]]] = 1,
        embedding_norm_ord: Optional[Union[str, int, float]] = 2,
        output_shape: Optional[Sequence[int]] = None,
        use_batch_norm: bool = True,
    ):
        """An Conv2d-based embedding layer with optional linear classifier layer

        Args:
            input_shape (Sequence[int]):
                The input shape of the data - [..., Feature]
                (Linear layers operate on the final dimension)
            embedding_dim (int):
                The dimension of the embedding layer
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
            embedding_norm_ord:
                The order of the norm applied to the embedding layer
                Set to None if no norm wanted
                See torch.linalg.norm for ord options
            output_shape Optional(Sequence[int]):
                If used, will apply a final linear layer to the embedding dim
                for an auxiliary classification loss
                The output shape of the data - [..., Feature]
            use_batch_norm (bool, optional):
                Whether to batch norm after each linear layer
                Defaults to True.
            encoder_output_activation (Optional[Type[nn.Module]], optional):
                Activation layer at end of encoder
                Defaults to Identity.
        """
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]

        self.embedding = Conv2dBlock(
            input_shape=input_shape,
            output_shape=[embedding_dim],
            out_chans_per_layer=out_chans_per_layer,
            kernel_size_per_layer=kernel_size_per_layer,
            stride_per_layer=stride_per_layer,
            use_batch_norm=use_batch_norm,
            output_activation=None,
            use_output_bias=False,
        )

        if embedding_norm_ord is not None:
            self.norm = Norm(ord=embedding_norm_ord)
        else:
            self.norm = nn.Identity()

        if output_shape:
            self.classifier = MLPBlock(
                input_shape=[embedding_dim],
                output_shape=output_shape,
                out_chans_per_layer=[],
                use_batch_norm=False,
                output_activation=None,
                use_output_bias=False,
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = self.norm(self.embedding(x))
        return self.x

    def classifier_output(self) -> torch.Tensor:
        return self.classifier(self.x)
