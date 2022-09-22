from typing import Sequence
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, output_shape: Sequence[int]):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((-1,) + tuple(self.output_shape))

