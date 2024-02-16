from typing import Sequence, Optional, Union

import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, output_shape: Sequence[int]):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((-1,) + tuple(self.output_shape))


class Norm(nn.Module):
    def __init__(self, ord: Optional[Union[int, float, str]] = None):
        super().__init__()
        self.ord = ord

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.divide(torch.linalg.norm(x, self.ord, dim=-1, keepdim=True))
