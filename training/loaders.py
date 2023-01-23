from typing import Optional

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule


class TensorLoader(LightningDataModule):
    """Generic lightning loader that takes pytorch tensor inputs"""

    def __init__(
        self,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        test_data: Optional[torch.Tensor] = None,
        test_targets: Optional[torch.Tensor] = None,
        batch_size: int = 64,
    ):
        super().__init__()

        self.train_data = TensorDataset(train_data, train_targets)
        if val_data is not None:
            self.val_data = TensorDataset(val_data, val_targets)
        if test_data is not None:
            self.test_data = TensorDataset(test_data, test_targets)

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
        )
