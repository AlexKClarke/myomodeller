from sklearn.datasets import load_digits
from torch.nn.functional import one_hot
import numpy as np
import torch

from training import LoaderModule
from loader_modules.utils import (
    get_split_indices,
    split_array_by_indices,
    array_to_tensor,
)

# to load full mnist dataset
from sklearn.datasets import fetch_openml



class MNIST(LoaderModule):
    """Loader module that retrieves sklearn's MNIST set"""

    def __init__(
        self,
        batch_size: int = 64,
        auto: bool = False,
        one_hot_labels: bool = False,
        flatten_input: bool = False,
        train_images: torch.int32 = None,
        train_labels: torch.int32 = None,
        val_images: torch.int32 = None,
        val_labels: torch.int32 = None,
        test_images: torch.int32 = None,
        test_labels: torch.int32 = None,
    ):
        (
            self.train_images,
            self.train_labels,
            self.val_images,
            self.val_labels,
            self.test_images,
            self.test_labels,
        ) = self._get_data(one_hot_labels, flatten_input)

        super().__init__(
            train_data=[self.train_images, self.train_images if auto else self.train_labels],
            val_data=[self.val_images, self.val_images if auto else self.val_labels],
            test_data=[self.test_images, self.test_images if auto else self.test_labels],
            batch_size=batch_size,
            input_shape=self.train_images.shape[1:],
            output_shape=self.train_images.shape[1:] if auto else self.train_labels.shape[1:],
        )

    def _get_data(self, one_hot_labels: bool = False, flatten_input: bool = False):
        """Loads MNIST digits from scikit.datasets"""
        print('Loading MNIST digits...')

        # sklearn flattens the images for some reason so also need to reshape
        images, labels = load_digits(return_X_y=True)
        images, labels = images.astype("float32"), labels.astype("int64")
        images = images.reshape((images.shape[0], 1, 8, 8))

        # Split out train, val and test sets
        train_data, test_data = split_array_by_indices(
            (images, labels), get_split_indices(labels)
        )
        train_data, val_data = split_array_by_indices(
            train_data, get_split_indices(train_data[1])
        )

        # Convert the numpy arrays to torch tensors and break up lists
        (
            [train_images, train_labels],
            [val_images, val_labels],
            [test_images, test_labels],
        ) = [array_to_tensor(data) for data in [train_data, val_data, test_data]]

        # If flattening is necessary
        if flatten_input:
            [train_images, val_images, test_images] = [
                images.flatten(1) for images in [train_images, val_images, test_images]
            ]

        # One hot labels if needed
        if one_hot_labels:
            [train_labels, val_labels, test_labels] = [
                one_hot(label, np.unique(labels).shape[0])
                for label in [train_labels, val_labels, test_labels]
            ]

        # Z-score standardise image sets with statistics from train set
        mean, std = train_images.mean(), train_images.std()
        [train_images, val_images, test_images] = [
            ((images - mean) / std)
            for images in [train_images, val_images, test_images]
        ]

        print('Train size: ', np.shape(train_images)[0])
        print('Test size: ', np.shape(test_images)[0])
        print('Validation size: ', np.shape(val_images)[0])

        return (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
        )
