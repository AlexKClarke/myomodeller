import torch
import numpy as np
from h5py import File
from scipy.io import loadmat
from scipy.signal.windows import gaussian
from kymatio.torch import Scattering2D

from training import LoaderModule
from loader_modules.utils import get_split_indices, split_array_by_indices


class Ultrasound(LoaderModule):
    """Loader module that gets paired ultrasound spike data"""

    def __init__(
        self,
        image_path: str,
        label_path: str,
        kernel_size: int = 100,
        log_scatter_scale: int = 3,
        split_fraction: float = 0.2,
        group_size: int = 200,
        batch_size: int = 64,
        auto: bool = False,
    ):
        (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
        ) = self._get_data(
            image_path,
            label_path,
            kernel_size,
            log_scatter_scale,
            split_fraction,
            group_size,
        )

        super().__init__(
            train_data=[train_images, train_images if auto else train_labels],
            val_data=[val_images, val_images if auto else val_labels],
            test_data=[test_images, test_images if auto else test_labels],
            batch_size=batch_size,
        )

    def _get_data(
        self,
        image_path,
        label_path,
        kernel_size,
        log_scatter_scale,
        split_fraction,
        group_size,
    ):
        """Loads data"""

        # Load data
        images = torch.from_numpy(np.array(File(image_path, "r")["images"]))
        labels = torch.from_numpy(list(loadmat(label_path).values())[-1]).t()
        labels = labels.type_as(images)

        # Pad or remove ends to make sure data is same shape up to a point
        diff = images.shape[0] - labels.shape[0]
        assert diff < 10, "Images and labels have very different lengths."
        left = int(diff / 2)
        right = int(diff / 2) + int(diff % 2)
        labels = torch.nn.functional.pad(labels, [0, 0, left, right])

        # Convolve the labels with a gaussian of appropriate support
        kernel = gaussian(
            kernel_size, kernel_size / np.sqrt(-8 * np.log(0.001))
        )
        kernel = torch.from_numpy(kernel).type_as(labels)
        kernel = kernel.unsqueeze(0).unsqueeze(0).tile([1, labels.shape[1], 1])
        labels = torch.nn.functional.conv1d(
            labels.t(), kernel, padding="same"
        ).t()

        # Preprocess the images with kymatio
        scattering = Scattering2D(
            J=log_scatter_scale, shape=(images.shape[1], images.shape[2])
        )
        images = torch.concat(
            [scattering(batch) for batch in images.split(64)], 0
        )

        # Currently we just want to flatten the scattering coefficients for MLP
        images = images.flatten(1)

        # Split out train, val and test sets using a grouped split
        train_data, test_data = split_array_by_indices(
            (images, labels),
            get_split_indices(
                labels, split_fraction=split_fraction, group_size=group_size
            ),
        )
        train_data, val_data = split_array_by_indices(
            train_data,
            get_split_indices(
                train_data[1],
                split_fraction=split_fraction,
                group_size=group_size,
            ),
        )

        # Break up lists
        (
            [train_images, train_labels],
            [val_images, val_labels],
            [test_images, test_labels],
        ) = [data for data in [train_data, val_data, test_data]]

        # Z-score standardise image sets with statistics from train set
        mean, std = train_images.mean(0, keepdims=True), train_images.std(
            0, keepdims=True
        )
        [train_images, val_images, test_images] = [
            ((images - mean) / std)
            for images in [train_images, val_images, test_images]
        ]

        return (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
        )
