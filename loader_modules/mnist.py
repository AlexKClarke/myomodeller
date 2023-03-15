from sklearn.datasets import load_digits

from training import LoaderModule
from loader_modules.utils import (
    get_split_indices,
    split_array_by_indices,
    array_to_tensor,
)


class MNIST(LoaderModule):
    """Loader module that retrieves sklearn's MNIST set"""

    def __init__(self, batch_size: int = 64):

        (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
        ) = self._get_data()

        super().__init__(
            train_data=[train_images, train_labels],
            val_data=[val_images, val_labels],
            test_data=[test_images, test_labels],
            batch_size=batch_size,
        )

    def _get_data(self):
        """Loads MNIST digits from scikit.datasets"""

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
        ) = [
            array_to_tensor(data) for data in [train_data, val_data, test_data]
        ]

        # Z-score standardise image sets with statistics from train set
        mean, std = train_images.mean(), train_images.std()
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
