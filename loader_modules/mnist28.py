from sklearn.datasets import load_digits
import torch.nn.functional as F
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



class MNIST28(LoaderModule):
    """Loader module that retrieves sklearn's MNIST set"""

    def __init__(
        self,
        batch_size: int = 64,
        auto: bool = False,
        one_hot_labels: bool = False,
        flatten_input: bool = False,
        downsample_pooling_size: tuple = (1, 1),
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
        ) = self._get_data(one_hot_labels, flatten_input, downsample_pooling_size)

        super().__init__(
            train_data=[self.train_images, self.train_images if auto else self.train_labels],
            val_data=[self.val_images, self.val_images if auto else self.val_labels],
            test_data=[self.test_images, self.test_images if auto else self.test_labels],
            batch_size=batch_size,
            input_shape=self.train_images.shape[1:],
            output_shape=self.train_images.shape[1:] if auto else self.train_labels.shape[1:],
        )

    def _get_data(self, one_hot_labels: bool = False, flatten_input: bool = False, downsample_pooling_size: tuple = (1, 1)):
        """Loads MNIST digits from scikit.datasets"""
        print('Loading MNIST28 digits...')

        mnist = fetch_openml('mnist_784')
        index_number = np.random.permutation(70000)
        x1, y1 = mnist.data.loc[index_number], mnist.target.loc[index_number]
        x1.reset_index(drop=True, inplace=True)
        y1.reset_index(drop=True, inplace=True)
        '''x_train, x_test = x1[:55000], x1[55000:]
        y_train, y_test = y1[:55000], y1[55000:]'''

        x1 = np.array(x1)
        y1 = np.array(y1)
        images, labels = x1.astype("float32"), y1.astype("int64")
        images = images.reshape((images.shape[0], 1, 28, 28))


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

        train_images = F.max_pool2d(train_images, kernel_size=downsample_pooling_size, stride=downsample_pooling_size)/255
        test_images = F.max_pool2d(test_images, kernel_size=downsample_pooling_size, stride=downsample_pooling_size)/255
        val_images = F.max_pool2d(val_images, kernel_size=downsample_pooling_size, stride=downsample_pooling_size)/255

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

        # Standardise between 0 and 1
        '''max_value_train = torch.max(train_images)
        max_value_test = torch.max(test_images)
        max_value_val = torch.max(val_images)

        train_images = train_images / max_value_train
        test_images = test_images / max_value_test
        val_images = val_images / max_value_val'''


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
