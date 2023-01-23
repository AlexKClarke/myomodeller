"""Example of training a 2D convolutional neural network to classify 
8x8 MNIST images"""

import os, sys

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from sklearn.datasets import load_digits

from training.utils import (
    get_split_indices,
    split_array_by_indices,
    array_to_tensor,
)
from training.modules import SupervisedClassifier
from training.core import TrainingModule, LoaderModule
from networks.blocks import Conv2dBlock


if __name__ == "__main__":

    # Load MNIST digits from scikit.datasets
    # sklearn flattens the images for some reason so also need to reshape
    images, targets = load_digits(return_X_y=True)
    images, targets = images.astype("float32"), targets.astype("int64")
    images = images.reshape((images.shape[0], 1, 8, 8))

    # Split out train, val and test sets
    train_data, test_data = split_array_by_indices(
        (images, targets), get_split_indices(targets)
    )
    train_data, val_data = split_array_by_indices(
        train_data, get_split_indices(train_data[1])
    )

    # Convert the numpy arrays to torch tensors and break up lists
    (
        [train_images, train_targets],
        [val_images, val_targets],
        [test_images, test_targets],
    ) = [array_to_tensor(data) for data in [train_data, val_data, test_data]]

    # Z-score standardise image sets with statistics from train set
    mean, std = train_images.mean(), train_images.std()
    [train_images, val_images, test_images] = [
        ((images - mean) / std)
        for images in [train_images, val_images, test_images]
    ]

    # Add all images and associated targets to the dataloader
    loader = LoaderModule(
        train_data=train_images,
        train_targets=train_targets,
        val_data=val_images,
        val_targets=val_targets,
        test_data=test_images,
        test_targets=test_targets,
        batch_size=64,
    )

    # Construct the neural network, in this case a 2d conv block
    # Pytorch CrossEntropyLoss needs raw logits so set output_activation to none
    network = Conv2dBlock(
        input_shape=[1, 8, 8],
        output_shape=[10],
        out_chans_per_layer=[32, 64],
        output_activation=None,
    )

    # Specify the supervised classifier module with the network to be trained
    model = SupervisedClassifier(network)

    # Pair the model with the loader in the trainer
    trainer = TrainingModule(model, loader)

    # Train the model
    trainer.train()
