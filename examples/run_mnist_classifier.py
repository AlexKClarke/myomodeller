"""Example of training a 2D convolutional neural network to classify 
8x8 MNIST images"""

import os, sys

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from sklearn.datasets import load_digits

from training import TrainingModule
from training.loader_modules import MNIST
from training.update_modules import SupervisedClassifier
from networks.blocks import Conv2dBlock

from training.configure import process_training_module_config

if __name__ == "__main__":
    # Load the mnist data module
    loader_module = MNIST(batch_size=64)

    # Specify the supervised classifier module with the network to be trained,
    # in this case a 2d conv block (Pytorch CrossEntropyLoss needs raw logits
    # so set output_activation to none)
    update_module = SupervisedClassifier(
        network=Conv2dBlock(
            input_shape=[1, 8, 8],
            output_shape=[10],
            out_chans_per_layer=[32, 64],
            output_activation=None,
        ),
        optimizer="AdamW",
        optimizer_kwargs={"lr": 0.001},
    )

    # Pair the model with the loader in the trainer
    training_module = TrainingModule(
        update_module, loader_module, log_name="mnist_classifier"
    )

    # Alternatively we can use a config dict instead:
    training_module_config = {
        "training_module_name": "BasicTrainer",
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "network_config": {
                "network_name": "blocks.MLPBlock",
                "network_kwargs": {
                    "input_shape": [32],
                    "output_shape": [16],
                    "out_chans_per_layer": [64, 32],
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64},
        },
        "training_module_kwargs": {"log_name": "mnist_classifier"},
    }

    training_module = process_training_module_config(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
