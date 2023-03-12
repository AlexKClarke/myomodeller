"""Example of training a 2D convolutional neural network to classify 
8x8 MNIST images"""

import os, sys
from ray import tune

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training.configure import process_training_module_config

if __name__ == "__main__":

    training_module_config = {
        "training_module_name": "BasicTrainer",
        "training_module_kwargs": {
            "log_dir": os.path.abspath("logs"),
            "log_name": "mnist_classifier",
        },
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "network_config": {
                "network_name": "blocks.Conv2dBlock",
                "network_kwargs": {
                    "input_shape": [1, 8, 8],
                    "output_shape": [10],
                    "out_chans_per_layer": [
                        tune.choice([32, 64]),
                        tune.choice([32, 64]),
                    ],
                    "kernel_size_per_layer": 3,
                    "output_activation": None,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64},
        },
    }

    def run(config):
        training_module = process_training_module_config(config)
        training_module.train()

    analysis = tune.run(
        run,
        resources_per_trial={"cpu": 1, "gpu": 1},
        metric="val_target",
        mode="max",
        config=training_module_config,
        num_samples=5,
    )

    print(analysis.best_config)
