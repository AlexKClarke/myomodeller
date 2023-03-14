"""Example of training a 2D convolutional neural network to classify 
8x8 MNIST images"""

import os, sys
from ray import tune

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":

    training_module_config = {
        "log_name": "mnist_classifier",
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "hpo_mode": True,
            "num_hpo_trials": 5,
            "maximize_val_target": True,
            "network_config": {
                "network_name": "blocks.Conv2dBlock",
                "network_kwargs": {
                    "input_shape": [1, 8, 8],
                    "output_shape": [10],
                    "out_chans_per_layer": [
                        tune.choice([32, 64]),
                        tune.choice([32, 64]),
                    ],
                    "output_activation": None,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64},
        },
    }

    training_module = TrainingModule(training_module_config)

    training_module.train()
