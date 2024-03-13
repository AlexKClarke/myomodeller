"""Example of training a 2D convolutional neural network to classify 
8x8 MNIST images using hyperparamteter optimisation"""

import os, sys
from ray import tune

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":
    # When using the automatic hyperparameter optimisation, we need to
    # tell the hpo module what we want it to vary. This is done by replacing
    # values in the config with ray tune search space api functions
    # For the full list of functions see:
    # https://docs.ray.io/en/latest/tune/api/search_space.html
    # The hpo_mode flag in also needs to be added to the config to tell the
    # training module to perform hpo. Also the number of hpo trials can
    # also be specified (defaults to 10 otherwise)
    training_module_config = {
        "log_name": "mnist_classifier",
        "hpo_mode": True,
        "num_hpo_trials": 5,
        "trainer_kwargs": {"max_epochs": 1000},
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
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

    # Once the config is ready, we just pass it to TrainingModule as usual
    training_module = TrainingModule(training_module_config)

    # Run HPO with the training module. Alongside the normal version files
    # it will create the best config on completion. This will have an
    # additional "version" key that can be used to identify the best folder
    training_module.train()
