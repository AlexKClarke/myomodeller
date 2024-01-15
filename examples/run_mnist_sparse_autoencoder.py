"""Example of training a 2D convolutional neural network to autoencode 
8x8 MNIST images"""

import os, sys

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":
    # The entirety of the run is defined by config, which will be unpacked
    # at run time. The training module will look in the loader_modules
    # and update_modules __init__.py for the named modules and then pass
    training_module_config = {
        "log_name": "mnist_autoencoder",
        "update_module_config": {
            "update_module_name": "SparseAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
                "l1_loss_coeff": 0.1,
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "sparse.Conv2dSparseAutoencoder",
                "network_kwargs": {
                    "input_shape": [1, 8, 8],
                    "output_shape": [1, 8, 8],
                    "sparse_dim": 3,
                    "out_chans_per_layer": [32, 64],
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64, "auto": True},
        },

        # Following block just to enable cpu training for testing
        # Not necessary for the merge in main branch
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 50,
            "log_every_n_steps": 1,
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
