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
        #"trainer_kwargs": {"accelerator": "cpu"},
        "log_name": "mnist_vae",
        "update_module_config": {
            "update_module_name": "VariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
                "l1_loss_coeff": 0.1,
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "vae.MLPVariationalAutoencoder",
                "network_kwargs": {
                    "input_shape": [64],
                    "output_shape": [64],
                    "latent_dim": 3,
                    "out_chans_per_layer": [16, 32],
                    "num_sampling_draws": 1
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64, "auto": True, "flatten_input": True},
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
