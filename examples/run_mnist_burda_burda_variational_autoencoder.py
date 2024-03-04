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
        "log_name": "mnist_vae_burda",
        "update_module_config": {
            "update_module_name": "BurdaVariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
                "starting_beta": 0.0,
                "beta_step": 1e-2,
                "max_beta": 1.0,
                "burda_samples": 32,
            },
            "maximize_val_target": False,
            "network_config": {
                #"network_name": "vae.Conv2dVariationalAutoencoder",
                "network_name": "vae.MLPVariationalAutoencoder",
                "network_kwargs": {
                    "latent_dim": 2,
                    "out_chans_per_layer": [32, 32],
                    "fix_recon_var": False,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {
                "batch_size": 64,
                "auto": True,
                "flatten_input": True,
            },
        },

        "trainer_kwargs": {
            "accelerator": "gpu",
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

    '''"trainer_kwargs": {
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": 200,
        "log_every_n_steps": 1,
    },'''
