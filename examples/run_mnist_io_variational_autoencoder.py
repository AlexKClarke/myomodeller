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
        "log_name": "mnist_vae",
        "update_module_config": {
            "update_module_name": "IOVariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "multimodel",
                "optimizer_kwargs": {
                    0: {
                        "optimizer": "AdamW",
                        "optimizer_kwargs": {"lr": 0.001},
                    },
                    1: {
                        "optimizer": "AdamW",
                        "optimizer_kwargs": {"lr": 0.001},
                    },
                },
                "beta_step": 1e-2,
                "max_beta": 1.0,
                "n_steps_per_switch": 5,
                "network_training_steps": [20, 10],
                "n_samples_in_aux": 16,
            },
            "maximize_val_target": False,
            "network_config": {
                "network_name": "multimodel",
                "network_kwargs": {

                    0: {
                        "network_name": "vae.Conv2dVariationalAutoencoder",
                        "network_kwargs": {
                            "latent_dim": 2,
                            "kernel_size_per_layer": 3,
                            "stride_per_layer": 1,
                            "out_chans_per_layer": [16, 16],
                            "fix_recon_var": False,
                        },
                    },

                    1: {
                        "network_name": "blocks.MLPBlock",
                        "network_kwargs": {
                            "input_shape": [2],
                            "output_shape": [1],
                            "out_chans_per_layer": [8, 8],
                            "output_activation": "Sigmoid",
                        },
                    },
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {
                "batch_size": 128,
                "auto": True,
                "flatten_input": False,
                "full_dataset": True,
            },
        },

        "latents_visualization": True,

        "trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 5,
            "log_every_n_steps": 1,
        },



    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()



    """"trainer_kwargs": {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 200,
        "log_every_n_steps": 1,
    },"""
