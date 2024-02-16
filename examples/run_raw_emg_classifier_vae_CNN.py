"""Example of training a 1D convolutional neural network to learn a classifier
 on MU labelled EMG"""

import os, sys
from ray import tune

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":
    # The entirety of the run is defined by config, which will be unpacked
    # at run time. The training module will look in the loader_modules
    # and update_modules __init__.py for the named modules and then pass
    training_module_config = {
        "log_name": "raw_emg_classifier_CNN",
        "update_module_config": {
            "update_module_name": "VariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.01},
                "beta_step": 1e-2,
                "max_beta": 1.5,
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "vae.Conv2dVariationalAutoencoder",
                "network_kwargs": {
                    "latent_dim": 10,
                    "kernel_size_per_layer": [(25, 1), (10, 1)], # (time, channels) for each lay
                    "stride_per_layer": [(10, 1), (5, 1)], # (time, channels)
                    "out_chans_per_layer": [10, 20],
                    "fix_recon_var": False,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "RawEMGLabelled",
            "loader_module_kwargs": {
                "file_path": "emg_data_folder",
                "test_fraction": 0.2,
                "group_size": 1,
                "batch_size": 32,
                "one_hot_labels": False,
                "shuffle_data": True,
                "flatten_input": False,
            },
        },
        "trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 100,
            "log_every_n_steps": 10,
        },

        "latents_visualization": True,
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
