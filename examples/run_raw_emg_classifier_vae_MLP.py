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
        "log_name": "raw_emg_classifier_MLP",
        "update_module_config": {
            "update_module_name": "VariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.01},
                "beta_step": 2.0e-2,
                "max_beta": 1.0,
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "vae.MLPVariationalAutoencoder",
                "network_kwargs": {
                    "latent_dim": 2,
                    "out_chans_per_layer": [1024, 512, 128, 64, 32],
                    "fix_recon_var": False,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "RawEMGLabelled",
            "loader_module_kwargs": {
                "file_path": "emg_data_folder/gesture_set_1",
                "test_fraction": 0.1, # of whole dataset
                "val_fraction": 0.03, # of test set
                "group_size": 1,
                "batch_size": 32,
                "one_hot_labels": False,
                "shuffle_data": True,
                "flatten_input": True,
            },
        },
        "trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 20,
            "log_every_n_steps": 10,
        },

        "latents_visualization": True,
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()