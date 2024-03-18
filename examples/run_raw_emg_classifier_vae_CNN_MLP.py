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
        "log_name": "raw_emg_classifier_CNN_MLP",
        "update_module_config": {
            "update_module_name": "VariationalAutoencoder",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                #"optimizer_kwargs": {"lr": 0.01},
                "beta_step": 1.0e-2,
                "max_beta": 1.0,
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "vae.Conv2d_MLP_VariationalAutoencoder",
                "network_kwargs": {
                    "latent_dim": 2,
                    "kernel_size_per_layer": [(15, 1), (15, 1), (15, 1)],  # (time, channels) for each layer
                    "stride_per_layer": [(1, 1), (7, 1), (15, 1)],  # (time, channels)
                    "out_chans_per_layer": [16, 32, 64],
                    "out_chans_per_layer_MLP": [2048, 512],
                    "fix_recon_var": False,
                    "zero_weights": False,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "RawEMGLabelled",
            "loader_module_kwargs": {
                "file_path": "emg_data_folder/5_finger_pressure/teo/flexion/session_1",
                "test_fraction": 0.1, # of whole dataset
                "val_fraction": 0.05, # of test set
                "group_size": 1,
                "batch_size": 32,
                "one_hot_labels": False,
                "shuffle_data": True,
                "flatten_input": False,
                "rectify_emg": False,
            },
        },
        "trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 30,
            "log_every_n_steps": 20,
        },

        "latents_visualization": True,
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
