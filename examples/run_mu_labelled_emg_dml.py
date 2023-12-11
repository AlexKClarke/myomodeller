"""Example of training a 1D convolutional neural network to learn a deep 
metric embedding on MU labelled EMG"""

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
        "log_name": "mu_labelled_emg_dml",
        "update_module_config": {
            "update_module_name": "DeepMetricLearner",
            "update_module_kwargs": {
                "margin": 0.1,
                "classifier_weighting": 0.1,
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.0001},
            },
            "maximize_val_target": False,
            "network_config": {
                "network_name": "embedding.Conv1dEmbedding",
                "network_kwargs": {
                    "embedding_dim": 128,
                    "out_chans_per_layer": [32, 64],
                    "kernel_size_per_layer": 5,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MotorUnitLabelledEMG",
            "loader_module_kwargs": {
                "mat_path": "path",  # [enter absolute path to Demuse file here]
                "half_window_size": 50,
                "test_fraction": 0.2,
                "group_size": 100,
                "batch_size": 512,
                "one_hot_labels": True,
                "weighted_sampler": True,
                "train_sample_weighting": 1.0,
                "test_sample_weighting": 1.0,
                "flatten_samples": False,
            },
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
