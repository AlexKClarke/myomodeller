"""Example of training a 1D convolutional neural network to learn a classifier
 on MU labelled EMG"""

import os, sys
from ray import tune

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":
    # The entirety of the run is defined by config, which will be unpacked
    # at run time. The training module will look in the loader_modules
    # and update_modules __init__.py for the named modules and then pass
    training_module_config = {
        "log_name": "mu_labelled_emg_classifier",
        "hpo_mode": True,
        "num_hpo_trials": 5,
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": tune.loguniform(1e-5, 1e-3)},
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "blocks.Conv1dBlock",
                "network_kwargs": {
                    "output_shape": [128],
                    "out_chans_per_layer": [32, 64, tune.choice([None, 128])],
                    "output_activation": None,
                    "kernel_size_per_layer": tune.randint(3, 40),
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MotorUnitLabelledEMG",
            "loader_module_kwargs": {
                "mat_path": "path",  # [enter absolute path to Demuse file here]
                "half_window_size": tune.randint(50, 100),
                "test_fraction": 0.2,
                "group_size": 100,
                "batch_size": 512,
                "one_hot_labels": False,
                "weighted_sampler": True,
            },
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
