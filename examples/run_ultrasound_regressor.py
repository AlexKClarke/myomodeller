"""Example of training a MLP to regress spikes from ultrasound data"""

import os, sys

if os.path.basename(os.getcwd()) != "pyrepo":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule

if __name__ == "__main__":
    # The entirety of the run is defined by config, which will be unpacked
    # at run time. The training module will look in the loader_modules
    # and update_modules __init__.py for the named modules and then pass
    training_module_config = {
        "log_name": "uss_spike_classifier",
        "update_module_config": {
            "update_module_name": "SupervisedRegressor",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "blocks.MLPBlock",
                "network_kwargs": {
                    "input_shape": [156240],
                    "output_shape": [1],
                    "out_chans_per_layer": [],
                    "output_activation": "Sigmoid",
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "Ultrasound",
            "loader_module_kwargs": {
                "image_path": r"EXP1_P5_R4\images.mat",
                "label_path": r"EXP1_P5_R4\Firings_8.mat",
                "kernel_size": 100,
                "log_scatter_scale": 3,
                "split_fraction": 0.2,
                "group_size": 50,
                "batch_size": 64,
            },
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
