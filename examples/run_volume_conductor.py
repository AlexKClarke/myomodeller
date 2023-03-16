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
        "log_name": "volume_conductor",
        "update_module_config": {
            "update_module_name": "SupervisedRegressor",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "maximize_val_target": False,
            "network_config": {
                "network_name": "blocks.MLPBlock",
                "network_kwargs": {
                    "input_shape": [6],
                    "output_shape": [3],
                    "out_chans_per_layer": [32, 64],
                    "output_activation": "Sigmoid",
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "VolumeConductor",
            "loader_module_kwargs": {"batch_size": 64},
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
