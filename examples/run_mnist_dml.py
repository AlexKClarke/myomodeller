"""Example of training a 2D convolutional neural network to learn a deep 
metric embedding on 8x8 MNIST images"""

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
        "log_name": "mnist_dml",
        "update_module_config": {
            "update_module_name": "DeepMetricLearner",
            "update_module_kwargs": {
                "margin": 0.1,
                "classifier_weighting": 0.1,
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "embedding.Conv2dEmbedding",
                "network_kwargs": {
                    "embedding_dim": 128,
                    "out_chans_per_layer": [32, 64],
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {
                "batch_size": 256,
                "one_hot_labels": True,
            },
        },
    }

    # Once the config is defined it can be passed to an instance of the
    # training module
    training_module = TrainingModule(training_module_config)

    # Train the model and pass the results to tensorboard
    training_module.train()
