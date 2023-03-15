"""Configs are used to specify models and to train them. For example:

    training_module_config = {
        "log_name": "mnist_classifier",
        "hpo_mode": True,
        "num_hpo_trials": 5,
        "trainer_kwargs": {"max_epochs": 1000},
        "update_module_config": {
            "update_module_name": "SupervisedClassifier",
            "update_module_kwargs": {
                "optimizer": "AdamW",
                "optimizer_kwargs": {"lr": 0.001},
            },
            "maximize_val_target": True,
            "network_config": {
                "network_name": "blocks.Conv2dBlock",
                "network_kwargs": {
                    "input_shape": [1, 8, 8],
                    "output_shape": [10],
                    "out_chans_per_layer": [
                        tune.choice([32, 64]),
                        tune.choice([32, 64]),
                    ],
                    "output_activation": None,
                },
            },
        },
        "loader_module_config": {
            "loader_module_name": "MNIST",
            "loader_module_kwargs": {"batch_size": 64},
        },
    }


"""

from typing import Dict


def process_network_config(config: Dict):
    """Converts a network config dict to the class
    Network config specified for example as:

    network_config = {
        "network_name": "blocks.Conv2dBlock",
        "network_kwargs": {
            "input_shape": [1, 8, 8],
            "output_shape": [10],
            "out_chans_per_layer": [32, 64],
            "output_activation": None,
        },
    },

    """

    import networks as net

    for s in config["network_name"].split("."):
        net = getattr(net, s)

    return net(**config["network_kwargs"])


def process_update_module_config(config: Dict):
    """Converts a update module config dict to the class
    Update module config specified for example as:

    update_module_config = {
        "update_module_name": "SupervisedClassifier",
        "update_module_kwargs": {
            "optimizer": "AdamW",
            "optimizer_kwargs": {"lr": 0.001},
        },
        "maximize_val_target": True,
        "network_config": {
            "network_name": "blocks.Conv2dBlock",
            "network_kwargs": {
                "input_shape": [1, 8, 8],
                "output_shape": [10],
                "out_chans_per_layer": [32, 64],
                "output_activation": None,
            },
        },
    },

    """

    import update_modules

    if "hpo_mode" not in config:
        config["hpo_mode"] = False

    return getattr(update_modules, config["update_module_name"])(
        network=process_network_config(config["network_config"]),
        maximize_val_target=config["maximize_val_target"],
        hpo_mode=config["hpo_mode"],
        **config["update_module_kwargs"]
    )


def process_loader_module_config(config: Dict):
    """Converts a loader module config dict to the class
    Update module config specified for example as:

    loader_module_config = {
        "loader_module_name": "MNIST",
        "loader_module_kwargs": {"batch_size": 64}
    }

    """

    import loader_modules

    return getattr(loader_modules, config["loader_module_name"])(
        **config["loader_module_kwargs"]
    )
