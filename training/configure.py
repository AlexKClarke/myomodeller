"""Converts a training module config dict to the class
Update module config specified for example as:

training_module_config = {
    "training_module_kwargs": {"log_name": "mnist_classifier"},
    "update_module_config": {
        "update_module_name": "SupervisedClassifier",
        "update_module_kwargs": {
            "optimizer": "AdamW",
            "optimizer_kwargs": {"lr": 0.001},
        },
        "hpo_mode": False,
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
        "hpo_mode": False,
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

    return getattr(update_modules, config["update_module_name"])(
        network=process_network_config(config["network_config"]),
        hpo_mode=config["hpo_mode"],
        maximize_val_target=config["maximize_val_target"],
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
