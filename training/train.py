from typing import Dict, Union, Optional

import os
import json

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune

from training.configure import (
    process_loader_module_config,
    process_update_module_config,
)


class TrainingModule:
    """
    Matches an UpdateModule subclass (see modules.py) with a LoaderModule
    subclass and then runs a pytorch lightning Trainer upon calls to the
    train method.

    Some of the more important lightning Trainer arguments have been pulled out,
    but additional kwargs can be added using trainer_kwargs.

    Arguments:

    -update_module
        An instance of an UpdateModule subclass
    -loader_module
        An instance of a LoaderModule
    -log_name
        A string specifying the name of the log where training results and
        hyperparameters will be saved
    -log_dir
        A string specifying the directory to place the tensorboard log.
        If left blank will save to a local logs folder
    -accelerator
        A string specifying "cpu" or "gpu" use. Defaults to "gpu"
    -devices
        Specifies number of devices to use e.g. 1 GPU. Defaults to 1
    -max_epochs
        Maximum epochs before training is stopped (if early stopping is not
        triggering). Defaults to 500.
    -log_every_n_steps
        How often to log results. Default is every step.
    -trainer_kwargs:
        Option to add additional kwargs to trainer.
        See pytorch_lightning.Trainer for full list
    """

    def __init__(
        self,
        config: Union[Dict, str],
    ):

        if isinstance(config, str):
            with open(config, "r") as file:
                self.config = json.load(file)
        else:
            self.config = config

        self.fit_flag = False

        self._build_path()

    def _build_path(self):

        self.save_dir = (
            os.path.abspath("logs")
            if "log_dir" not in self.config
            else self.config["log_dir"]
        )

        self.log_name = self.config["log_name"]

        self.whole_path = os.path.abspath(self.save_dir + "/" + self.log_name)

    def _get_version(self):
        if os.path.exists(self.save_dir + "/" + self.log_name):
            versions = []
            for l in os.listdir(self.save_dir + "/" + self.log_name):
                if l.startswith("version_"):
                    versions.append(int(l.split("_")[-1]))
            version = 0 if len(versions) == 0 else max(versions) + 1
        else:
            version = 0

        return version

    def _dump_config(
        self, config: Dict, version: Optional[int] = None, best: bool = False
    ):
        path = self.whole_path
        if best is False:
            path += "/version_" + str(version)

        if os.path.isdir(path) is False:
            os.makedirs(path)

        filename = "best_config.json" if best else "config.json"
        with open(
            os.path.abspath(path + "/" + filename), "w", encoding="utf-8"
        ) as file:
            json.dump(config, file, indent=4)

    def _train(self, config):

        loader_module = process_loader_module_config(
            config["loader_module_config"]
        )

        update_module = process_update_module_config(
            config["update_module_config"]
        )
        update_module.example_input_array = next(
            iter(loader_module.train_dataloader())
        )[0]

        trainer_kwargs = (
            {} if "trainer_kwargs" not in config else config["trainer_kwargs"]
        )
        trainer_kwargs = {
            key: (val if key not in trainer_kwargs else trainer_kwargs[key])
            for key, val in zip(
                ["accelerator", "devices", "max_epochs", "log_every_n_steps"],
                ["gpu", 1, 500, 1],
            )
        }
        trainer_kwargs["logger"] = TensorBoardLogger(
            save_dir=self.save_dir,
            name=self.log_name,
            version=self._get_version(),
            log_graph=True,
        )

        self._dump_config(config, self._get_version())

        trainer_module = Trainer(**trainer_kwargs)
        trainer_module.fit(update_module, datamodule=loader_module)
        if loader_module.test_data_present:
            trainer_module.test(update_module, dataloaders=loader_module)

    def _hpo(self):

        mode = (
            "max"
            if self.config["update_module_config"]["maximize_val_target"]
            else "min"
        )

        if "num_hpo_trials" not in self.config["update_module_config"]:
            num_samples = 10
        else:
            num_samples = self.config["update_module_config"]["num_hpo_trials"]

        analysis = tune.run(
            self._train,
            resources_per_trial={"cpu": 1, "gpu": 1},
            metric="val_target",
            mode=mode,
            config=self.config,
            num_samples=num_samples,
        )

        best_config = analysis.best_config
        best_config["hpo_version"] = analysis.trials.index(analysis.best_trial)
        self._dump_config(best_config, best=True)

    def train(self):
        if self.config["update_module_config"]["hpo_mode"] is False:
            self._train(self.config)
        else:
            self._hpo()
