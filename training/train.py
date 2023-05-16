from typing import Dict, Union, Optional

import os
import json
import shutil
import time

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

    Config Arguments:

    -update_module_config
        config specifying an instance of an UpdateModule subclass
    -loader_module_config
        config specifying an instance of a LoaderModule subclass
    -log_name
        A string specifying the name of the log where training results and
        hyperparameters will be saved
    -log_dir
        A string specifying the directory for saving results
    -hpo_mode
        A bool that switches from a single train to hyperparameter optimisation
    -num_hpo_trials
        Number of hyperparameter optimisation trials
    -trainer_kwargs:
        Dict to add additional kwargs to trainer.
        See pytorch_lightning.Trainer for full list

    """

    def __init__(
        self,
        config: Union[Dict, str],
    ):
        # Detect if config is a dict or a json path
        if isinstance(config, str):
            with open(config, "r") as file:
                config = json.load(file)
        else:
            config = config

        # Need to add the hpo_mode flag if missing
        if "hpo_mode" not in config:
            config["hpo_mode"] = False

        # Need to set the ckpt_path as None if missing
        if "ckpt_path" not in config:
            config["ckpt_path"] = None

        self.config = config

        self._build_path()

    def _build_path(self) -> None:
        """Sets the internal paths used by lightning logger and ray tune"""

        self.save_dir = (
            os.path.abspath("logs")
            if "log_dir" not in self.config
            else self.config["log_dir"]
        )

        self.log_name = self.config["log_name"]

        self.whole_path = os.path.abspath(self.save_dir + "/" + self.log_name)

    def _get_version(self) -> int:
        """Detects versioning in the specified file paths to avoid
        overwriting existing versions
        """

        if os.path.exists(self.save_dir + "/" + self.log_name):
            versions = []
            for l in os.listdir(self.save_dir + "/" + self.log_name):
                if l.startswith("version_"):
                    versions.append(int(l.split("_")[-1]))
            version_number = 0 if len(versions) == 0 else max(versions) + 1
        else:
            version_number = 0

        return version_number

    def _dump_config(
        self, config: Dict, version: Optional[int] = None, best: bool = False
    ) -> None:
        """Saves the config as a json file which will match the version
        naming scheme of the lightning logger (using the version arg).
        HPO-related arguments are removed as output config will not
        contain ray tune search space functions. The best flag is for the end
        of hpo optimisation and will place the config json outside the version
        folders (and change the name of the config to "best_config")
        """

        if version is None:
            assert (
                best
            ), "Version number needed if not saving best config after HPO."

        path = self.whole_path
        if best is False:
            path += "/version_" + str(version)

        if os.path.isdir(path) is False:
            os.makedirs(path)

        # Remove hpo-related flags
        for key in ["hpo_mode", "num_hpo_trials"]:
            if key in config:
                del config[key]

        filename = "best_config.json" if best else "config.json"
        with open(
            os.path.abspath(path + "/" + filename), "w", encoding="utf-8"
        ) as file:
            json.dump(config, file, indent=4)

    def _run(self, config: Dict, test_mode=False) -> None:
        """The internal train/test function which will create instances of the
        modules from the config and then start a trainer instance."""

        # Create the loader module from the config
        loader_module = process_loader_module_config(
            config["loader_module_config"]
        )

        # Add network input shape if missing
        network_config = config["update_module_config"]["network_config"]
        if "input_shape" not in network_config["network_kwargs"]:
            if loader_module.train_data_present:
                input_shape = (
                    loader_module.train_dataloader().dataset[0][0].shape
                )
            elif loader_module.test_data_present:
                input_shape = (
                    loader_module.test_dataloader().dataset[0][0].shape
                )
            else:
                raise ValueError("No data in train or test dataloaders.")
            network_config["network_kwargs"]["input_shape"] = input_shape

        # The update module also needs a hpo_flag if in hpo mode as this will
        # be used to add a ray tune callback to the callbacks list
        if config["hpo_mode"]:
            config["update_module_config"]["hpo_mode"] = True

        # Create the update module from the config
        update_module = process_update_module_config(
            config["update_module_config"]
        )

        # The update module needs an example sample from the loader module
        # as this is used to build the network graph
        if loader_module.train_data_present:
            update_module.example_input_array = (
                loader_module.train_dataloader().dataset[0][0]
            ).unsqueeze(0)
        elif loader_module.test_data_present:
            update_module.example_input_array = (
                loader_module.test_dataloader().dataset[0][0]
            ).unsqueeze(0)
        else:
            raise ValueError("No data in train or test dataloaders.")

        # Create the trainer kwargs dict if it does not already exist and then
        # add a few good default arguments if these are not specified
        trainer_kwargs = (
            {} if "trainer_kwargs" not in config else config["trainer_kwargs"]
        )
        trainer_kwargs = {
            key: (val if key not in trainer_kwargs else trainer_kwargs[key])
            for key, val in zip(
                ["accelerator", "devices", "max_epochs", "log_every_n_steps"],
                ["gpu", 1, 5000, 1],
            )
        }

        # Create the logger with the internal save_dir and log_name
        # This will control what is sent to TensorBoard
        trainer_kwargs["logger"] = TensorBoardLogger(
            save_dir=self.save_dir,
            name=self.log_name,
            version=self._get_version(),
            log_graph=True,
        )

        # Save the config file to the newly created version folder
        if test_mode is False:
            self._dump_config(config, self._get_version())

        # Instantiate the lighting trainer module
        trainer_module = Trainer(**trainer_kwargs)

        # Check that dataloader has correct data for training/testing
        if test_mode:
            assert (
                loader_module.test_data_present
            ), "Loader module needs test data for test mode."
        else:
            assert (
                loader_module.train_data_present
                and loader_module.val_data_present
            ), "Loader module needs train and validation data for train mode."

        # Begin training if not in test mode
        if test_mode is False:
            trainer_module.fit(
                update_module,
                datamodule=loader_module,
                ckpt_path=config["ckpt_path"],
            )

        # Run testing
        if loader_module.test_data_present:
            ckpt_path = config["ckpt_path"] if test_mode else None
            trainer_module.test(
                update_module,
                dataloaders=loader_module,
                ckpt_path=ckpt_path,
            )

    def _hpo(self) -> None:
        """Internal hyperparameter optimisation using ray tune"""

        # We need to tell tune if a high val_target result is good or bad
        mode = (
            "max"
            if self.config["update_module_config"]["maximize_val_target"]
            else "min"
        )

        # Set the number of tune HPO trials
        if "num_hpo_trials" not in self.config:
            num_samples = 10
        else:
            num_samples = self.config["num_hpo_trials"]

        # Tune will take the internal train function and run it, passing
        # different versions of config with the search space functions
        # replaced by specific values
        analysis = tune.run(
            tune.with_parameters(self._run),
            resources_per_trial={"cpu": 1, "gpu": 1},
            metric="val_target",
            mode=mode,
            config=self.config,
            num_samples=num_samples,
            local_dir="tune_temp",
            name="temp",
        )

        # We want to add a specific key to the output best config telling us
        # which version folder was the best, in case we want to use the
        # checkpoints in that folder
        best_config = analysis.best_config
        best_config["hpo_version"] = analysis.trials.index(analysis.best_trial)
        self._dump_config(best_config, best=True)

        # Delete the temp folder created by tune after waiting for processes
        # to finish
        time.sleep(1)
        if os.path.isdir("tune_temp"):
            shutil.rmtree("tune_temp")

    def train(self) -> None:
        """Runs either a single lightning training session or multiple
        depending on the hpo_mode flag in the config"""

        if self.config["hpo_mode"] is False:
            self._run(self.config)
        else:
            self._hpo()

    def test(self) -> None:
        """Runs a test session and outputs a tensorboard file with results"""

        self._run(self.config, test_mode=True)
