from typing import Optional, Dict

from training.core import TrainingModule


class BasicTrainer(TrainingModule):
    """Implements a single run of training with set network parameters"""

    def __init__(
        self,
        update_module,
        loader_module,
        log_name: str,
        log_dir: Optional[str] = None,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 500,
        log_every_n_steps: int = 1,
        trainer_kwargs: Optional[Dict] = None,
    ):

        super().__init__(
            update_module,
            loader_module,
            log_name,
            log_dir,
            accelerator,
            devices,
            max_epochs,
            log_every_n_steps,
            trainer_kwargs,
        )

    def train(self):
        """Fit data"""
        self.trainer.fit(self.update_module, datamodule=self.loader_module)

    def train_test(self):
        """Fit data and then test"""
        self.trainer.fit(self.update_module, datamodule=self.loader_module)
        self.trainer.test(self.update_module, dataloaders=self.loader_module)
