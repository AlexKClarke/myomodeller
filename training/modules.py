from typing import Dict, Optional

from pytorch_lightning import LightningModule


class CoreModule(LightningModule):
    def __init__(
        self,
        network,
        loss_fn,
        optimiser,
        optimiser_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn

        self.optimiser = optimiser
        if optimiser_kwargs is None:
            self.optimiser_kwargs = {}
        else:
            self.optimiser_kwargs = optimiser_kwargs

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return self.optimiser(self.parameters(), **self.optimiser_kwargs)

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        self.log_dict = {"train_loss": loss}
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        self.log_dict = {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_predict = self(x)
        loss = self.loss_fn(y_predict, y_target)
        self.log_dict = {"train_loss": loss}
