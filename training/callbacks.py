import torch
from pytorch_lightning import Callback


class TrainingEpochEnd(Callback):
    def on_train_epoch_end(self, trainer, module):
        results = {k: [] for k in module.training_step_outputs[0].keys()}
        [
            [results[k].append(v) for k, v in s.items()]
            for s in module.training_step_outputs
        ]
        [module.log(k, torch.stack(v).mean()) for k, v in results.items()]
        module.training_step_outputs.clear()


class ValidationEpochEnd(Callback):
    def on_validation_epoch_end(self, trainer, module):
        results = {k: [] for k in module.validation_step_outputs[0].keys()}
        [
            [results[k].append(v) for k, v in s.items()]
            for s in module.validation_step_outputs
        ]
        [module.log(k, torch.stack(v).mean()) for k, v in results.items()]
        module.validation_step_outputs.clear()


class TestEpochEnd(Callback):
    def on_test_epoch_end(self, trainer, module):
        results = {k: [] for k in module.test_step_outputs[0].keys()}
        [
            [results[k].append(v) for k, v in s.items()]
            for s in module.test_step_outputs
        ]
        [module.log(k, torch.stack(v).mean()) for k, v in results.items()]
        module.test_step_outputs.clear()
