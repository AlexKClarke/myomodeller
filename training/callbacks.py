import torch
from pytorch_lightning import Callback


class TrainingEpochEnd(Callback):
    def on_train_epoch_end(self, trainer, module):
        results = {}
        for step in module.training_step_outputs:
            for k, v in step.items():
                if k not in results:
                    results[k] = [v]
                else:
                    results[k].append(v)

        [module.log(k, torch.stack(v).float().mean()) for k, v in results.items()] # converted to float before mean() to avoid type issue
        module.training_step_outputs.clear()


class ValidationEpochEnd(Callback):
    def on_validation_epoch_end(self, trainer, module):
        results = {}
        for step in module.validation_step_outputs:
            for k, v in step.items():
                if k not in results:
                    results[k] = [v]
                else:
                    results[k].append(v)

        [module.log(k, torch.stack(v).mean()) for k, v in results.items()]
        module.validation_step_outputs.clear()


class TestEpochEnd(Callback):
    def on_test_epoch_end(self, trainer, module):
        results = {}
        for step in module.test_step_outputs:
            for k, v in step.items():
                if k not in results:
                    results[k] = [v]
                else:
                    results[k].append(v)

        [module.log(k, torch.stack(v).mean()) for k, v in results.items()]
        module.test_step_outputs.clear()
