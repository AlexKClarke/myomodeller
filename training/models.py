import torch
import torch.nn as nn


class SupervisedLearner(nn.Module):
    def __init__(
            self,
            network,
            loss_function,
            optimiser,
            optimiser_kwargs,
            train_data,
            train_targets,
            valid_data,
            valid_targets,
            batch_size,
            device=None,
            dtype=None
            ):
        super().__init__()
        
        d_kwargs = {}
        if device is None:
            d_kwargs["device"] = train_data.device
        else:
            d_kwargs["device"] = device
        if dtype is None:
            d_kwargs["dtype"] = train_data.dtype
        else:
            d_kwargs["dtype"] = dtype
        self.d_kwargs = d_kwargs
            
        self.network = network.to(**d_kwargs)
        self.loss_function = loss_function.to(**d_kwargs)
        self.optimiser = optimiser(self.parameters(), **optimiser_kwargs)
        
        self.train_data = train_data.split(batch_size)
        self.train_targets = train_targets.to(**d_kwargs).split(batch_size)
        self.valid_data = valid_data.to(**d_kwargs).split(batch_size)
        self.valid_targets = valid_targets.to(**d_kwargs).split(batch_size)
        
        self.best_params = self.network.state_dict()
        
    def training_step(self, index):
        self.train()
        self.optimiser.zero_grad()
        prediction = self.network(self.train_data[index])
        loss = self.loss_function(prediction, self.train_targets[index])
        loss.backward()
        self.optimiser.step()
        return loss.detach().cpu()
            
    def get_validation_loss(self):
        self.eval()
        loss_total = 0
        for index in range(len(self.valid_data)):
            prediction = self.network(self.valid_data[index])
            loss = self.loss_function(prediction, self.valid_targets[index])
            loss_total = loss_total + loss
        loss_total = loss_total / len(self.valid_data)
        return loss_total.detach().cpu()
        
    def fit(self, max_steps=5000, forgiveness=50):
        index = 0
        max_index = len(self.train_data) - 1
        steps_since_best = 0
        tracker = torch.zeros((max_steps, 2))
        message = "Step {}, Batch {}, Train Loss {}, Valid Loss {}"
        for step in range(max_steps):
            tracker[step, 0] = self.training_step(index)
            tracker[step, 1] = self.get_validation_loss()
            self.tracker = tracker
            
            print(message.format(
                step, 
                index, 
                tracker[step, 0], 
                tracker[step, 1])
                )
            
            best_step = tracker[:(step+1), 1].argmin()
            if best_step == step:
                steps_since_best = 0
                self.best_params = self.network.state_dict()
            else:
                steps_since_best += 1
                
            if steps_since_best == forgiveness:
                self.network.load_state_dict(self.best_params)
                break
                
            if index == max_index: 
                index = 0
            else:
                index += 1
        return tracker[:(step+1), :].detach().cpu().numpy()
                
    def predict(self, data):
        self.eval()
        prediction = self.network(data.to(**self.d_kwargs))
        return prediction.detach().cpu()
            
class SparseAutoencoderLearner(SupervisedLearner):
    def __init__(
            self,
            network,
            loss_function,
            optimiser,
            optimiser_kwargs,
            train_data,
            train_targets,
            valid_data,
            valid_targets,
            batch_size,
            device=None,
            dtype=None
            ):
        super().__init__(
            network,
            loss_function,
            optimiser,
            optimiser_kwargs,
            train_data,
            train_targets,
            valid_data,
            valid_targets,
            batch_size,
            device=None,
            dtype=None
            )
    
    def predict_sparse_encoding(self, data):
        self.eval()
        prediction = self.network.encoder(data.to(**self.d_kwargs))
        return prediction.detach().cpu()