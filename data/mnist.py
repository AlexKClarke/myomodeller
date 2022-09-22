import torch
import torch.nn as nn
from sklearn import datasets

def get_64_mnist(
        one_hot_targets=True,
        treat_height_as_channel=False,
        flatten_features=False
        ):
    dataset = datasets.load_digits()
    images = torch.Tensor(dataset.images)
    if treat_height_as_channel is False:
        images = images.unsqueeze(1)
    if flatten_features:
        images = images.flatten(1)
    targets = torch.as_tensor(dataset.target)
    if one_hot_targets:
        targets = nn.functional.one_hot(targets.long(), 10)
    return images, targets
    