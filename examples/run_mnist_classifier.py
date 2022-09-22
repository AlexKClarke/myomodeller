import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(""))
from data.mnist import get_64_mnist
from data.preprocessing import get_standardised_library
from networks.blocks import MLPBlock, Conv1dBlock, Conv2dBlock
from training.models import SupervisedLearner
from visual.plotters import plot_tracker

if __name__ == "__main__":

    # select neural network architecture & number of neurons/filters per layer
    architecture = Conv2dBlock
    out_chans_per_layer = [64, 32, 16]

    # set parameters of model training
    learning_rate = 0.001
    batch_size = 64

    # modify MNIST dimensions based on architecture
    # Simple MLPs operate on batch x feature
    # Conv1ds operate on batch x channels x time
    # Conv2ds operate on batch x channels x height x width
    if architecture == MLPBlock:
        treat_height_as_channel = False
        flatten_features = True
    elif architecture == Conv1dBlock:
        treat_height_as_channel = True
        flatten_features = False
    elif architecture == Conv2dBlock:
        treat_height_as_channel = False
        flatten_features = False

    # get mnist data, split into train/valid/test data and z-score standardise
    images, targets = get_64_mnist(
        treat_height_as_channel=treat_height_as_channel,
        flatten_features=flatten_features,
    )
    library = get_standardised_library(images, targets, data_splits=[0.6, 0.8])

    # Initialise model with data, architecture and optimiser - then train
    model = SupervisedLearner(
        network=architecture(
            input_shape=library["train_data"].shape[1:],
            output_shape=library["train_targets"].shape[1:],
            out_chans_per_layer=out_chans_per_layer,
            output_activation=nn.Sigmoid,
        ),
        loss_function=nn.BCELoss(),
        optimiser=optim.Adam,
        optimiser_kwargs={"lr": learning_rate},
        train_data=library["train_data"],
        train_targets=library["train_targets"],
        valid_data=library["valid_data"],
        valid_targets=library["valid_targets"],
        batch_size=batch_size,
    )
    tracker = model.fit()

    # Plot training and validation curves
    plot_tracker(tracker)

    # Get prediction on test data and output accuracy
    prediction = model.predict(library["test_data"]).argmax(1)
    target = library["test_targets"].argmax(1)
    accuracy = 100 * torch.count_nonzero(prediction == target) / target.numel()
    print("\nAccuracy on test data is %d%%." % accuracy.detach())

