import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(""))
from networks.sparse import (
    MLPSparseAutoencoder,
    Conv1dSparseAutoencoder,
    Conv2dSparseAutoencoder,
)
from data.mnist import get_64_mnist
from data.preprocessing import get_standardised_library
from visual.plotters import plot_tracker, plot_images, plot_latents
from training.models import SparseAutoencoderLearner


if __name__ == "__main__":

    # select neural network architecture & number of neurons/filters per layer
    architecture = Conv2dSparseAutoencoder
    out_chans_per_layer = [64, 32]
    sparse_dim = 2

    # set parameters of model training
    learning_rate = 0.001
    batch_size = 64

    # modify MNIST dimensions based on architecture
    # Simple MLPs operate on batch x feature
    # Conv1ds operate on batch x channels x time
    # Conv2ds operate on batch x channels x height x width
    if architecture == MLPSparseAutoencoder:
        treat_height_as_channel = False
        flatten_features = True
    elif architecture == Conv1dSparseAutoencoder:
        treat_height_as_channel = True
        flatten_features = False
    elif architecture == Conv2dSparseAutoencoder:
        treat_height_as_channel = False
        flatten_features = False

    # get mnist data, split into train/valid/test data and z-score standardise
    images, targets = get_64_mnist(
        treat_height_as_channel=treat_height_as_channel,
        flatten_features=flatten_features,
    )
    library = get_standardised_library(images, targets, data_splits=[0.6, 0.8])

    # Initialise model with data, architecture and optimiser - then train
    model = SparseAutoencoderLearner(
        network=architecture(
            input_shape=library["train_data"].shape[1:],
            output_shape=library["train_data"].shape[1:],
            sparse_dim=sparse_dim,
            out_chans_per_layer=out_chans_per_layer,
        ),
        loss_function=nn.MSELoss(),
        optimiser=optim.Adam,
        optimiser_kwargs={"lr": learning_rate},
        train_data=library["train_data"],
        train_targets=library["train_data"],
        valid_data=library["valid_data"],
        valid_targets=library["valid_data"],
        batch_size=batch_size,
    )
    tracker = model.fit()

    # Plot training and validation curves
    plot_tracker(tracker)

    # Get prediction on test data and output accuracy
    prediction = model.predict(library["test_data"])
    target = library["test_data"]
    error = (prediction - target).square().mean()
    print("\nSelf-prediction error on test data is %f." % error.numpy())

    # Plot randomly selected prediction next to actual image
    select = int(torch.randint(prediction.shape[0], (1,)))

    def pop(x, select):
        return x.split((select, 1, x.shape[0] - select - 1))[1]

    single_pred_image = pop(prediction, select).squeeze()
    single_target_image = pop(target, select).squeeze()
    if architecture == MLPSparseAutoencoder:
        single_pred_image = single_pred_image.reshape((8, 8))
        single_target_image = single_target_image.reshape((8, 8))
    plot_images(single_pred_image.numpy(), single_target_image.numpy())

    # Get latents across train dataset and plot by digit class
    # (data dimension reduction by t-sne if more than 2)
    latents = model.predict_sparse_encoding(library["train_data"]).numpy()
    labels = torch.argmax(library["train_targets"], dim=1).numpy()
    plot_latents(latents, labels)
