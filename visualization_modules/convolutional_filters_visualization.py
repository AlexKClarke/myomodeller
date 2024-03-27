import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class VisualizeConvolutionalFilters():
    """
    Class used for the visualization of the 2D latent space, parametrised by the mean and variance of each latent

    Inputs:
    - batch of samples
    - trainer_module
    - loader_module
    - config (to get latent space size)

    Output:
    - latent space plots

    """

    def __init__(
        self,
        model,
    ):
        self.model = model


    def plot_convolutional_filters(self):

        # Extract convolutional layers
        conv_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]

        # Display filters for each convolutional layer
        for idx, layer in enumerate(conv_layers):
            print("Layer:", idx)
            filters = layer.weight

            filter_count = filters.shape[0]
            plt.figure(figsize=(8, 8))
            plt.title(f"Conv. Layer: {idx}")
            for i in range(filter_count):
                plt.subplot(int(filter_count / 4), 4, i + 1)
                plt.imshow(filters[i, 0, :, :].detach().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()

