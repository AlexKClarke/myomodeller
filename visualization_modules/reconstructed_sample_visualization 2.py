import matplotlib.pyplot as plt
import torch

class VisualizeReconstructedSamples():
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
        data_batch,
        labels_batch,
        trainer_module,
        loader_module,
        config,
    ):
        self.data_batch = data_batch
        self.labels_batch = labels_batch
        self.trainer_module = trainer_module
        self.loader_module = loader_module
        self.config = config
        self.reconstructed_data = None

    def plot_reconstructed_input(self):
        '''if self.config["loader_module_config"]["loader_module_kwargs"]["flatten_input"] == True:
            self.data_batch = self.data_batch.flatten()'''
        original_data, reconstructed_data = self.batch_inference(self.data_batch, self.trainer_module)

        '''if self.config["loader_module_config"]["loader_module_kwargs"]["flatten_input"] == True:

            self.data_batch = self.data_batch.flatten()'''

        N = original_data.size(0)
        random_numbers = torch.randint(low=0, high=N, size=(10,))
        original_data = original_data[random_numbers].squeeze()
        reconstructed_data = reconstructed_data[random_numbers].squeeze()

        original_data = torch.transpose(original_data, -1, -2)
        reconstructed_data = torch.transpose(reconstructed_data, -1, -2)

        fig, axs = plt.subplots(2, 10, figsize=(20, 10))

        # Plot images from the first tensor in the first column
        for i in range(10):
            axs[0, i].imshow(original_data[i].detach().numpy(), cmap='gray')  # Assuming grayscale images
            axs[0, i].set_title(f'Original image - Sample {i + 1}')
            axs[0, i].axis('off')
        # Plot images from the second tensor in the second column
        for i in range(10):
            axs[1, i].imshow(reconstructed_data[i].detach().numpy(), cmap='gray')  # Assuming grayscale images
            axs[1, i].set_title(f'Reconstructed image - Sample {i + 1}')
            axs[1, i].axis('off')

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()

        self.reconstructed_data = reconstructed_data
        return original_data, reconstructed_data



    def batch_inference(self, data_batch, trainer_module):
        model = trainer_module.model
        if self.config["update_module_config"]["update_module_name"] == 'IOVariationalAutoencoder':
            vae_model = model.network[0]
        else:
            vae_model = model.network

        reconstructed_input = vae_model.forward(data_batch) # returns reconstructed input

        return data_batch, reconstructed_input