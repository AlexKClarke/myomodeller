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
        data_test,
        labels_test,
        trainer_module,
        loader_module,
        config,
    ):
        self.data_test = data_test
        self.labels_test = labels_test
        self.trainer_module = trainer_module
        self.loader_module = loader_module
        self.config = config

    def plot_reconstructed_input(self):

        original_data, reconstructed_data = self.batch_inference(self.data_test, self.trainer_module)

        N = original_data.size(0)
        random_numbers = torch.randint(low=0, high=N, size=(2,))
        original_data = original_data[random_numbers].squeeze()
        reconstructed_data = reconstructed_data[random_numbers].squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Plot images from the first tensor in the first column
        for i in range(2):
            axs[i, 0].imshow(original_data[i].detach().numpy(), cmap='gray')  # Assuming grayscale images
            axs[i, 0].set_title(f'Original image - Sample {i + 1}')
            axs[i, 0].axis('off')
        # Plot images from the second tensor in the second column
        for i in range(2):
            axs[i, 1].imshow(reconstructed_data[i].detach().numpy(), cmap='gray')  # Assuming grayscale images
            axs[i, 1].set_title(f'Reconstructed image - Sample {i + 1}')
            axs[i, 1].axis('off')

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()



    def batch_inference(self, data_batch, model):

        if self.config["update_module_config"]["update_module_name"] == 'IOVariationalAutoencoder':
            vae_model = model.network[0]
        else:
            vae_model = model.network

        reconstructed_input = vae_model.forward(data_batch) # returns reconstructed input

        return data_batch, reconstructed_input