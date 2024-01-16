import matplotlib.pyplot as plt

class VisualizeLatentSpace:
    """
    Class used for the visualization of the 2D latent space, parametrised by the mean and variance of each latent

    Inputs:
    - batch of samples
    - trainer_module
    - config (to get latent space size)

    Output:
    - latent space plot

    """

    def __init__(
        self,
        data_batch,
        labels_batch,
        trainer_module,
    ):
        self.data_batch = data_batch
        self.labels_batch = labels_batch
        self.trainer_module = trainer_module

    def plot_latent_space(self):
        (mean_list, var_list) = self.batch_inference(self.data_batch, self.trainer_module)

        # Create a figure with two subplots side by side
        plt.figure(figsize=(12, 6))

        # Mean scatter plot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        scatter1 = plt.scatter(mean_list.detach().numpy()[:, 0], mean_list.detach().numpy()[:, 1], c=self.labels_batch, cmap='viridis')  # change colour for each latent
        plt.title('Mean Plot')
        plt.xlabel('Latent 1')
        plt.ylabel('Latent 2')

        # Variance scatter plot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        scatter2 = plt.scatter(var_list.detach().numpy()[:, 0], var_list.detach().numpy()[:, 1], c=self.labels_batch, cmap='viridis')  # change colour for each latent
        plt.title('Variance Plot')
        plt.xlabel('Latent 1')
        plt.ylabel('Latent 2')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plots
        plt.show()



    def batch_inference(self, data_batch, trainer_module):
        model = trainer_module.model
        inference_results = model.network.encode(data_batch)

        return inference_results