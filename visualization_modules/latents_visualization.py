import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('macosx')



class VisualizeLatentSpace:
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

    def plot_latent_space(self):
        (mean_list, var_list) = self.batch_inference(self.data_batch, self.trainer_module)

        # Create a figure with two subplots side by side
        '''plt.figure(figsize=(12, 6))'''

        # Mean scatter plot
        '''plt.subplot(1, 2, 1)'''  # 1 row, 2 columns, first subplot
        scatter1 = plt.scatter(mean_list.detach().numpy()[:, 0], mean_list.detach().numpy()[:, 1], c=self.labels_batch, cmap='viridis')  # change colour for each latent
        plt.title('Mean Plot')
        plt.xlabel('Latent 1')
        plt.ylabel('Latent 2')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plots
        plt.show()

        return mean_list, var_list



    def batch_inference(self, data_batch, trainer_module):
        model = trainer_module.model

        # CHECK IF IOVAE
        if self.config["update_module_config"]["update_module_name"] == 'IOVariationalAutoencoder':
            vae_model = model.network[0]
        else:
            vae_model = model.network

        # CHECK IF FLATTEN IS NEEDED
        if self.config["loader_module_config"]["loader_module_kwargs"]["flatten_input"] == True:
            data_batch = data_batch.reshape(data_batch.size()[0], -1)
        inference_results = vae_model.encode(data_batch) # returns mean and variance of the latents

        return inference_results