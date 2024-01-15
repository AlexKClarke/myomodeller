import matplotlib.pyplot as plt

class VisualizeLatentSpace:
    """
    Class used for the visualization of the 2D latent space, parametrised by the mean and variance of each latent

    Inputs:
    - batch of samples
    - trainer_module
    - config (to get latent space size) TODO: shuldn't the model contain info about the size of the latent space already? ot should it be recovered from the config

    Output:
    - latent space plot

    """

    def __init__(
        self,
        data_batch,
        trainer_module,
    ):
        self.data_batch = data_batch
        self.trainer_module = trainer_module

    def plot_latent_space(self):
        (mean_list, var_list) = self.batch_inference(self.data_batch, self.trainer_module)

        #TODO Change in order to plot one latent per axes
        for latent in range(mean_list.shape[1]):
            for i in range(len(mean_list)):
                plt.plot(mean_list.detach().numpy()[i, latent], var_list.detach().numpy()[i, latent])  # change colour for each latent
        plt.show()


    def batch_inference(self, data_batch, trainer_module):
        model = trainer_module.model
        inference_results = model.network.encode(data_batch)

        return inference_results