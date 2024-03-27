import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
        data_train,
        labels_train,
        data_test,
        labels_test,
        trainer_module,
        loader_module,
        config,
        class_list
    ):
        self.data_train = data_train
        self.labels_train = labels_train
        self.data_test = data_test
        self.labels_test = labels_test
        self.trainer_module = trainer_module
        self.loader_module = loader_module
        self.config = config
        self.class_list = class_list

    def plot_latent_space(self):
        (mean_list, var_list) = self.batch_inference(self.data_train, self.trainer_module)
        principal_components_train = self.pca(mean_list.detach().numpy())

        (mean_list, var_list) = self.batch_inference(self.data_test, self.trainer_module)
        principal_components_test = self.pca(mean_list.detach().numpy())


        # Create a figure with two subplots side by side
        plt.figure(figsize=(10, 4))

        # Mean scatter plot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.title('Train Data')

        for i in range(len(self.class_list)):
            indices = self.labels_train == i
            plt.scatter(principal_components_train[indices, 0], principal_components_train[indices, 1],
                        label=f'{self.class_list[i]}')

        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.legend()
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, first subplot

        plt.title('Test Data')
        for i in range(len(self.class_list)):
            indices = self.labels_test == i
            plt.scatter(principal_components_test[indices, 0], principal_components_test[indices, 1],
                        label=f'{self.class_list[i]}')

        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.legend()

        # Adjust layout for better spacing
        plt.tight_layout()
        # Show the plots
        plt.show()

        return principal_components_train, principal_components_test



    def batch_inference(self, data_batch, model):

        # CHECK IF IOVAE
        if self.config["update_module_config"]["update_module_name"] == 'IOVariationalAutoencoder':
            network = model.network[0]
        else:
            network = model.network

        # CHECK IF FLATTEN IS NEEDED
        '''if self.config["loader_module_config"]["loader_module_kwargs"]["flatten_input"] == True:
            data_batch = data_batch.reshape(data_batch.size()[0], -1)'''
        latent_space = network.encode(data_batch) # returns mean and variance of the latents

        return latent_space

    def pca(self, latent_mean):
        # Step 1: Center the Data
        mean_centered_data = latent_mean - np.mean(latent_mean, axis=0)

        # Step 2: Compute the Covariance Matrix
        covariance_matrix = np.cov(mean_centered_data, rowvar=False)

        # Step 3: Compute Eigenvectors and Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Select the Top Two Eigenvectors
        top_eigenvectors = eigenvectors[:, :2]

        # Step 5: Project the Data onto the Principal Components
        principal_components_test = np.dot(mean_centered_data, top_eigenvectors)

        return principal_components_test