"""Example of using a checkpoint from a 2D convolutional neural network 
to classify 8x8 MNIST images"""

import os, sys, json

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule
from loader_modules import MNIST
from training.configure import (
    process_loader_module_config,
    process_update_module_config,
)
from visualization_modules.latents_visualization import VisualizeLatentSpace


if __name__ == "__main__":
    # To do inference on new data with a trained model we need the
    # config that specifies the model (this is saved in the logs on training)
    # and the checkpoint which contains the model parameters, which should
    # have been saved in the same location. The path to the checkpoint
    # needs to be added to the config using the "ckpt_path" key.
    # Usually it is easiest to add this to the config json and pass the
    # json path to the TrainingModule rather than a dict. If you look at the
    # inference_example_config.json in the examples folder you can see
    # that an example checkpoint .ckpt has already been added.

    # Pass the json to the TrainingModule and call the get_inference_module
    # method. This returns the update module with the checkpoint network
    # weights replacing the original random weights.

    path = os.path.join("logs/mnist_vae/version_0", "config.json")
    inference_module = TrainingModule(path).get_inference_module()

    # We need data preprocessed in the same way as the training data
    # For convenience we will pull it out the MNIST loader for this example
    with open(path, "r") as file:
        config = json.load(file)
    loader_module = process_loader_module_config(config["loader_module_config"])

    # get the data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = loader_module._get_data(flatten_input=config["loader_module_config"]['loader_module_kwargs']['flatten_input'])

    # PLOT LATENT SPACE
    class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    latent_visualizer = VisualizeLatentSpace(data_train=train_images, labels_train=train_labels, data_test=test_images, labels_test=test_labels, trainer_module=inference_module, loader_module=loader_module, config=config, class_list=class_list)
    latent_visualizer.plot_latent_space()

    '''import matplotlib.pyplot as plt

    pred = prediction[50, :].detach().numpy().reshape(8, 8)

    plt.imshow(pred)
    plt.show()'''
