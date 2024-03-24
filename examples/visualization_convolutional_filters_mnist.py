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
from visualization_modules.reconstructed_sample_visualization import VisualizeReconstructedSamples
from visualization_modules.convolutional_filters_visualization import VisualizeConvolutionalFilters


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

    path = os.path.join("logs/mnist_vae/version_1", "config.json")
    inference_module = TrainingModule(path).get_inference_module()

    model = inference_module.network.encoder.block
    convolutional_filters_visualizer = VisualizeConvolutionalFilters(model=model)
    convolutional_filters_visualizer.plot_convolutional_filters()


