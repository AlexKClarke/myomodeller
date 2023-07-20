"""Example of using a checkpoint from a 2D convolutional neural network 
to classify 8x8 MNIST images"""

import os, sys

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
sys.path.append(os.path.abspath(""))

from training import TrainingModule
from loader_modules import MNIST


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
    path = os.path.join("examples", "inference_example_config.json")
    inference_module = TrainingModule(path).get_inference_module()

    # We need data preprocessed in the same way as the training data
    # For convenience we will pull it out the MNIST loader for this example
    images, labels = MNIST()._get_data(one_hot_labels=False)[:2]

    # For inference you can just pass the data to the inference module
    # Here we use the prediction to calculate an accuracy
    prediction = inference_module(images).argmax(1)
    score = (prediction == labels).sum() / prediction.shape[0]
    print("Accuracy on data is " + str(int(100 * score)) + "%")
