"""
All loader module subclasses need to be placed in this init to be seen by the 
training module. See training.core for the parent class.
"""

from loader_modules.mnist import MNIST
from loader_modules.mnist28 import MNIST28
from loader_modules.mu_labelled_emg import MotorUnitLabelledEMG
from loader_modules.raw_emg import RawEMGLabelled
