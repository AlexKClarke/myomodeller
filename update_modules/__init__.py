"""
All update module subclasses need to be placed in this init to be seen by the 
training module. See training.core for the parent class.
"""

from update_modules.supervised_classifier import SupervisedClassifier
from update_modules.supervised_regressor import SupervisedRegressor
from update_modules.sparse_autoencoder import SparseAutoencoder
