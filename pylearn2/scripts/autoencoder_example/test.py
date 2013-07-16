import numpy as np
from scipy import misc
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing

cifar = cifar10.CIFAR10('train')
cifar.apply_preprocessor(preprocessing.Standardize(), can_fit=True)
