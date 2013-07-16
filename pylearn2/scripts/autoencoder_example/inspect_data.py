import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylearn2.datasets.preprocessing
import pylearn2.datasets.cifar10

cifar = pylearn2.datasets.cifar10.CIFAR10('train')
pipeline = pylearn2.datasets.preprocessing.Pipeline()

pipeline.items.append(pylearn2.datasets.preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))

cifar.apply_preprocessor(pipeline)

hist, bins = np.histogram(cifar.X[0],bins = 50)

width = 0.7*(bins[1]-bins[0])

center = (bins[:-1]+bins[1:])/2

plt.bar(center, hist, align = 'center', width = width)

plt.savefig("/var/www/hist.png")
