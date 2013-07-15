"""
This class requires an ImageMagick install for the 'identify' command.
"""
import os
import os.path
import subprocess
import logging
import scipy
_logger = logging.getLogger(__name__)

import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize

class LocalImages(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, source_directory, axes=('b', 0, 1, 'c'), remove_misfits = False):

        self.axes = axes

        # we define here:
        dtype = 'uint8'
        files = self.file_list_of_source(source_directory)

        # we also expose the following details:
        self.img_shape = self.determine_shape(files[0]) #this is rather dangerous if the first file is not representative to the remainder
        self.img_size = np.prod(self.img_shape)

        # prepare loading
        x = np.zeros((len(files), self.img_size), dtype=dtype)

        # load train data
        X = self.flatten_images(x, files)
        X = global_contrast_normalize(X)

        view_converter = dense_design_matrix.DefaultViewConverter(self.img_shape, self.axes)

        super(LocalImages, self).__init__(X = X, view_converter = view_converter)

    def flatten_images(self, target_matrix, files):
      idx = 0
      for image in files:
        image = scipy.misc.imread(image)
        target_matrix[idx] = np.reshape(image, (-1))
        idx += 1

      return target_matrix


    def determine_shape(self, image_file):
      try:
        output = subprocess.check_output(["identify", "-format", "'%[fx:w]x%[fx:h]'", image_file])
        output = output.strip()
        width, height = output.split("x")
        return (int(width.translate(None, "'")), int(height.translate(None, "'")), 3)
      except Exception as e:
        print "Exception {0} calling 'identify' on {1}".format(e, image_file)
        raise

    def file_list_of_source(self, directory):
      return [ os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) ]

    def adjust_for_viewer(self, X):
        #assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(rval[i,:]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example = False):
        # if the scale is set based on the data, display X oring the scale determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i,:] /= np.abs(orig[i,:]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def get_test_set(self):
      pass
        #return CIFAR10(which_set='test', center=self.center, rescale=self.rescale, gcn=self.gcn,
                #one_hot=self.one_hot, toronto_prepro=self.toronto_prepro, axes=self.axes)
