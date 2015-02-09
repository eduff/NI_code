# Authors: Alexandre Gramfort, Gael Varoquaux
# Copyright: INRIA


import numpy
from os.path import join

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('covariance',parent_package,top_path)
    config.add_extension('_cov_estimator_l1',
                         sources=['_cov_estimator_l1.c'],
                         # libraries=['m'],
                         include_dirs=[numpy.get_include()])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
