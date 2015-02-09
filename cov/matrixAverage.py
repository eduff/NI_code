#!/usr/bin/python2.6
import pylab as pl
import numpy as np
from scipy import linalg
from scipy import stats
from covariance.generate_data import generate_standard_sparse_mvn
from covariance.cov_estimator_l1 import CovEstimatorL1CV
from covariance.cov_estimator_l1 import CovEstimatorL21CV

import glob
import sys

inputSubjects=np.loadtxt(sys.argv[1])
nSubjects=1

for file in sys.argv[2:]:
    print "reading "+str(file)
    inputSubjects+=np.loadtxt(file)
    nSubjects+=1

print inputSubjects
print inputSubjects/nSubjects
print inputSubjects.shape[0]
print inputSubjects.shape[1]


#model = CovEstimatorL1CV()
#model.fit(reducedSubjects[0])
#l1 = model.best_model.l1
#
#if 1:
# prec_ = model.precision
#
#pl.figure()
#vmin = stats.scoreatpercentile(prec_.ravel(),2)
#vmax = stats.scoreatpercentile(prec_.ravel(),98)
#pl.imshow(prec_, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=pl.cm.YlOrRd)
#cbar2=pl.colorbar()
#pl.show()







