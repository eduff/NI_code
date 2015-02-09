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

inputSubjects=[]
reducedSubjects=[]

for file in sys.argv[1:]:
    print "reading "+str(file)
    inputSubjects.append(np.loadtxt(file))
    
for subject in range(len(inputSubjects)):
    print "reducing subject "+str(subject)
    reducedSubjects.append(inputSubjects[subject])
    print "Found "+str(inputSubjects[subject].shape[0])+" timepoints and "+str(inputSubjects[subject].shape[1])+" components."
    print "Reduced to "+str(reducedSubjects[subject].shape[1])+" components."
    if subject == 0:
        concatSubjects=reducedSubjects[subject]
    else:
        concatSubjects=np.append(concatSubjects,reducedSubjects[subject],axis=0)

print "concatenated: "+str(concatSubjects.shape)
print "Number of subjects: " + str(len(inputSubjects)) + " with " + str(concatSubjects.shape[1]) + " components."
print "Samples per subject: " + str(inputSubjects[0].shape[0])

labels  = np.repeat(np.arange(len(inputSubjects)), inputSubjects[0].shape[0])

print labels.shape
print labels

model = CovEstimatorL21CV()
model.fit(concatSubjects,labels)
l1=model.l21

precs_ = model.precisions
covs_ = np.array([linalg.inv(prec) for prec in precs_])

for subject in range(len(inputSubjects)):
   fileName="prec_subject"+str(subject)+"_l1_"+str(l1)
   np.savetxt(str(fileName),precs_[subject])
   fileName="cov_subject"+str(subject)+"_l1_"+str(l1)
   np.savetxt(str(fileName),covs_[subject])

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







