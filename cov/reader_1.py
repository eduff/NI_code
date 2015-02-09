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
    reducedSubjects.append(inputSubjects[subject].take([0,1,5,6,7,8,9,11,12,13,16,17,20,24,25,26,27,28,29,31,34,36,41,42,43,44,49,50,51,52,53,55,56,57,59,61,63,66,67,68,69,70,72,73,78,79,80,82,83,84,85,86,89,91,92,94,95,96,98,99],axis=1))
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







