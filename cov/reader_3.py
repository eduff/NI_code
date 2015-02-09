#!/usr/bin/python2.6
#import pylab as pl
import numpy as np
import scikits.learn.base
from scipy import linalg
from scipy import stats
from covariance.generate_data import generate_standard_sparse_mvn
from covariance.cov_estimator_l1 import CovEstimatorL1CV
from covariance.cov_estimator_l1 import CovEstimatorL21CV

# Allow python embedding
from IPython.Shell import IPShellEmbed
ipshell = IPShellEmbed('=1')

import glob
import sys

def main():
    
    sys.excepthook = info
    args=(sys.argv[1:])
    run(args)

def run(args):

    inputSubjects=[]
    reducedSubjects=[]

    for file in args:
        print "reading "+str(file)
        inputSubject=np.loadtxt(file)
        print "demeaning columns: "
        means = inputSubject.mean(0)
        inputSubject = inputSubject - means
        inputSubjects.append(inputSubject)
        
    for subject in range(len(inputSubjects)):
        print "reducing subject "+str(subject)
        reducedSubjects.append(inputSubjects[subject])
        print "Found "+str(inputSubjects[subject].shape[0])+" timepoints and "+str(inputSubjects[subject].shape[1])+" components."
        print "Reduced to "+str(reducedSubjects[subject].shape[1])+" components."
        if subject == 0:
            concatSubjects=reducedSubjects[subject]
        else:
            concatSubjects=np.append(concatSubjects,reducedSubjects[subject],axis=0)

    #print "concatenated: "+str(concatSubjects.shape)
    #print "Number of subjects: " + str(len(inputSubjects)) + " with " + str(concatSubjects.shape[1]) + " components."
    #print "Samples per subject: " + str(inputSubjects[0].shape[0])

    labels  = np.repeat(np.arange(len(inputSubjects)), inputSubjects[0].shape[0])

    #print labels.shape
    #print labels

    model = CovEstimatorL21CV()
    model.fit(concatSubjects,labels)
    l1=model.l21

    precs_ = model.precisions
    covs_ = np.array([linalg.inv(prec) for prec in precs_])
    shp=precs_.shape
    tmp=np.zeros((shp[0],shp[1]**2))
    tmp[:,:]=precs_.reshape(precs_.shape[0],'')
    fileName="prec_l1_"+str(l1)+'.txt'
    np.savetxt(fileName,tmp)

    tmp[:,:]=precs_.reshape(covs_.shape[0],'')
    fileName="cov_l1_"+str(l1)+'.txt'
    np.savetxt(str(fileName),tmp)

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

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        print
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)


if __name__ == "__main__":
    sys.exit(main())





