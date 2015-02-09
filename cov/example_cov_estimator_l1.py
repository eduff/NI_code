# Author: Gael Varoquaux
# Copyright: INRIA
import pylab as pl
import numpy as np
from scipy import linalg

from covariance.generate_data import generate_standard_sparse_mvn
from covariance.cov_estimator_l1 import CovEstimatorL1CV


################################################################################
N_SAMPLES = 30
DIM = 20

prng = np.random.RandomState(10)
x, true_prec = generate_standard_sparse_mvn(N_SAMPLES, DIM, prng=prng)

emp_cov = np.dot(x.T, x)/N_SAMPLES
true_cov = linalg.inv(true_prec)

model = CovEstimatorL1CV(lambdas=np.logspace(-1, 1, 100))
model.fit(x)
l1 = model.best_model.l1


if 1:
 prec_ = model.precision
 cov_ = linalg.inv(prec_)

 #gap, pobj, dobj = model.dual_gap(emp_cov, with_obj=True)
 #print "Dual gap : %s" % gap
 #print "Criterion : %s" % pobj
 #print "Dual criterion : %s" % dobj
 ###############################################################################
 # Visualize

 vmin = min(true_cov.min(), emp_cov.min(), cov_.min())
 vmax = max(true_cov.max(), emp_cov.max(), cov_.max())
 vmax = max(-vmin, vmax)

 pl.figure()
 pl.subplot(2, 3, 1)
 pl.imshow(true_cov, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.axis('off')
 pl.title('True (simulated) covariance', fontsize=10)
 pl.subplot(2, 3, 2)
 pl.imshow(emp_cov, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.axis('off')
 pl.title('sample covariance', fontsize=10)
 pl.subplot(2, 3, 3)
 pl.imshow(cov_, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.axis('off')
 pl.title('L1 covariance estimate \n for lambda=%s' % l1, fontsize=10)

 vmin = min(true_prec.min(), prec_.min())
 vmax = max(true_prec.max(), prec_.max())
 vmax = max(-vmin, vmax)

 pl.subplot(2, 3, 4)
 pl.imshow(true_prec, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.imshow(np.ma.masked_array(np.ones_like(true_prec), true_prec!=0), cmap=pl.cm.gray, interpolation='nearest', vmin=0, vmax=2)
 pl.axis('off')
 pl.title('True (simulated) precision', fontsize=10)
 pl.subplot(2, 3, 5)
 pl.imshow(linalg.inv(emp_cov), interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.axis('off')
 pl.title('Empirical precision', fontsize=10)
 pl.subplot(2, 3, 6)
 pl.imshow(prec_, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
 pl.imshow(np.ma.masked_array(np.ones_like(true_prec), np.abs(prec_)>1e-2), cmap=pl.cm.gray, interpolation='nearest', vmin=0, vmax=2)
 pl.axis('off')
 pl.title('L1 precision estimate \n for lambda=%s' % l1, fontsize=10)
 pl.show()

