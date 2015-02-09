# Authors: Gael Varoquaux, Alexandre Gramfort
# Copyright: INRIA

import numpy as np
from scipy import linalg
import pylab as pl

from covariance.generate_data import generate_standard_sparse_mvn
from covariance.cov_estimator_l1 import CovEstimatorL21CV

N_SAMPLES = 100
N_SUBJECTS = 3
DIM = 100

prng = np.random.RandomState(1)
x, true_prec = generate_standard_sparse_mvn(N_SAMPLES*N_SUBJECTS, DIM, prng=prng)
x.shape = (N_SAMPLES*N_SUBJECTS, DIM)
labels  = np.repeat(np.arange(N_SUBJECTS), N_SAMPLES)

emp_covs = np.array([np.dot(x[labels==label].T, x[labels==label])/N_SAMPLES for label in labels])
true_cov = linalg.inv(true_prec)

model = CovEstimatorL21CV()
model.fit(x, labels, maxiter=500, miniter=10, eps=1e-5, use_dgap=False, gap_threshold=1e-9, sparsify=True, verbose=1)
l1 = model.l21

if 1:
 precs_ = model.precisions
 covs_ = np.array([linalg.inv(prec) for prec in precs_])

 ###############################################################################
 # Visualize

 vmin = min(true_cov.min(), emp_covs.min(), covs_.min())
 vmax = max(true_cov.max(), emp_covs.max(), covs_.max())

 pl.figure()
 pl.subplot(2, 2+N_SUBJECTS, 1)
 pl.imshow(true_cov, interpolation='nearest', vmin=vmin, vmax=vmax)
 pl.axis('off')
 pl.title('True (simulated) covariance', fontsize=10)
 pl.subplot(2, 2+N_SUBJECTS, 2)
 pl.imshow(emp_covs[0], interpolation='nearest', vmin=vmin, vmax=vmax)
 pl.axis('off')
 pl.title('sample covariance', fontsize=10)
 for s in range(N_SUBJECTS):
    pl.subplot(2, 2+N_SUBJECTS, 3+s)
    pl.imshow(covs_[s], interpolation='nearest', vmin=vmin, vmax=vmax)
    pl.axis('off')
    pl.title('L1 covariance estimate \n for lambda=%f' % l1, fontsize=10)

 vmin = min(true_prec.min(), precs_.min())
 vmax = max(true_prec.max(), precs_.max())

 pl.subplot(2, 2+N_SUBJECTS, 3+N_SUBJECTS)
 pl.imshow(true_prec, interpolation='nearest', vmin=vmin, vmax=vmax)
 pl.axis('off')
 pl.title('True (simulated) precision', fontsize=10)
 pl.subplot(2, 2+N_SUBJECTS, 4+N_SUBJECTS)
 pl.imshow(linalg.inv(emp_covs[0]), interpolation='nearest', vmin=vmin, vmax=vmax)
 pl.axis('off')
 pl.title('Empirical precision', fontsize=10)
 for s in range(N_SUBJECTS):
    pl.subplot(2, 2+N_SUBJECTS, 5+N_SUBJECTS+s)
    pl.imshow(precs_[s], interpolation='nearest', vmin=vmin, vmax=vmax)
    pl.axis('off')
    pl.title('L21 precision estimate \n for lambda=%f' % l1, fontsize=10)

 pl.show()

