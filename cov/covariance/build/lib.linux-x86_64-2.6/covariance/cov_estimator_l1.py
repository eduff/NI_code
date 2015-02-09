"""
Estimate a sparse covariance with l1 regularization.

Based on 'sparse permutation-invariant covariance estimate' (SPICE) procedure
Rothman et al.,Electronic Journal of Statistics
Vol. 2 (2008) 494-515
"""

# Authors: Gael Varoquaux, Alexandre Gramfort
# Copyright: INRIA

import numpy as np
from scipy import linalg

from scikits.learn.base import BaseEstimator
from scikits.learn.cross_val import KFold, LeaveOneLabelOut

from logdet import sym_inv
from logdet import exact_logdet as logdet


try:
    from _cov_estimator_l1 import _choleski_update_l1
except:
    # On scalars, math.sqrt is faster than np.sqrt
    from math import sqrt
    print "WARNING : use CYTHON"
    def positive_root(a, b, c):
        """
        solve the second order equation and take the positive root
        """
        delta = b**2 - 4*a*c
        if delta<0:
            raise ValueError, 'Complex values not handled'
        return (-b+sqrt(delta))/(2*a)

    def _choleski_update_l1(emp_cov, precision, abs_inv_prec0, l1, chol_prec=None):
        """ Inner step of cholesky update

            Parameters
            ----------
            emp_cov, array of shape (n, n),
                the sample covariance of the obervations
            abs_inv_prec0, array of shape (n, n),
                inverse of the absolute values of the intial precision
                The diagonal should be to zero!!!

            Warning: precision and chol_prec are modified IN PLACE.
        """
        # XXX: Cythonize for more speed?
        if chol_prec is None:
            chol_prec = linalg.cholesky(precision, lower=True)

        # Update of non-diagonal parameters
        old_chol_prec = chol_prec.copy()
        for l in range(emp_cov.shape[0]):
            # this_chol_line is a view on the precision cholesky, it is
            # modified _inplace_ to modify the original array itself
            this_chol_line = chol_prec[l]
            for c in range(l):
                this_precision_line = precision[c]
                pa = np.dot(this_chol_line, emp_cov[:,c])
                pa -= this_chol_line[c]*emp_cov[c,c]
                pb = np.dot((  this_precision_line
                                - this_chol_line*this_chol_line[c]),
                        this_chol_line*abs_inv_prec0[c])
                pd =  np.dot(this_chol_line**2, abs_inv_prec0[c])

                # Update the choleski inplace
                this_chol_line[c] = -(pa + l1*pb)/(emp_cov[c,c] + l1*pd)

                # Update the precision
                pcc = (  this_precision_line[c]
                        + this_chol_line[c]**2
                        - old_chol_prec[l,c]**2 )
                this_precision_line += (this_chol_line
                                    *(this_chol_line[c] - old_chol_prec[l,c]))
                precision[:,c] = this_precision_line
                precision[c,c] = pcc

        # Update of the diagonal terms
        for l in range(emp_cov.shape[0]):
            # Inplace operations for speed
            pa = np.dot(chol_prec[l]**2, abs_inv_prec0[l])
            pa = emp_cov[l,l] + l1*pa
            pd = np.dot(chol_prec[l], emp_cov[:,l])
            pd -= (chol_prec[l,l]*emp_cov[l,l])
            pb = np.dot((precision[l] - chol_prec[l,l]*chol_prec[l]),
                        chol_prec[l]*abs_inv_prec0[l])
            pb = pd + l1*pb

            # Update the choleski
            chol_prec[l,l] = positive_root(pa, pb, -1.)

            # Update the precision
            pll = precision[l,l] + chol_prec[l,l]**2-old_chol_prec[l,l]**2
            precision[l] += chol_prec[l]*(chol_prec[l,l]-old_chol_prec[l,l])
            precision[:,l] = precision[l]
            precision[l,l] = pll

        return precision, chol_prec


def regularized_inverse(M, l2=None):
    """
    L2-regularized inverse of a sp matrix M

    Parameters
    ----------
    M, array of shape (dim,dim), assume symmetric positive
    l2=None, float, regularization index

    if l2==None, the regularization is set as trace(M)/dim
    """
    dim = M.shape[0]
    if l2 is None:
        l2 = np.trace(M)/dim

    return linalg.inv(M + l2*np.eye(dim))

       

################################################################################
# class CovEstimatorL1
################################################################################

class CovEstimatorL1(BaseEstimator):
    """ Covariance estimator with an L1 regularisation.

        Attributes
        ----------
        l1, float
            The L1 penalty used to obtain sparse precision estimate
    """

    def __init__(self, l1):
        if l1<0:
            raise ValueError, 'l1 must be non-negative'
        self.l1 = l1

    #-------------------------------------------------------------------------
    # scikit-learn Estimator interface
    #-------------------------------------------------------------------------

    def fit(self, X, prec=None,
                        maxiter=100, miniter=10, sparsify=False,
                        stop_early=True,
                                gap_threshold=.2, eps=1.e-4, verbose=0):
        """ Fit the model to the given covariance.

        Parameters
        ----------
        X: array of shape (n, p)
            The obervations
        prec: None or array of shape (p, p), optional
            The initial guess for the precision
        maxiter: int, optional
            Maximum number iterations
        miniter: int, optional
            Minimum number of iterations
        eps: float, optional
            Machine precision limit, to regularise inverses
        gap_threshold, float, optional
            The value of the dual gap to reach to declare convergence.
        verbose: bool, optional
            Verbosity mode
        stop_early: boolean, optional
            Stop on percent change in precision in addition to dual_gap
        """
        n_samples = X.shape[0]
        emp_cov = np.dot(X.T, X)/n_samples
        emp_cov = np.ascontiguousarray(emp_cov)
        # Initial values
        if prec is None:
            prec = regularized_inverse(emp_cov)
        else:
            prec = np.ascontiguousarray(prec)
        self.precision = prec
        self.chol_prec = linalg.cholesky(prec, lower=True)

        prec_old = self.precision.copy()
        # Iterate choleski re-estimation and update of all parameters
        for i in range(maxiter):
            self._update_cholesky(emp_cov, eps=eps)

            if (i>miniter-1) and i % 3 == 0:
                # Check the dual_gap every third iteration, to speed
                # up (the factor of 3 is chosen because of the ratio
                # in speed between main loop and dual gap computing.
                gap, p_obj, d_obj = self.dual_gap(emp_cov, with_obj=True)
                if verbose:
                    print 'Iteration: % 2i, cost func: %s, dual gap: %s' % (
                        i, p_obj, gap)

                if (gap<gap_threshold):
                    break

                if stop_early:
                    # Dual gap is very noisy and unreliable (because the
                    # dual variable drops off the cone), we use a second
                    # stopping criteria
                    prec_old_norm = (prec_old**2).sum()
                    if ( ((prec_old - self.precision)**2).sum() 
                                                < eps**2*prec_old_norm):
                        if verbose:
                            print 'Stopping without dual gap convergence'
                        break
                    prec_old = self.precision.copy()
        else:
            if verbose:
                import sys
                print >>sys.stderr, 'Maximum number of iterations reached'

        if sparsify:
            self.precision[np.abs(self.precision) < eps**2] = 0.0

        return self


    #-------------------------------------------------------------------------
    # scikit-learn Model interface
    #-------------------------------------------------------------------------

    def log_likelihood(self, test_cov):
        """ Estimate the likelihood of a given covariance under the model.
        """
        return -np.sum(test_cov*self.precision) + self.get_log_det_prec()


    def score(self, X_test):
        n_samples = X_test.shape[0]
        test_cov = np.dot(X_test.T, X_test)/n_samples
        return self.log_likelihood(test_cov)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _update_cholesky(self, emp_cov, maxiter=10, eps=1.e-4, 
                    regularization=1e-8, verbose=0):
        """
        Core of spice: cholesky update

        Parameters
        ----------
        emp_cov: array of shape (n, n),
            the sample covariance of the obervations
        maxiter: int, optional
            max. number iterations
        eps: float, optional
            threshold value on precision change to declare convergence
        regularization: float, optional
            regularization of the inverse
        verbose: integer, optional
            verbosity mode
        """
        chol_prec_old = self.chol_prec.copy()
        delta = eps*np.sum(chol_prec_old**2)

        initial_abs_prec = np.absolute(self.precision)
        # Regularized 1/initial_abs_prec:
        abs_inv_prec0 = 1/(initial_abs_prec + regularization)
        # Put the diagonal to zero
        p = abs_inv_prec0.shape[0]
        abs_inv_prec0.flat[::p+1] = 0

        for i in range(maxiter):
            _choleski_update_l1(emp_cov, self.precision, abs_inv_prec0,
                                    self.l1, chol_prec=self.chol_prec)
            diff = np.sum((chol_prec_old - self.chol_prec)**2)
            if diff<delta:
                if verbose:
                    print i, diff, delta
                break
            chol_prec_old = self.chol_prec.copy()

        # Don't forget to update the precision
        self.precision = np.dot(self.chol_prec.T, self.chol_prec)

    def cost_func(self, emp_cov):
        """ Return the cost function of the optimisation for the provided
            covariance  matrix

            Parameters
            ----------
            emp_cov = None, array of shape (self.dim,self.dim)
                the empirical covariance matrix
                if None, self.empcov is used instead
        """
        cost = -self.log_likelihood(emp_cov)
        cost += self.penalty()
        return cost

    def penalty(self):
        """ Returns the penalty term in the cost function
        """
        l1term = (np.sum(np.absolute(self.precision))
                  -np.sum(np.absolute(np.diag(self.precision))))
        return self.l1*l1term

    def dual_obj(self, emp_cov, A):
        B = A + emp_cov
        B *= .5
        return logdet(B + B.T) + A.shape[0]

    def dual_variable(self, emp_cov):
        #W = sym_inv(self.precision) - emp_cov
        W = linalg.inv(self.precision) - emp_cov
        # Maximum abs value of an off diagonal term
        diag_stride = W.shape[0] + 1
        W.flat[::diag_stride] = 0.0 # diag of A should be filled with 0
        # Scale dual variable if necessary
        W /= np.maximum(np.absolute(W)/self.l1, 1)
        return W

    def dual_gap(self, emp_cov, W=None, with_obj=False):
        """ Compute duality gap to check optimality of estimated
            precision with the given covariance.
        """
        if with_obj:
            W = self.dual_variable(emp_cov)
            p_obj = self.cost_func(emp_cov)
            d_obj = self.dual_obj(emp_cov, W)
            # Compute duality gap
            gap = p_obj - d_obj
            return gap, p_obj, d_obj
        else:
            gap = 0
            gap += np.sum(emp_cov * self.precision)
            gap += self.penalty() - self.precision.shape[0]
            return gap

    def get_log_det_prec(self):
        """ Returns the determinant of self.precision
        """
        out = 2*np.sum(np.log(np.diag(self.chol_prec)))
        if np.isnan(out):
            out = -np.inf
        return out


################################################################################
# class BaseCovEstimatorCV
################################################################################

class BaseCovEstimatorCV(BaseEstimator):

    def __init__(self, n_lambdas=20, lambdas=None):
        if lambdas is not None:
            n_lambdas = len(lambdas)
        self.lambdas   = lambdas
        self.n_lambdas = n_lambdas


    def fit(self, X, cv=None, **fit_params):
        X = np.asanyarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        if self.lambdas is None:
            lambdas = np.logspace(0, -4, self.n_lambdas)
        else:
            lambdas = np.sort(self.lambdas)[::-1]

        n_lambdas = len(lambdas)
        # init cross-validation generator
        cv = cv if cv else KFold(n_samples, 5)

        scores = np.zeros(n_lambdas)
        for train, test in cv:
            models_train = self.path(X[train], lambdas=lambdas,
                                    fit_params=fit_params)
            for i_lambda, model in enumerate(models_train):
                scores[i_lambda] += model.score(X[test])

        i_best_lambda = np.argmax(scores)
        best_lambda   = lambdas[i_best_lambda]
        model = self.path(X, lambdas=[best_lambda,],
                                    fit_params=fit_params)[0]
        self.precision = model.precision
        param_name, param_value = model._get_params().items()[0]
        setattr(self, param_name, param_value)
        self.score = model.score 
        self.best_model = model
        self.scores = scores
        return self

    def score(self, *args):
        return self.best_model.score(*args)

################################################################################
# class CovEstimatorL1CV
################################################################################

def l1_cov_path(X, lambdas, verbose=False, fit_params=dict()):
    prec = None # init prec
    models = []
    n_samples = X.shape[0]
    emp_cov = np.dot(X.T, X)/n_samples

    for l1 in lambdas:
        model = CovEstimatorL1(l1=l1)
        model.fit(X, prec=prec, **fit_params)
        # Heuristic to find a good starting point
        prec = .5*model.precision + .5*regularized_inverse(emp_cov)
        models.append(model)
        if verbose:
            print '%s, dual gap: %s' % (model, 
                                model.dual_gap(emp_cov, with_obj=True)[0])
    return models


class CovEstimatorL1CV(BaseCovEstimatorCV):
    path = staticmethod(l1_cov_path)


################################################################################
# class CovEstimatorL21
################################################################################

class CovEstimatorL21(CovEstimatorL1):

    def __init__(self, l21):
        if l21<0:
            raise ValueError, 'l21 must be non-negative'
        self.l21 = l21

    #-------------------------------------------------------------------------
    # scikit-learn Estimator interface
    #-------------------------------------------------------------------------

    def fit(self, X, labels, maxiter=100, miniter=10, sparsify=False,
                                precs=None,
                                stop_early=True,
                                use_dgap=True,
                                gap_threshold=.1, eps=1.e-4, verbose=0):
        """ Fit the model to the given covariances.

        Parameters
        ----------
        X: array of shape (n*g, p)
            The observations of the each of the obervation groups
        labels: array of shape (n*g)
            The labels corresponding to each group
        maxiter: int, optional
            Maximum number iterations
        miniter: int, optional
            Minimum number of iterations
        eps: float, optional
            Machine precision limit, to regularise inverses
        gap_threshold, float, optional
            The value of the dual gap to reach to declare convergence.
        verbose: bool, optional
            Verbosity mode
        """
        # Initial values
        group_idx   = np.unique(labels)
        # Store the labels corresponding to the precisions, 
        # for reuse them while testing
        # XXX: not using them currently: will fail for non contiguous
        # labels
        self.labels = group_idx
        n_groups    = len(group_idx)
        dim         = X.shape[1]
        shape       = (n_groups, dim, dim)
        compute_prec = (precs is None)
        precs       = np.empty(shape)
        chol_precs  = np.empty(shape)
        emp_covs    = np.empty(shape)
        for group_id, prec, chol_prec, emp_cov in zip(
                    group_idx, precs, chol_precs, emp_covs):
            this_X = X[labels == group_id]
            n_samples    = len(this_X)
            emp_cov[...] = np.dot(this_X.T, this_X)/n_samples
            if compute_prec:
                prec[...] = regularized_inverse(emp_cov)
            chol_prec[...] = linalg.cholesky(prec, lower=True)
        self.precisions = precs
        self.chol_precs = chol_precs

        prec_old = self.precisions.copy()
        # Iterate choleski re-estimation and update of all parameters
        for i in range(maxiter):
            self._update_cholesky(emp_covs, eps=eps)

            if (i>miniter-1) and i % 3 == 0:
                # Check the dual_gap every third iteration, to speed
                # up (the factor of 3 is chosen because of the ratio
                # in speed between main loop and dual gap computing.
                if use_dgap:
                    gap, p_obj, d_obj = self.dual_gap(emp_covs, with_obj=True)
                    if verbose:
                        print 'Iteration: % 2i, cost func: %s, dual gap: %s' % (
                            i, p_obj, gap)

                    if (gap<gap_threshold):
                        break

                if stop_early:
                    # Dual gap is very noisy and unreliable (because the
                    # dual variable drops off the cone), we use a second
                    # stopping criteria
                    prec_old_norm = (prec_old**2).sum()
                    if ( ((prec_old - self.precisions)**2).sum() 
                                                < eps**2*prec_old_norm):
                        if verbose:
                            print 'Stopping without dual gap convergence'
                        break
                    prec_old = self.precisions.copy()
        else:
            if verbose:
                import sys
                print >>sys.stderr, 'Maximum number of iterations reached'

        if sparsify:
            norms = np.sqrt(np.sum(self.precisions**2, axis=0))
            mask = norms < eps**2
            for precision in self.precisions:
                precision[mask] = 0.0

        return self


    #-------------------------------------------------------------------------
    # scikit-learn Model interface
    #-------------------------------------------------------------------------

    def log_likelihood(self, test_covs, labels=None):
        """ Estimate the likelihood of a given covariance under the model.
        """
        if labels is None:
            labels = self.labels
        labels_idx = np.searchsorted(self.labels, labels)
        loglik = -sum([np.sum(test_cov*precision)
                     for test_cov, precision in zip(test_covs, 
                                        self.precisions[labels_idx])
                      ])
        loglik += sum(self.get_log_det_prec(labels=labels))
        return loglik


    def score(self, X_test, labels_test):
        group_idx   = np.unique(labels_test)
        n_groups    = len(group_idx)
        dim         = X_test.shape[1]
        shape       = (n_groups, dim, dim)
        test_covs    = np.empty(shape)
        for group_id, test_cov in zip(group_idx, test_covs):
            this_X = X_test[labels_test == group_id]
            n_samples     = len(this_X)
            test_cov[...] = np.dot(this_X.T, this_X)/n_samples
        return self.log_likelihood(test_covs, labels=labels_test)


    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _update_cholesky(self, emp_covs, maxiter=10, eps=1.e-4, verbose=0):
        """
        Core of spice: cholesky update

        Parameters
        ----------
        emp_covs: array of shape (n, p, p),
            the sample covariances for each group of obervations
        maxiter: int, optional
            max. number iterations
        eps: float, optional
            threshold value on precision change to declare convergence
        verbose: integer, optional
            verbosity mode
        """
        chol_precs_old = self.chol_precs.copy()
        delta = eps*np.sum(np.sum(chol_precs_old**2, axis=-1),
                            axis=-1).min()

        # implement a L21 norm
        initial_abs_precs = np.sqrt(np.sum(self.precisions**2, axis=0))
        # Regularized 1/initial_abs_prec:
        abs_inv_prec0 = emp_covs.shape[0]/(initial_abs_precs + eps**2)
        # Put the diagonal to 0
        p = abs_inv_prec0.shape[0]
        abs_inv_prec0.flat[::p+1] = 0

        for i in range(maxiter):
            # L21: Call the L1 problem for all chol with the same
            # penalty (initial_abs_prec)
            for emp_cov, precision, chol_prec in zip(
                                emp_covs, self.precisions, self.chol_precs):
                _choleski_update_l1(emp_cov, precision, abs_inv_prec0,
                                        self.l21, chol_prec=chol_prec)
            diff = np.sum(np.sum((chol_precs_old - self.chol_precs)**2,
                            axis=-1), axis=-1).min()
            if diff < delta:
                if verbose:
                    print i, diff, delta
                break
            chol_precs_old = self.chol_precs.copy()

        # Don't forget to update the precision
        for precision, chol_prec in zip(self.precisions, self.chol_precs):
            precision[...] = np.dot(chol_prec.T, chol_prec)

    def penalty(self):
        """ Returns the penalty term in the cost function
        """
        norms = np.sqrt(np.sum(self.precisions**2, axis=0))
        diag_stride = self.precisions.shape[-1] + 1
        norms.flat[::diag_stride] = 0.0
        return self.l21 * norms.sum()

    def get_log_det_prec(self, labels=None):
        """ Returns the determinant of each precision in self.precisions
        """
        if labels is None:
            labels = self.labels
        labels_idx = np.searchsorted(self.labels, labels)
        return [2*np.sum(np.log(np.diag(chol_prec)))
                for chol_prec in self.chol_precs[labels_idx]]

    def dual_variables(self, emp_covs):
        """ Compute dual variables from emp_covs and precisions
        """
        # Compute dual objective
        Ws = np.empty_like(emp_covs)
        diag_stride = Ws.shape[-1]+1
        for emp_cov, precision, W in zip(emp_covs, self.precisions, Ws):
            W[:] = linalg.inv(precision) - emp_cov
            # Maximum abs value of an off diagonal term
            W.flat[::diag_stride] = 0.0

        # Scale dual variable if necessary
        scaling = np.maximum(np.sqrt(np.sum(Ws**2, axis=0)) / self.l21, 1)
        Ws /= scaling[np.newaxis, ...]
        return Ws

    def dual_obj(self, emp_covs, Ws=None):
        """ Compute value of dual objective function
        """
        if Ws is None:
            Ws = self.dual_variables(emp_covs)
        dobj = 0.0
        for emp_cov, W in zip(emp_covs, Ws):
             dobj += logdet(emp_cov + W) + W.shape[0]
        return dobj

    def dual_gap(self, emp_covs, Ws=None, with_obj=False):
        """ Compute duality gap to check optimality of estimated
            precision with the given covariance.
        """
        if with_obj:
            p_obj = self.cost_func(emp_covs)
            d_obj = self.dual_obj(emp_covs, Ws)
            # Compute duality gap
            gap = p_obj - d_obj
            return gap, p_obj, d_obj
        else:
            gap = 0.0
            for emp_cov, precision in zip(emp_covs, self.precisions):
                gap += np.sum(emp_cov * precision) - precision.shape[0]
            gap += self.penalty()
            return gap


################################################################################
# class LeaveHalfALabelOut
################################################################################

class LeaveHalfALabelOut(LeaveOneLabelOut):
    """
    Leave Half a Label Out cross-validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, labels):
        """
        Leave Half a Label Out cross-validation iterator:
        Provides train/test indexes to split data in train test sets
    
        Parameters
        ----------
        labels : list
                List of labels
            
        Examples
        ----------
        >>> from parietal.learn.covariance.cov_estimator_l1 import LeaveHalfALabelOut
        >>> N_SUBJECTS = 2
        >>> N_SAMPLES = 2
        >>> labels  = np.repeat(np.arange(N_SUBJECTS), N_SAMPLES)
        >>> labels
        array([0, 0, 1, 1])
        >>> cv = cross_val.LeaveHalfALabelOut(labels)
        >>> len(cv)
        4
        >>> for train_mask, test_mask in cv:
        ...    print "TRAIN:", train_mask, "TEST:", test_mask
        TRAIN: [ True False  True  True] TEST: [False  True False False]
        TRAIN: [False  True  True  True] TEST: [ True False False False]
        TRAIN: [ True  True  True False] TEST: [False False False  True]
        TRAIN: [ True  True False  True] TEST: [False False  True False]

        """
        LeaveOneLabelOut.__init__(self, labels)


    def __iter__(self):
        for train_mask, test_mask in LeaveOneLabelOut.__iter__(self):
                n_test = test_mask.sum()
                sub_mask = np.ones(n_test, dtype=np.bool)
                sub_mask[:n_test/2] = False
                new_test_mask = test_mask.copy()
                new_test_mask[test_mask] = sub_mask
                train_mask[test_mask] = np.logical_not(sub_mask)
                yield train_mask, new_test_mask
                new_test_mask[test_mask] = np.logical_not(sub_mask)
                train_mask[test_mask] = sub_mask
                yield train_mask, new_test_mask



################################################################################
# class CovEstimatorL21CV
################################################################################

def l21_cov_path(X, labels, lambdas, verbose=False, fit_params=dict()):
    precs  = None # init prec
    models = []
    n_samples = X.shape[0]

    group_idx   = np.unique(labels)
    n_groups    = len(group_idx)
    dim         = X.shape[1]
    shape       = (n_groups, dim, dim)
    emp_covs    = np.empty(shape)
    for group_id, emp_cov in zip(group_idx, emp_covs):
        this_X = X[labels == group_id]
        n_samples     = len(this_X)
        emp_cov[...] = np.dot(this_X.T, this_X)/n_samples

    for l21 in lambdas:
        model = CovEstimatorL21(l21=l21)
        model.fit(X, labels, precs=precs, 
                    **fit_params)
        # Heuristic to find a good starting point
        for prec, emp_cov in zip(model.precisions, emp_covs):
            prec[...] += .5*regularized_inverse(emp_cov)
        models.append(model)

        if verbose:
            print '%s, dual gap: %s' % (model, 
                                model.dual_gap(emp_cov, with_obj=True)[0])
    return models


class CovEstimatorL21CV(BaseCovEstimatorCV):
    def fit(self, X, labels, cv=None, **fit_params):
        X = np.asanyarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.lambdas is None:
            lambdas = np.logspace(0, -4, self.n_lambdas)
        else:
            lambdas = np.sort(self.lambdas)[::-1]

        n_lambdas = len(lambdas)
        # init cross-validation generator
        cv = cv if cv else LeaveHalfALabelOut(labels)

        scores = np.zeros(n_lambdas)
        for train, test in cv:
            models_train = self.path(X[train], labels[train],
                                    lambdas=lambdas, fit_params=fit_params)
            for i_lambda, model in enumerate(models_train):
                scores[i_lambda] += model.score(X[test], labels[test])

        i_best_lambda = np.argmax(scores)
        best_lambda   = lambdas[i_best_lambda]
        model = self.path(X, labels, lambdas=[best_lambda,],
                                    fit_params=fit_params)[0]
        self.precisions = model.precisions
        param_name, param_value = model._get_params().items()[0]
        setattr(self, param_name, param_value)
        self.score = model.score 
        self.best_model = model
        return self


    path = staticmethod(l21_cov_path)



################################################################################
# class DualCovEstimatorL1
################################################################################

class LineSearchFailure(Exception):
    pass

class DualCovEstimatorL1(CovEstimatorL1):
    """ Covariance estimator with an L1 regularisation using dual optim.

        Attributes
        ----------
        l1, float
            The L1 penalty used to obtain sparse precision estimate
    """

    def get_log_det_prec(self):
        return logdet(self.precision)

    def fit(self, emp_cov, maxiter=100, tol=1e-4, verbose=True, use_dgap=True,
            **kwargs):
        """ Fit the model to the given covariance.

        Parameters
        ----------
        emp_cov: array of shape (p, p)
            The sample covariance of the obervations
        maxiter: int, optional
            Maximum number iterations
        miniter: int, optional
            Minimum number of iterations
        eps: float, optional
            Machine precision limit, to regularise inverses
        gap_threshold, float, optional
            The value of the dual gap to reach to declare convergence.
        verbose: bool, optional
            Verbosity mode
        """

        l1 = self.l1
        if l1<0:
            raise ValueError, 'l1 must be non-negative'
        if emp_cov.shape[0] != emp_cov.shape[1]:
            raise ValueError, 'emp_cov should be a square matrix of shape (dim,dim)'

        def inf_norm_projection(W):
            W /= np.maximum(np.abs(W)/l1,1)
            diag_stride = W.shape[0] + 1
            W.flat[::diag_stride] = 0.0
            # Make W symmetric
            W =  W + W.T
            W *= .5
            return W

        def dual_obj_gradient(W):
            G = linalg.inv(emp_cov + W)
            diag_stride = G.shape[0] + 1
            G.flat[::diag_stride] = 0
            mask = (  ((np.abs(W-l1) < 1e-8*l1) & (G>0))
                    | ((np.abs(W+l1) < 1e-8*l1) & (G<0)) )
            mask = mask & mask.T
            G[mask] = 0
            if np.max(np.abs(G - G.T)) > 1e-12:
                raise ValueError, "Non symmetric matrix"
            # Make the gradient really symmetric
            G = G + G.T
            G *= .5
            return G

        def barrier(W):
            if np.max(np.abs(W)) > l1*(1+1e-10) or np.max(np.abs(np.diag(W)))>1e-10:
                return - np.inf
            else:
                return 0

        tol *= np.sum(emp_cov**2)

        def line_search(precision, G, W):
            f0 = self.dual_obj(emp_cov, W)
            prod = np.dot(precision, G)
            t = np.trace(prod) / (prod**2).sum()
            for _ in range(100): # 100 max iter in line search
                W_tG = inf_norm_projection(W + t*G)
                if self.dual_obj(emp_cov, W_tG) + barrier(W_tG) >= f0:
                    break
                t *= .5
            else:
                raise LineSearchFailure
            return t

        W = np.zeros_like(emp_cov)
        self.precision = precision = linalg.inv(emp_cov + W)
        for i in range(maxiter):
            G = dual_obj_gradient(W)
            try:
                t = line_search(precision, G, W)
            except LineSearchFailure:
                print 'Line search failed'
                print 'Dual gap : %s' % self.dual_gap(emp_cov, W)[0]
                break
            W = inf_norm_projection(W + t*G)
            self.precision = precision = linalg.inv(emp_cov+W)
            if use_dgap and ((i+1) % 10)==0 and self.dual_gap(emp_cov, W) < tol:
                if verbose:
                    print "Convergence reached after %d iterations" % i
                break
        else:
            if verbose and use_dgap:
                print "Convergence NOT reached"

        self.precision = precision

        if verbose:
            gap, pobj, dobj = self.dual_gap(emp_cov, W, with_obj=True)
            print "Final objective   : %s" % pobj
            print "Final duality gap : %s (tol = %s)" % (gap, tol)

        return self

################################################################################
# class DualCovEstimatorL21
################################################################################

class DualCovEstimatorL21(CovEstimatorL21):

    def get_log_det_prec(self):
        return [logdet(precision) for precision in self.precisions]

    def fit(self, emp_covs, maxiter=100, tol=1e-4,
                                    verbose=True, use_dgap=True, **kwargs):
        """ Fit the model to the given covariances.

        Parameters
        ----------
        emp_covs: array of shape (n, p, p)
            The sample covariances of the each of the obervation groups
        maxiter: int, optional
            Maximum number iterations
        miniter: int, optional
            Minimum number of iterations
        eps: float, optional
            Machine precision limit, to regularise inverses
        gap_threshold, float, optional
            The value of the dual gap to reach to declare convergence.
        verbose: bool, optional
            Verbosity mode
        """

        l21 = self.l21
        if l21 < 0:
            raise ValueError, 'l21 must be non-negative'
        if emp_covs.shape[1] != emp_covs.shape[2]:
            raise ValueError, 'emp_covs should be square matrices of shape (dim,dim)'

        def inf_norm_projection(Ws):
            scaling = np.maximum(np.sqrt(np.sum(Ws**2, axis=0)) / l21, 1.0)
            Ws = Ws / scaling[np.newaxis, ...]
            diag_stride = Ws.shape[2] + 1
            for W in Ws:
                W.flat[::diag_stride] = 0.0
                # Make W symmetric
                W[:] = W + W.T
                W *= .5
            return Ws

        def dual_obj_gradient(Ws):
            Gs = np.empty_like(Ws)
            diag_stride = Gs.shape[2] + 1
            for emp_cov, G, W in zip(emp_covs, Gs, Ws):
                G[:] = sym_inv(emp_cov + W)
                G.flat[::diag_stride] = 0.0
            norms = np.sqrt(np.sum(Ws**2, axis=0))
            mask = (np.abs(norms-l21) < 1e-8*l21) & (np.sqrt(np.sum(Gs**2, axis=0)) > 1e-12)
            mask = mask & mask.T
            for G in Gs:
                G[mask] = 0.0
                # Make the gradient symmetric
                G[:] = 0.5 * (G + G.T)
            return Gs

        def barrier(Ws):
            norms = np.sqrt(np.sum(Ws**2, axis=0))
            if np.max(norms) > l21*(1+1e-12) or np.max(np.diag(norms)) > 1e-12:
                return - np.inf
            else:
                return 0

        tol *= np.sum(emp_covs**2)

        def line_search(precisions, Gs, Ws):
            f0 = self.dual_obj(emp_covs, Ws)
            t = 0
            for precision, G in zip(precisions, Gs):
                prod = np.dot(precision, G)
                t += np.trace(prod) / (prod**2).sum()
            for _ in range(100): # 100 max iter in line search
                Ws_tG = inf_norm_projection(Ws + t*G)
                if self.dual_obj(emp_covs, Ws_tG) + barrier(Ws_tG) >= f0:
                    break
                t *= .5
            else:
                raise LineSearchFailure
            return t

        def compute_precisions(Ws):
            precisions = np.empty_like(Ws)
            for precision, emp_cov, W in zip(precisions, emp_covs, Ws):
                precision[:] = linalg.inv(emp_cov+W)
            return precisions

        Ws = np.zeros_like(emp_covs)
        self.precisions = precisions = compute_precisions(Ws)
        for i in range(maxiter):
            G = dual_obj_gradient(Ws)
            try:
                t = line_search(precisions, G, Ws)
            except LineSearchFailure:
                print 'Line search failed'
                print 'Dual gap : %s' % self.dual_gap(emp_covs, Ws)
                break
            Ws = inf_norm_projection(Ws + t*G)
            self.precisions = precisions = compute_precisions(Ws)
            if use_dgap and ((i+1) % 10)==0 and self.dual_gap(emp_covs, Ws) < tol:
                if verbose:
                    print "Convergence reached after %d iterations" % i
                break
        else:
            if verbose and use_dgap:
                print "Convergence NOT reached"

        self.precisions = compute_precisions(Ws)

        if verbose:
            gap, pobj, dobj = self.dual_gap(emp_covs, Ws, with_obj=True)
            print "Final objective   : %s" % pobj
            print "Final duality gap : %s (tol = %s)" % (gap, tol)

        return self
