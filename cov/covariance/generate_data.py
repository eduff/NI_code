"""
Helper functions to generate synthetic data for sparse covariance
learning.
"""
# Author: Gael Varoquaux
# Copyright: INRIA

import numpy as np
from scipy import linalg

def generate_sparse_spd_matrix(dim=1, alpha=0.95, prng=np.random):
    """
    generate a sparse symetric definite positive matrix with the given dimension

    Returns
    -------
    prec: array of shape(dim,dim)
    """
    chol = -np.eye(dim)
    aux = prng.rand(dim, dim)
    aux[aux<alpha] = 0
    aux[aux>alpha] = .9*prng.rand(np.sum(aux>alpha))
    aux = np.tril(aux, k=-1)
    permutation = prng.permutation(dim)
    aux = aux[permutation].T[permutation]
    chol += aux
    # Permute the lines: we don't want to have assymetries in the final 
    # SPD matrix
    prec = np.dot(chol.T, chol)
    return prec


def generate_standard_sparse_mvn(n_samples, dim=1, alpha=.95,
                                 prng=np.random):
    """ Generate a multivariate normal samples with sparse precision, null 
        mean and covariance diagonal equal to ones (dim)

        Returns
        -------
        x, array of shape (n_samples,dim): 
            The samples
        prec, array of shape (dim,dim): 
            The theoretical precision
    """
    prec = generate_sparse_spd_matrix(dim, alpha=alpha, prng=prng)
    # Inversion for SPD matrices
    vals, vecs = linalg.eigh(prec)
    cov = np.dot(vecs/vals, vecs.T)


    # normalize covariance (and precision)
    # in order to have the diagonal=1
    idelta = np.diag(np.sqrt(np.diag(cov)))
    delta = np.diag(1./np.sqrt(np.diag(cov)))
    cov = np.dot(np.dot(delta,cov), delta)
    prec = np.dot(np.dot(idelta, prec), idelta) 

    # generate the samples
    x = prng.multivariate_normal(mean=np.zeros(dim), cov=cov,
                            size=(n_samples,))
    return x, prec

