# Authors: Gael Varoquaux, Alexandre Gramfort
# Copyrigth: INRIA

import numpy as np
from scipy import linalg
from math import exp, log

def fast_logdet(A):
    """
    Compute log(det(A)) for A symmetric
    Equivalent to : np.log(linalg.det(A))
    but more robust
    It returns -Inf if det(A) is non positive

    XXX: The limitation with this function is that it does not always
    find that a matrix is non positive definite.
    """
    ld = np.sum(np.log(np.diag(A)))
    if not np.isfinite(ld):
        return -np.inf
    a = exp(ld/A.shape[0])
    d = linalg.det(A/a)
    if d <= 0:
        return -np.inf
    ld += log(d)
    if not np.isfinite(ld):
        return -np.inf
    return ld


def logdet(A):
    """
    Compute log(det(A)) for A symmetric
    Equivalent to : np.log(linalg.det(A))
    but more robust
    It returns -Inf if A is non positive
    """
    try:
        lambdas = linalg.cholesky(A)
    except linalg.LinAlgError:
        # Matrix non positive definite
        return -np.inf
    return fast_logdet(A)


def exact_logdet(A):
    """
    Compute log(det(A)) for A symmetric
    Equivalent to : np.log(linalg.det(A))
    but more robust
    It returns -Inf if A is non positive

    Very accurate, but slow.
    """
    lambdas = linalg.eigvalsh(A)
    #lambdas = linalg.eigvals(A).real
    if np.any(lambdas <= 0):
        return -np.inf
    return np.sum(np.log(lambdas))


def sym_inv(mat):
    """ Matrix inversion for symetric matrices, slower, but more robust.
    """
    u, s, _ = linalg.svd(mat)
    return np.dot(u/s, u.T)
