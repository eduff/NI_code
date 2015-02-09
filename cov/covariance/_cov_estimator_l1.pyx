cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double f)
    double pow(double f, int k)

ctypedef np.float64_t DOUBLE

cdef double positive_root(DOUBLE a,DOUBLE b,DOUBLE c):
    """
    solve the second order equation and take the positive root
    """
    cdef DOUBLE delta = pow(b,2) - 4*a*c
    if delta < 0.0:
        raise ValueError, 'Complex values not handled'
    return (-b+sqrt(delta))/(2.0*a)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
@cython.cdivision(True)
@cython.cdivision_warnings(False)
def _choleski_update_l1(np.ndarray[DOUBLE, ndim=2, mode='c'] emp_cov,
                        np.ndarray[DOUBLE, ndim=2, mode='c'] precision,
                        np.ndarray[DOUBLE, ndim=2, mode='c'] abs_inv_prec0,
                        double l1,
                        np.ndarray[DOUBLE, ndim=2, mode='c'] chol_prec):
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

    cdef unsigned int ii, l, c
    cdef unsigned int p = emp_cov.shape[0]
    cdef double pa, pb, pd, pcc, pll
    # Update of non-diagonal parameters
    cdef np.ndarray[DOUBLE, ndim=2] old_chol_prec = chol_prec.copy()
    for l in range(p):
        # this_chol_line is a view on the precision cholesky, it is
        # modified _inplace_ to modify the original array itself
        # this_chol_line = chol_prec[l]
        for c in range(l):
            # this_precision_line = precision[c]
            # pa = np.dot(this_chol_line, emp_cov[:,c])
            pa = 0.0
            for ii in range(p):
                pa += chol_prec[l,ii]*emp_cov[ii,c]
            # pa -= this_chol_line[c]*emp_cov[c,c]
            pa -= chol_prec[l,c]*emp_cov[c,c]

            pb = 0.0
            # pb = np.dot((  this_precision_line
            #                 - this_chol_line*this_chol_line[c]),
            #         this_chol_line*abs_inv_prec0[c])
            for ii in range(p):
                pb += (precision[c,ii] - chol_prec[l,ii] * chol_prec[l,c]) * \
                        chol_prec[l,ii] * abs_inv_prec0[c,ii]
            pd = 0.0
            for ii in range(p):
                pd += pow(chol_prec[l,ii],2) * abs_inv_prec0[c,ii]
                # pd += chol_prec[l,ii]**2 * abs_inv_prec0[c,ii]
            # pd =  np.dot(this_chol_line**2, abs_inv_prec0[c])

            # Update the choleski inplace
            # this_chol_line[c] = -(pa + l1*pb)/(emp_cov[c,c] + l1*pd)
            chol_prec[l,c] = -(pa + l1*pb)/(emp_cov[c,c] + l1*pd)

            # Update the precision
            # pcc = (  this_precision_line[c]
            #         + this_chol_line[c]**2
            #         - old_chol_prec[l,c]**2 )
            pcc = precision[c,c] + pow(chol_prec[l,c],2) - pow(old_chol_prec[l,c],2)
            # pcc = precision[c,c] + chol_prec[l,c]**2 - old_chol_prec[l,c]**2
            # this_precision_line += (this_chol_line
            #                     *(this_chol_line[c] - old_chol_prec[l,c]))
            for ii in range(p):
                precision[c,ii] += chol_prec[l,ii] * \
                                   (chol_prec[l,c] - old_chol_prec[l,c])
                precision[ii,c] = precision[c,ii]
            precision[c,c] = pcc

    # Update of the diagonal terms
    for l in range(p):
        # Inplace operations for speed
        # pa = np.dot(chol_prec[l]**2, abs_inv_prec0[l])
        pa = 0.0
        for ii in range(p):
            pa += pow(chol_prec[l,ii],2) * abs_inv_prec0[l,ii]
            # pa += chol_prec[l,ii]**2 * abs_inv_prec0[l,ii]
        pa = emp_cov[l,l] + l1*pa
        
        # pd = np.dot(chol_prec[l], emp_cov[:,l])
        pd = 0.0
        for ii in range(p):
            pd += chol_prec[l,ii] * emp_cov[ii,l]
        pd -= (chol_prec[l,l]*emp_cov[l,l])
        
        # pb = np.dot((precision[l] - chol_prec[l,l]*chol_prec[l]),
        #             chol_prec[l]*abs_inv_prec0[l])
        pb = 0.0
        for ii in range(p):
            pb += (precision[l,ii] - chol_prec[l,l]*chol_prec[l,ii]) * \
                  chol_prec[l,ii]*abs_inv_prec0[l,ii]
        pb = pd + l1*pb

        # Update the choleski
        chol_prec[l,l] = positive_root(pa, pb, -1.)

        # Update the precision
        pll = precision[l,l] + pow(chol_prec[l,l],2) - pow(old_chol_prec[l,l],2)
        # pll = precision[l,l] + chol_prec[l,l]**2-old_chol_prec[l,l]**2
        # precision[l] += chol_prec[l]*(chol_prec[l,l]-old_chol_prec[l,l])
        # precision[:,l] = precision[l]
        for ii in range(p):
            precision[l,ii] += chol_prec[l,ii]*(chol_prec[l,l]-old_chol_prec[l,l])
            precision[ii,l] = precision[l,ii]
        precision[l,l] = pll

    return precision, chol_prec
