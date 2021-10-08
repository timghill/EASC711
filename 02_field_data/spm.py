"""
Core functions for stochastic process model.

TODO: Consider making a class to more easily pass variables among
functions. Or, code flat for readability, adaptability, and simplicity :)
"""

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

"""
--------------------------------------------------------------------------
 Helper functions
--------------------------------------------------------------------------
"""

def dist(x1, x2, theta, p):
    """Calculate weighted distance between points x1 and x2.
    """
    if hasattr(x1, '__iter__'):
        # If x is a vector, sum over components
        return sum(theta*np.abs(x1 - x2)**p)
    else:
        # If x is a scalar, return one component
        return theta*np.abs(x1-x2)**p

def corr(x1, x2, theta, p):
    """Calculate correlation between errors E(x1) and E(x2).
    """
    return np.exp(-dist(x1, x2, theta, p))

def R(x, theta, p):
    """Calculate correlation matrix R for samples in x.

    R is defined as R[i, j] = corr(x[i], x[j])
    """
    n = x.shape[0]
    Rmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Rmat[i, j] = corr(x[i], x[j], theta, p)
    return Rmat

def mu_estimate(y, Rinv):
    """Calculate best estimate of mean mu
    """
    n = Rinv.shape[0]
    onevec = np.ones((n, 1))
    return np.matmul(onevec.T, np.matmul(Rinv, y))/\
            np.matmul(onevec.T, np.matmul(Rinv, onevec))

def sigma2_estimate(y, mu, Rinv):
    """Calculate best estimate of variance sigma**2
    """
    n = Rinv.shape[0]
    onevec = np.ones((n, 1))
    return np.matmul((y - onevec*mu).T, np.matmul(Rinv, (y - onevec*mu)))/n

"""
--------------------------------------------------------------------------
 Core functions
--------------------------------------------------------------------------
"""
def likelihood(x, y, theta, p):
    """Calculate likelihood of y given x, theta, and p.

    This function works with scalar or vector inputs:
     * If x is shape (n, 1), y must be shape (n, 1), and theta and p
       must be scalars
     * If x is shape (n, k), y must be shape (n, 1), and theta and p
       must be shape (k, 1) or (k,)
    """
    [n, k] = x.shape

    Rmat = R(x, theta, p)
    Rinv = np.linalg.inv(Rmat)

    onevec = np.ones((n, 1))
    mu = mu_estimate(y, Rinv)
    muvec = onevec*mu

    sigma2 = sigma2_estimate(y, mu, Rinv)

    pref = (2*np.pi)**(n/2)*(sigma2)**(n/2)*np.abs(np.linalg.det(Rmat))**(0.5)
    arg = np.matmul((y - muvec).T, np.matmul(Rinv, (y - muvec)))/(2*sigma2)
    likeli = float((1/pref)*np.exp(-arg))

    return likeli

def conc_ln_likelihood(x, y, theta, p):
    [n, k] = x.shape

    Rmat = R(x, theta, p)
    Rmat = 0.5*(Rmat + Rmat.T)

    # if np.linalg.matrix_rank(Rmat)<n:
    #     return 1e2
    # else:
    Rinv = np.linalg.inv(Rmat)

    # onevec = np.ones((n, 1))
    mu = mu_estimate(y, Rinv)
    # muvec = onevec*mu

    sigma2 = sigma2_estimate(y, mu, Rinv)

    l = float(0.5*n*np.log(sigma2) + 0.5*np.log(np.linalg.det(Rmat)))
    if np.isnan(l) or np.isinf(l):
        L = 0
    else:
        L = l
    return L

    # pref = (2*np.pi)**(n/2)*(sigma2)**(n/2)*np.abs(np.linalg.det(Rmat))**(0.5)
    # arg = np.matmul((y - muvec).T, np.matmul(Rinv, (y - muvec)))/(2*sigma2)
    # likeli = float((1/pref)*np.exp(-arg))

# Prediction - the whole point of this method
def predictor(xs, x, y, theta, p, mu, sigma2, Rinv):
    """Calculated predicted y value and standard error.

    Dimensions work the same as in function likelihood, where xs is shape
    (1, k)
    """
    n = Rinv.shape[0]
    r = np.zeros((n, 1))
    for i in range(n):
        r[i] = corr(xs, x[i], theta, p)
    onevec = np.ones((n, 1))
    muvec = mu*onevec

    ys = mu + float(np.matmul(r.T, np.matmul(Rinv, y - muvec)))

    # Standard error
    t1 = 1
    t2 = np.matmul(r.T, np.matmul(Rinv, r))
    t3 = (1 - float(np.matmul(onevec.T, np.matmul(Rinv, r))))**2/\
                float((np.matmul(onevec.T, np.matmul(Rinv, onevec))))

    s2 = sigma2*(t1 - t2 + t3)

    return (float(ys), float(s2))

def solve_stochastic_model(x, y, p=2, x0=10, **kwargs):
    """Solve stochastic model given data (x, y).

    Uses scipy.optimize.minimize to minimize the concentrated
    likelihood function to optimize parameters theta for a FIXED p

    Uses estimated theta and p to calculate the mean and variance, as
    well as the correlation matrix R. This function does not calculate the
    inverse of the correlation matrix.

    Any additional kwargs are passed to scipy.optimize.minimize

    Returns: Dictionary with keys 'mu', 'sigma2', 'theta', p', and 'R'
    corresponding to the obvious values.
    """
    # Maximize likelihood function
    # p = 2 # Enforce this for now
    # objective = lambda theta: -np.log(np.abs(likelihood(x, y, theta, p)))
    k = x.shape[1]
    # bnds = [(0, None) for m in range(k)]
    # cons = ({'type': 'ineq', 'fun':lambda x: x})
    objective = lambda theta: conc_ln_likelihood(x, y, theta, p)
    # objective = lambda theta: -n/2*np.log(
    minimize_res = scipy.optimize.minimize(objective, x0,**kwargs)
    print(minimize_res)
    theta_hat = minimize_res.x

    # Estimate parameters
    R_hat = R(x, theta_hat, p)
    Rinv = np.linalg.inv(R_hat)

    mu_hat = mu_estimate(y, Rinv)
    sigma2_hat = sigma2_estimate(y, mu_hat, Rinv)

    stoch_model = { 'mu': float(mu_hat),
                    'sigma2': float(sigma2_hat),
                    'theta': theta_hat,
                    'p': p,
                    'R':R_hat,
                    }

    return stoch_model

def solve_full_stochastic_model(x, y, x0=[1, 1], **kwargs):
    """Solve stochastic model given data (x, y).

    Uses scipy.optimize.minimize to minimize the concentrated
    likelihood function to optimize parameters theta AND p. Unlike
    solve_stochastic_model, this function does not assume p is known

    Uses estimated theta and p to calculate the mean and variance, as
    well as the correlation matrix R. This function does not calculate the
    inverse of the correlation matrix.

    Any additional kwargs are passed to scipy.optimize.minimize

    Returns: Dictionary with keys 'mu', 'sigma2', 'theta', p', and 'R'
    corresponding to the obvious values.
    """
    # Maximize likelihood function
    k = x.shape[1]
    # objective = lambda params: -np.log(np.abs(likelihood(x, y, params[:k], params[k:])))

    objective = lambda params: conc_ln_likelihood(x, y, params[:k], params[k:])
    cons = ({'type': 'ineq', 'fun':lambda x: x[1]})

    bnds_theta = [(1e-8, None) for m in range(k)]
    bnds_p = [(1, 2) for m in range(k)]

    bnds = []
    bnds.extend(bnds_theta)
    bnds.extend(bnds_p)

    minimize_res = scipy.optimize.minimize(objective, x0, method='SLSQP',
                    bounds=bnds, constraints=cons,**kwargs)
    print(minimize_res)
    theta_hat = minimize_res.x[:k]
    p = minimize_res.x[k:]


    # Estimate parameters
    R_hat = R(x, theta_hat, p)
    Rinv = np.linalg.inv(R_hat)

    mu_hat = mu_estimate(y, Rinv)
    sigma2_hat = sigma2_estimate(y, mu_hat, Rinv)

    stoch_model = { 'mu': mu_hat,
                    'sigma2': sigma2_hat,
                    'theta': theta_hat,
                    'p': p,
                    'R':R_hat,
                    }

    return stoch_model
