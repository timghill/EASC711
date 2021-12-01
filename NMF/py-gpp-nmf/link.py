import numpy as np
from scipy import stats
from scipy.special import erf, erfinv

def gauss(x, sigma):
    """
    x: Values to compute link function values for
    sigma: Diagonal of covariance matrix
    """
    return stats.norm.cdf(0.5 + 0.5*erf(x/np.sqrt(2)/sigma))

def exponential_inv(x, mu, sigma, lambd, eps=1e-12):
    Psi = erf(x/np.sqrt(2)/sigma)
    z =  -1/lambd * np.log(eps + 0.5 - 0.5*Psi)
    # z[z<0] = 0
    # z[np.isnan(z)] = 0
    return z

def rect_gauss_inv(x, mu, sigma, lambd, eps=0):
    z = lambd*np.sqrt(2)*erfinv(eps + 0.5 + 0.5*erf((x-mu)/np.sqrt(2)/sigma))
    # z[z<0] = 0
    # z[np.isnan(z)] = 0
    return z

def exponential_inv_deriv(x, mu, sigma, lambd, eps=1e-12):
    Psi = lambd*exponential_inv(x, mu, sigma, lambd, eps=eps) - x**2/2/sigma**2
    z = 1/np.sqrt(2*np.pi)/sigma/lambd * np.exp(Psi)
    return z

def rect_gauss_inv_deriv(x, mu, sigma, lambd, eps=1e-12):
    fh = exponential_inv_deriv(x, mu, sigma, lambd, eps=eps)
    Psi = fv**2/2/lambd**2 - x**2/2/sigma**2
    z = lambd/2/sigma * np.exp(Psi)
    return z
