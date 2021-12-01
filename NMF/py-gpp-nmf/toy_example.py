"""
Recreate 'toy example' from Schmidt et al. (2008)
Non-negative matrix factorization with Gaussian Process Priors
"""

import numpy as np
from matplotlib import pyplot as plt
import cmocean

from scipy.optimize import minimize

import link
import nmf
# import gppnmf

# PARAMETERS
m = 100             # Number of rows of data matrix
n = 200             # Number of columns of data matrix
L = 2               # Rank of matrix factors

mu = 0              # Mean of generated data
sigma = 1          # Std of generated data
sigmaN = 5          # Std of noise
lambd = 1           # H link function parameter
s = 1               # D link function parameter

eps = 1e-12         # Nugget-type term to avoid zero division

betaD = 0.01        # Gaussian radial basis function parameter for toy data
betaH = 0.01        # Gaussian radial basis function parameter for toy data

indices = np.arange(m)
[ii, jj] = np.meshgrid(indices, indices)

# Generate D covariance matrix
covD = np.exp(-betaD*(ii - jj)**2) + eps*np.eye(m)
cD = np.linalg.cholesky(covD)

indices = np.arange(n)
[ii, jj] = np.meshgrid(indices, indices)
# Generate H covariance matrix
covH = np.exp(-betaH*(ii-jj)**2) + eps*np.eye(n)
cH = np.linalg.cholesky(covH)

## Generate random data
np.random.seed(50)
D_toy = link.rect_gauss_inv(np.matmul(np.random.normal(size=(L,m)), cD.T), mu, sigma, lambd)

H_toy = link.exponential_inv(np.matmul(np.random.normal(size=(L,n)), cH.T), mu, sigma, lambd)

Y = np.matmul(D_toy.T, H_toy)
Y_toy = Y + sigmaN*np.random.normal(size=(m,n))

## SIMPLE LEAST SQUARES NMF
classical_nmf = nmf.NMF(Y_toy, L)
w, h, err = classical_nmf.solve(rtol=1e-2)
Y_LS = np.matmul(w, h)

## GPP NMF
X = Y_toy

# Initial guesses: random
n_delta = L*m
n_eta = L*n

delta = np.random.normal(size=n_delta)
eta = np.random.normal(size=n_eta)

D_link = lambda delta: link.rect_gauss_inv(np.matmul(delta.reshape((L, m)), cD.T), mu, sigma, lambd)
H_link = lambda eta: link.exponential_inv(np.matmul(eta.reshape((L,n)), cH.T), mu, sigma, lambd)

D = D_link(delta)
H = H_link(eta)

Y_estimate = np.matmul(D.T, H)
print(np.sum((Y_estimate - X)**2))
for i in range(3):
    print(i)
    # Optimize for delta, treating eta as fixed
    print('Optimizing over delta:')
    def objective(delta):
        D_star = D_link(delta)
        Z = X - np.matmul(D_star.T, H)
        return 0.5*(np.sum(Z**2)/sigmaN**2 + np.sum(delta**2))

    delta_new = minimize(objective, delta).x
    D_new = D_link(delta_new)
    # Optimize for eta, treating delta as fixed
    print('Optimizing over eta')
    def objective_eta(eta):
        H_eta = H_link(eta)
        Z = X - np.matmul(D_new.T, H_eta)
        return 0.5*(np.sum(Z**2)/sigmaN**2 + np.sum(eta**2))

    eta_new = minimize(objective_eta, eta).x
    H_new = H_link(eta_new)

    Y_new = np.matmul(D_new.T, H_new)

    err = np.linalg.norm(Y_new - Y_estimate)
    print(err)

    print(np.sum( (Y_new - X)**2))

    Y_estimate = Y_new
    eta = eta_new
    delta = delta_new

fargs = {'cmap':'cmo.matter', 'vmin':0, 'vmax':20}
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
im = ax[0][0].imshow(Y_toy, **fargs)
ax[0][0].set_title('Data matrix')

ax[0][1].imshow(Y, **fargs)
ax[0][1].set_title('Underlying data')

ax[1][0].imshow(Y_LS, **fargs)
ax[1][0].set_title('Simple NMF')

ax[1][1].imshow(Y_estimate, **fargs)
ax[1][1].set_title('GPP NMF')

plt.tight_layout()
fig.savefig('gpp_nmf.png', dpi=600)
# fig.colorbar(im)

plt.show()
