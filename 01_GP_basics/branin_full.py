"""
Test stochastic process model on Branin function
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmocean
import scipy.stats

import spm

# Define function to emulate
a = 1
b = 5.1/(4*np.pi**2)
c = 5/np.pi
r = 6
s = 10
t = 1/(8*np.pi)

def func(x1, x2):
    """
    Branin function: see https://www.sfu.ca/~ssurjano/branin.html
    """
    f1 = a*(x2 - b*x1**2 + c*x1 - r)**2
    f2 = s*(1-t)*np.cos(x1) + s
    return f1 + f2

# Coordinates: Optimized Latin Hypercube

# Coordinates: Optimized Latin Hypercube
x = np.array([[0.8000, 0.7500],
                [0.9000,0.2000],
                [0.4000,0.6500],
                [0.3000, 0.2500],
                [1.0000, 0.6000],
                [0, 0.7000],
                [0.6000, 0.8500],
                [0.1000, 0.9500],
                [0.9500, 0.9000],
                [0.3500, 0],
                [0.2500, 0.8000],
                [0.6500, 0.5500],
                [0.1500, 0.1000],
                [0.5000, 0.4000],
                [0.0500, 0.3500],
                [0.2000, 0.5000],
                [0.7000, 0.3000],
                [0.7500, 0.0500],
                [0.4500, 1.0000],
                [0.5500, 0.1500],
                [0.8500, 0.4500]])

x[:, 0] = 15*x[:, 0] - 5
x[:, 1] = 15*x[:, 1]

# Plot function and sample points
fig, (ax, axp) = plt.subplots(ncols=2, figsize=(9, 4))
y = func(x[:, 0], x[:, 1])
y = np.vstack(y)

xf = 15*np.linspace(0, 1, 51) - 5
yf = 15*np.linspace(0, 1, 51)

[xxf, yyf] = np.meshgrid(xf, yf)
levels = np.arange(0, 350, 50)
levels[0] = 3
vals = np.zeros(xxf.shape)
n = vals.shape[0]
for ii in range(n):
    for jj in range(n):
        vals[ii,jj] = func(xxf[ii,jj], yyf[ii,jj])

qcs = ax.contour(xxf, yyf, vals, levels=levels)
ax.plot(x[:, 0], x[:, 1], 'bx')
ax.set_title('Analytic')

# Optimization: initial guess
theta0 = [0.003, 0.003, 2, 2]

SM = spm.solve_full_stochastic_model(x, y, x0=theta0, tol=1e-12)

# Plot prediction
y_predict = np.zeros(xxf.shape)
e_predict = np.zeros(xxf.shape)
n = y_predict.shape[0]

R = SM['R']

Rinv = np.linalg.inv(R)
for ii in range(n):
    for jj in range(n):
        xij = [xxf[ii,jj], yyf[ii, jj]]
        yij, eij = spm.predictor(xij, x, y, SM['theta'], SM['p'],
            SM['mu'], SM['sigma2'], Rinv)

        y_predict[ii, jj] = yij
        e_predict[ii, jj] = eij

axp.contour(xxf, yyf, y_predict, levels=levels)
axp.set_xlabel('x')
axp.set_ylabel('y')
axp.set_title('Predictor')

fig.savefig('01_branin_predictor_full.png', dpi=600)

# # Cross-validation
# n = x.shape[0]
# Y_cv = np.zeros(y.shape)
# S_cv = np.zeros(y.shape)
# for k in range(n):
#     xm = np.zeros((n-1, 2))
#     xm[:k] = x[:k]
#     xm[k:] = x[k+1:]
#
#     ym = np.zeros((n-1, 1))
#     ym[:k] = y[:k]
#     ym[k:] = y[k+1:]
#
#     # Use existing parameter estimates for theta and p with the smaller
#     # xm, ym to create a cross-validation correlation matrix and
#     # predictor
#     R_cv = spm.R(xm, SM['theta'], SM['p'])
#     R_cv_inv = np.linalg.inv(R_cv)
#
#     predict_result = spm.predictor(x[k], xm, ym, SM['theta'], SM['p'],
#                         SM['mu'], SM['sigma2'], R_cv_inv)
#
#     Y_cv[k] = predict_result[0]
#     S_cv[k] = np.sqrt(predict_result[1])
#
# cvfig, (cvax1, cvax2) = plt.subplots(ncols=2, figsize=(8, 4))
# cvax1.scatter(Y_cv, y)
# cvax1.plot(y, y, 'k')
# cvax1.set_xlabel('Predicted y')
# cvax1.set_ylabel('y')
# cvax1.grid(True)
#
# cvax2.scatter(y, (Y_cv - y)/S_cv)
# cvax2.set_xlabel('y')
# cvax2.set_ylabel('Standard error')
# cvax2.axhline(-3, linestyle='--', color='k')
# cvax2.axhline(3, linestyle='--', color='k')
# cvax2.grid(True)
#
# cvfig.suptitle('Cross validation analytics')
#
# cvfig.savefig('branin_crossvalidation.png', dpi=600)
