"""
Fit a GP to ice sheet model output, including cross-validation
"""

import numpy as np
from matplotlib import pyplot as plt

from ice_sheet_model import model
import spm

import pyDOE
import cmocean

np.random.seed(50)
N = 30
d = 3
X_train = pyDOE.lhs(d, N, 'cm', 15)

Y_train = np.array(model(X_train[:, 0], X_train[:, 1], X_train[:, 2])).T

fig, axs = plt.subplots(figsize=(9, 4), ncols=3)
for ii in range(3):

    y_train = Y_train[:, ii]

    y_train = np.vstack(y_train)

    theta = 5*pyDOE.lhs(3, 100, 'cm', 15)
    p = 2
    likeli = np.zeros(100)
    for i in range(100):
        likeli[i] = spm.conc_ln_likelihood(X_train, y_train, theta[i], p)
    im = np.argmax(likeli)
    x0 = theta[im]

    SM = spm.solve_stochastic_model(X_train, y_train,
            x0=x0)

    norm_cv_err = np.zeros(N)
    y_predict_cv = np.zeros(N)
    for k in range(N):
        y_cv_exact = y_train[k]
        x_cv = X_train[k]

        x_loo = np.zeros((N-1, d))
        y_loo = np.vstack(np.zeros(N-1))

        x_loo[:k] = X_train[:k]
        x_loo[k:] = X_train[k+1:]

        y_loo[:k] = y_train[:k]
        y_loo[k:] = y_train[k+1:]

        SM_cv = SM.copy()
        SM_cv['R'] = spm.R(x_loo, SM['theta'], SM['p'])
        SM_cv['Rinv'] = np.linalg.inv(SM_cv['R'])

        y_predict, e_predict = spm.predictor(x_cv, x_loo, y_loo, **SM_cv)
        norm_cv_err[k] = (y_predict - y_cv_exact)/np.sqrt(e_predict)

        y_predict_cv[k]= y_predict

    # ax.scatter(y_predict_cv, y_train)
    axs[ii].scatter(y_train, norm_cv_err)
    axs[ii].grid()
    axs[ii].set_ylim([-3.5, 3.5])
axs[1].set_xlabel('y(x)')
axs[0].set_ylabel('Normalized error')
fig.savefig('ism_gp_cv.png', dpi=600)

# New design
Nx = 30
xp = np.linspace(0, 1, Nx)
[xxp, yyp] = np.meshgrid(xp, xp)
XX = np.zeros((Nx**2, 3))
XX[:, 0] = xxp.flatten()
XX[:, 1] = yyp.flatten()
XX[:, 2] = 0.5

y_emulator = np.zeros((Nx, Nx))
y_model = model(XX[:, 0], XX[:, 1], XX[:, 2])[2]
y_model = y_model.reshape((Nx, Nx))
for i in range(Nx):
    for j in range(Nx):
        y_emulator [i,j]= spm.predictor([xxp[i, j], yyp[i, j], 0.5], X_train, np.vstack(Y_train[:, 2]), **SM)[0]

fig, ax = plt.subplots(ncols=2, nrows=2)
ax[0][0].pcolormesh(xxp, yyp, y_model, vmin=0.75, vmax=1.25)
ax[0][1].pcolormesh(xxp, yyp, y_emulator, vmin=0.75, vmax=1.25)

ax[1][0].pcolormesh(xxp, yyp, y_emulator - y_model, vmin=-0.01, vmax=0.01, cmap=cmocean.cm.balance)
plt.show()
