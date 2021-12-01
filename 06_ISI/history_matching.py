"""
Explore History Matching from McNeall et al. (2013)

NOTE: For now, explicitly use model directly as "fast"
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import cmocean

import pyDOE

from ice_sheet_model import model
import spm

emulators = [0, 0, 0]
measure_sigma = 0.05

np.random.seed(50)
N = 30
d = 3
X_train = pyDOE.lhs(d, N, 'cm', 15)

Y_train = np.array(model(X_train[:, 0], X_train[:, 1], X_train[:, 2])).T
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

    SM = spm.solve_stochastic_model(X_train, y_train, x0=x0)
    emulators[ii] = SM.copy()

def emulator(X):
    N = X.shape[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    y3 = np.zeros(N)

    e1 = np.zeros(N)
    e2 = np.zeros(N)
    e3 = np.zeros(N)
    for i in range(X.shape[0]):
        em1 = spm.predictor(X[i], X_train, np.vstack(Y_train[:, 0]), **emulators[0])
        y1[i] = em1[0]
        e1[i] = em1[1]

        em2 = spm.predictor(X[i], X_train, np.vstack(Y_train[:, 1]), **emulators[1])
        y2[i] = em2[0]
        e2[i] = em2[1]

        em3 = spm.predictor(X[i], X_train, np.vstack(Y_train[:, 2]), **emulators[2])
        y3[i] = em3[0]
        e3[i] = em3[1]
    y = (y1, y2, y2)
    e = (e1, e2, e3)
    return (y, e)


# Design ensemble for simulator
# X = pyDOE.lhs(3, 50, 'cm', 50)
x_star = np.array([0.5, 0.5, 0.5])

# Measurement at the ensemble points
z_thick, z_area, z_vol = model(*x_star)

def implausibility(x):
    y_em, e_em = emulator(x)
    y_thick, y_area, y_vol = y_em
    e_thick, e_area, e_vol = e_em

    # Total variance is sum of emulator variance and measurement std dev
    var_thick = e_thick + measure_sigma**2
    var_area = e_area + measure_sigma**2
    var_vol = e_vol + measure_sigma**2

    I2_thick = np.sqrt((y_thick - z_thick)**2/var_thick)
    I2_area = np.sqrt((y_area - z_area)**2/var_area)
    I2_vol = np.sqrt((y_vol - z_vol)**2/var_vol)
    return (I2_thick, I2_area, I2_vol)

N_samples = 250
X_experiment = pyDOE.lhs(3, N_samples, 'cm', 25)

I2_thick, I2_area, I2_vol = implausibility(X_experiment)
I2_max = np.max(np.array([I2_thick, I2_area, I2_vol]), axis=0)

fig, ax = plt.subplots(figsize=(8, 8), ncols=3, nrows=3, sharey=True, sharex=True)
args = {'cmap':cmocean.cm.thermal, 'vmin':0, 'vmax':5}

Y_outputs, E_outputs = emulator(X_experiment)
for i in range(3):
    sp = ax[0][i].scatter(X_experiment[:, i], Y_outputs[0], c=I2_max, **args)

    ax[1][i].scatter(X_experiment[:, i], Y_outputs[1], c=I2_max, **args)
    ax[2][i].scatter(X_experiment[:, i], Y_outputs[2], c=I2_max, **args)

ax[-1][0].set_xlabel('F')
ax[-1][1].set_xlabel('S')
ax[-1][2].set_xlabel('D')

ax[0][0].set_ylabel('I2 Thick')
ax[1][0].set_ylabel('I2 Area')
ax[2][0].set_ylabel('I2 Vol')


# Two-at-a-time sensitivity analysis
Nx = 50
xplot = np.linspace(0, 1, Nx)
yplot = np.linspace(0, 1, Nx)
[xxp, yyp] = np.meshgrid(xplot, yplot)

fig, ax = plt.subplots(figsize=(8, 8), ncols=2, nrows=2, sharex=True, sharey=True)
for i in range(1, 3):
    for j in range(i):
        Xij = 0.5*np.ones((Nx**2, 3))
        Xij[:, i] = yyp.flatten()
        Xij[:, j] = xxp.flatten()
        implaus = np.array(implausibility(Xij))
        # im_max = np.ma
        implaus = np.max(implaus, axis=0)
        # implaus = implaus[0]

        pc = ax[i-1][j].pcolormesh(xxp, yyp, implaus.reshape((Nx, Nx)), vmin=0, vmax=3,
            cmap=cmocean.cm.balance)

ax[0][1].spines['right'].set_visible(False)
ax[0][1].spines['left'].set_visible(False)
ax[0][1].spines['top'].set_visible(False)
ax[0][1].spines['bottom'].set_visible(False)
ax[0][1].set_visible(False)

ax[0][0].set_ylabel('S')
ax[1][0].set_ylabel('D')
ax[1][0].set_xlabel('F')
ax[1][1].set_xlabel('S')


cb = fig.colorbar(pc, ax=ax[0][1])
cb.set_label('Implausibility')

I2_max_indicator = np.ones(N_samples)
I2_max_indicator[I2_max>3] = 0
I2_max_indicator[np.isnan(I2_max)] = 0
plaus_vol = np.mean(I2_max_indicator)
print('Plausible volume: ', plaus_vol)
print(I2_max[I2_max_indicator==1])

plt.tight_layout()
fig.savefig('ism_history_matching_max.png', dpi=600)

plt.show()
