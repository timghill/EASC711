"""
Test PCA and PLS methods. This script uses the same setup as the
high dimensional case
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import cmocean
import pyDOE


import sklearn
import sklearn.cross_decomposition
from sklearn.metrics import mean_squared_error as mse

# import spm

# Define coordinates
N = 20  # Grid size
m = 30 # Number of experiments

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)

qwidth = int(N/4)
qhalf = int(N/2)

def phys_model(alpha, beta):
    """Calculate artifical data for high-dimensional GP emulation
    """
    Psqrt = (1-alpha) + alpha*np.sqrt(xx)

    pert_channel = np.zeros(xx.shape)
    # Pchannel[qhalf, :qwidth] = - alpha

    if beta>0:
        pert_channel[qhalf-1, :qwidth] = 1
        for i in range(1, qhalf):
            pert_channel[qhalf  - 1 + i, qwidth + i -1] = 1
            pert_channel[qhalf - i - 1, qwidth + i -1] = 1

        pert_channel = scipy.ndimage.gaussian_filter(pert_channel, 2)

        pert_max = np.max(pert_channel.flatten())
        pert_min = np.min(pert_channel.flatten())

        # Center so min is zero
        pert_channel = pert_channel - pert_min

        # Scale so max is alpha/4
        pert_channel = pert_channel * (beta/4)/pert_max

    P = Psqrt - pert_channel
    P[P<0] = 0
    return P

# sim_fig, (m1_ax, m2_ax) = plt.subplots(figsize=(8, 4), ncols=2)
# m1_ax.pcolormesh(xx, yy, phys_model(1, 0), vmin=0, vmax=1)
# m2_ax.pcolormesh(xx, yy, phys_model(0, 1), vmin=0, vmax=1)
# sim_fig.savefig('simulator_modes.png', dpi=600)

# Experimental design
np.random.seed(12)
X = pyDOE.lhs(2, samples=m, criterion='cm', iterations=250)
#
fig, ax = plt.subplots()
pc = ax.pcolormesh(xx, yy, phys_model(1, 1), vmin=-0, vmax=1)
# ax.contour(xx, yy, phys_model(1, 1), colors='k', linewidths=0.5)
fig.colorbar(pc)
# ax.set_title('Simulator')
# fig.savefig('simulator.png', dpi=600)

# fig2, ax2 = plt.subplots()
# ax2.plot(X[:,0], X[:,1], 'ro')
# ax2.set_xlabel('$\\beta$')
# ax2.set_ylabel('$\\alpha$')
# ax2.set_title('Experimental design')

# Experimental data
Eta = np.zeros((int(N*N), m))
for i in range(m):
    alpha, beta = X[i]
    Pi = phys_model(alpha, beta)
    Eta[:, i] = Pi.flatten()
# Eta = Eta - np.mean(Eta.flatten())

# Choose an arbitrary index to plot to check the reshapign
ind = 6
# fig3, ax3 = plt.subplots()
# pc = ax3.pcolormesh(xx, yy, np.reshape(Eta[:, ind], (N, N)), vmin=0, vmax=1)

U, s, v = np.linalg.svd(Eta)

n_svs = len(s[s>1])
print('Keeping %d singular values' % n_svs)

K = U[:, :n_svs]

fig, ax = plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
ax = ax.flatten()
for l in range(n_svs):
    ax[l].pcolormesh(xx, yy, np.reshape(K[:, l], (N, N)), cmap=cmocean.cm.balance)

# fig.savefig('PCA_components.png', dpi=600)
# Recreate the experimental data
Eta_pc = np.zeros(Eta.shape)
W_pc = np.zeros((m, n_svs))
for j in range(m):
    # eta_pc = np.zeros((int(N*N)))
    for k in range(n_svs):
        w_k = np.matmul(Eta[:, j].T, K[:, k])
        Eta_pc[:, j] += w_k*K[:, k]
        W_pc[j, k] = w_k

fig4, axes = plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
axes[0][0].pcolormesh(xx, yy, np.reshape(Eta[:, ind], (N, N)), vmin=0, vmax=1,
    cmap=cmocean.cm.haline)
axes[0][0].set_title('Experimental data')
axes[0][1].pcolormesh(xx, yy, np.reshape(Eta_pc[:,ind],(N,N)), vmin=0, vmax=1,
    cmap=cmocean.cm.haline)
axes[0][1].set_title('PCA representation (p=%d)' % n_svs)
Delta = np.reshape(Eta_pc[:, ind] - Eta[:, ind], (N, N))
em = axes[1][0].pcolormesh(xx, yy, Delta, vmin=-1e-3, vmax=1e-3, cmap=cmocean.cm.balance)
axes[1][0].set_title('Residual')
fig4.colorbar(em, ax=axes[1][1])


X_pls = np.zeros((m, 2+2*N*N))
X_pls[:, :2] = X
for i in range(m):
    X_pls[i,2:2+N*N] = xx.flatten()
    X_pls[i, 2+N*N:] = yy.flatten()

n_comps = list(range(1, 5))
rmse_plot = np.zeros(len(n_comps))
for j in range(len(n_comps)):
    plsr = sklearn.cross_decomposition.PLSRegression(n_components=n_comps[j])
    plsr.fit(X_pls, Eta.T)

    preds = plsr.predict(X_pls)
    # print(preds.shape)
    # print(Eta.shape)
    rmse_plot[j] = np.sqrt(mse(Eta.T, preds))

fig, ax = plt.subplots()
ax.semilogy(n_comps, (rmse_plot))
ax.set_xlabel('Number of PLS components')
ax.set_ylabel('RMSE')
fig.savefig('PLS_simulator_RMSE.png', dpi=600)

Xarr = X
Yarr = Eta.T
# # Validation
q = 20
X_val = np.zeros((q, 2+2*N*N))
X_val[:, :2] = pyDOE.lhs(2, samples=q, criterion='cm', iterations=250)
X_val[:, 2:] = X_pls[:q, 2:]

i_val = 6

plsr = sklearn.cross_decomposition.PLSRegression(n_components=2, scale=True)
plsr.fit(X_pls, Eta.T)
pred = plsr.predict(X_val)
# print(pred.shape)
pred_val = pred[i_val]

fig, ax = plt.subplots(ncols=3, figsize=(8, 3), sharey=True)
ax[0].pcolormesh(xx, yy, np.reshape(Eta[:, ind], (N, N)))
ax[1].pcolormesh(xx, yy, np.reshape(Eta_pc[:,ind],(N,N)))
ax[-1].pcolormesh(xx, yy, pred_val.reshape((N, N)))

ax[0].set_title('Exact')
ax[1].set_title('PCA')
ax[2].set_title('PLS')

fig.savefig('PLS_simulator.png', dpi=600)

plt.show()
