"""
Script to test generating two-dimensional data and calculating principal
components
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import cmocean
import pyDOE

import spm

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
        pert_channel[qhalf-1, :qwidth] = beta
        for i in range(1, qhalf):
            pert_channel[qhalf  - 1 + i, qwidth + i -1] = beta
            pert_channel[qhalf - i - 1, qwidth + i -1] = beta

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

# Experimental design
np.random.seed(12)
X = pyDOE.lhs(2, samples=m, criterion='cm', iterations=250)
#
fig, ax = plt.subplots()
pc = ax.pcolormesh(xx, yy, phys_model(1, 1), vmin=-0, vmax=1)
# ax.contour(xx, yy, phys_model(1, 1), colors='k', linewidths=0.5)
fig.colorbar(pc)
# ax.set_title('Simulator')
fig.savefig('simulator.png', dpi=600)

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

fig.savefig('PCA_components.png', dpi=600)
# Recreate the experimental data
Eta_pc = np.zeros(Eta.shape)
W_pc = np.zeros((m, n_svs))
for j in range(m):
    # eta_pc = np.zeros((int(N*N)))
    for k in range(n_svs):
        w_k = np.matmul(Eta[:, j].T, K[:, k])
        Eta_pc[:, j] += w_k*K[:, k]
        W_pc[j, k] = w_k

print('Variance captured:')
print(np.var(Eta_pc)/np.var(Eta))

fig4, axes = plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
axes[0][0].pcolormesh(xx, yy, np.reshape(Eta[:, ind], (N, N)), vmin=0, vmax=1,
    cmap=cmocean.cm.haline)
axes[0][0].set_title('Experimental data')
axes[0][1].pcolormesh(xx, yy, np.reshape(Eta_pc[:,ind],(N,N)), vmin=0, vmax=1,
    cmap=cmocean.cm.haline)
axes[0][1].set_title('PCA representation (p=%d)' % n_svs)
Delta = np.reshape(Eta_pc[:, ind] - Eta[:, ind], (N, N))
axes[1][0].pcolormesh(xx, yy, Delta, vmin=-1e-3, vmax=1e-3, cmap=cmocean.cm.balance)
axes[1][0].set_title('Residual')

fig4.savefig('PCA_representation.png', dpi=600)

# Now make three independent GP emulators
stoch_models = []
for comp in range(n_svs):
    y = np.vstack(W_pc[:, comp])
    res = spm.solve_stochastic_model(X, y, x0=[1, 1], method='SLSQP')
    stoch_models.append(res.copy())

# A non-training point
xs = [0.425, 0.525]
w_emulator = np.array([spm.predictor(xs, X, np.vstack(W_pc[:, i]), **stoch_models[i])[0] for i in range(n_svs)])

Y_emulator = np.matmul(K, w_emulator)
Z_emulator = np.reshape(np.matmul(K, w_emulator), (N, N))
Z_phys = phys_model(*xs)
fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
ax[0][0].pcolormesh(Z_phys, vmin=0, vmax=1)
ax[0][0].set_title('Physical')
ax[0][1].pcolormesh(Z_emulator, vmin=0, vmax=1)
ax[0][1].set_title('Emulator')
pc = ax[1][0].pcolormesh(Z_emulator - Z_phys, vmax=1e-3, vmin=-1e-3, cmap=cmocean.cm.balance)
ax[1][0].set_title('Error')
fig.colorbar(pc)

fig.savefig('emulator_performance.png', dpi=600)

plt.show()
