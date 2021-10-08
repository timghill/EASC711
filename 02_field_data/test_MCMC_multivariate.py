import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import cmocean

import MCMC

"""
One-dimensional exampe
"""
# Parameters
a = 2
x_eval = np.linspace(0, 1, 51, endpoint=True)

model = MCMC.Model()
# prior = lambda theta: stats.norm.pdf(x, loc=4.5, scale=1)
# model.sample_pdf = lambda theta: stats.beta.pdf(x, a, theta)*prior(theta)
model.jumping_model = lambda loc: stats.norm.rvs(loc, 0.3)
model.sample_pdf = lambda x: 10*stats.beta.pdf(x, a, 5)

x0 = 0.5
chain = model.chain(x0, steps=5e3, discard=1e3)
probs = model.calculate_pdf(chain)

fig, ax = plt.subplots()
ax.plot(chain)

fig, ax = plt.subplots()
ax.plot(x_eval, model.sample_pdf(x_eval)/10)
ax.plot(x_eval, probs(x_eval))


"""
Two-dimensional example
"""

mv_model = MCMC.Model()
factor = 1
# Probability function: Mulivariate normal distribution
cov = np.array([[1, 0.8], [0.8, 1]])
sample_pdf = lambda x, y: factor*stats.multivariate_normal.pdf([x, y], mean=[0, 0], cov=cov)
jumping_cov = (0.5**2)*np.array([[1, -0.5], [-0.5, 1]])
jumping_model = lambda loc: stats.multivariate_normal.rvs(mean=loc, cov=jumping_cov)
# print(sample_pdf(0.5, 0.5))
v = [0.4, 0.5]

steps = int(1e4)
discard = int(1e3)
samples = np.zeros((steps-discard, len(v)))
for i in range(steps):
    v_prop = jumping_model(v)
    p_theta_prop = sample_pdf(*v_prop)
    p_theta = sample_pdf(*v)
    accep = min(1, p_theta_prop/p_theta)

    u = stats.uniform.rvs()
    if u<=accep:
        v = v_prop
    else:
        v = v

    if i>discard:
        samples[i-discard] = v

KDE = stats.gaussian_kde(samples.T)

x = np.linspace(-2, 2, 101, endpoint=True)
y = np.linspace(-2, 2, 101, endpoint=True)

xx_plot, yy_plot = np.meshgrid(x, y)

xx = xx_plot[:-1, :-1]
yy = yy_plot[:-1, :-1]


positions = np.vstack([xx.ravel(), yy.ravel()])
Z = np.reshape(KDE(positions).T, xx.shape)
xv = np.vstack(xx.ravel())
yv = np.vstack(yy.ravel())
Z_exact = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z_exact[i, j] = sample_pdf(xx[i,j], yy[i, j])
# values = np.vstack([m1, m2])
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)

fig1, (ax1, ax2) = plt.subplots(ncols=2)
pc = ax1.pcolormesh(xx_plot, yy_plot, Z_exact, cmap=cmocean.cm.dense, vmin=0, vmax=0.2)
# fig1.colorbar(pc, ax=ax1)

pc2=ax2.pcolormesh(xx_plot, yy_plot, Z, cmap=cmocean.cm.dense, vmin=0, vmax=0.2)
# fig1.colorbar(pc2, ax=ax2)
#
# cov = np.array([[1, 0.5], [0.5, 1]])
#
# z = stats.multivariate_normal.pdf(np.dstack((xx, yy)), mean=[0, 0], cov=cov)
#
# fig1, ax1 = plt.subplots()
# ax1.pcolormesh(xx_plot, yy_plot, z, shading='flat', cmap=cmocean.cm.dense)
#
plt.show()
