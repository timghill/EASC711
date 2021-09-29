"""
Script to play around with Markov Chain Monte Carlo methods,
specifically using the Metropolis-Hastings algorithm
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

# Parameters
measure_std = 0.25

RV = stats.beta(2, 5)
def prob(x):
    return RV.pdf(x)
    # norm_rv = stats.norm()
    # return norm_rv.pdf(x)


def MH_step(theta_t):
    theta_star = np.random.normal(loc=theta_t, scale=0.25)
    p_theta_t = prob(theta_t)
    p_theta_s = prob(theta_star)
    alpha = np.min([1, p_theta_s/p_theta_t])

    rng = np.random.default_rng()
    rv = rng.uniform()
    # print(rv)
    if rv<=alpha:
        theta_new = theta_star
    else:
        theta_new = theta_t

    return theta_new

N_steps = int(5e4)
mcmc_steps = np.zeros(N_steps)
mcmc_steps[0] = 0.1
for i in range(1, N_steps):
    mcmc_steps[i] = MH_step(mcmc_steps[i-1])

MCMC_subsample = mcmc_steps[int(1e4):]

KDE = stats.gaussian_kde(MCMC_subsample)
pos = np.linspace(0, 1, 101)
pdist = KDE(pos)

fig, ax = plt.subplots()
ax.plot(pos, pdist)
ax.plot(pos, RV.pdf(pos))

plt.show()
