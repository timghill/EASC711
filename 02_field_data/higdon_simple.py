import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate
import scipy.stats
import scipy.optimize

# True parameter values
zeta = 0.5  # The only varying parameter
w0 = 2*np.pi

def damped_sho_rhs(t, y, zeta, w0):
    return np.array([y[1], -w0**2*y[0] - 2*zeta*w0*y[1]])

# Simple, fast function: eta(x, t), where x = spatial coordinate
# and t = (zeta, w0) unobservable tuning parameters
objective = lambda t, y: damped_sho_rhs(t, y, zeta, w0)


# Plot the exact curve
t_span = [0, 1]
t_eval = np.linspace(0, 1, 49)
y0 = np.array([1, w0])

sol = scipy.integrate.solve_ivp(objective, t_span, y0, t_eval=t_eval)
fig, ax = plt.subplots()

# Sample points
x = np.array([0.2, 0.3, 0.5, 0.8])
sol_exact = scipy.integrate.solve_ivp(objective, t_span, y0, t_eval=x)
y_exact = sol_exact.y[0]
measure_sigma = 0.1
np.random.seed(1234567)
y = y_exact + np.random.normal(scale=measure_sigma, size=x.shape)


def eta(t):
    object = lambda t, y: damped_sho_rhs(t, y, t, w0)
    tspan = [np.min(x), np.max(x)]
    y0 = np.array([1, w0])
    ss = scipy.integrate.solve_ivp(object, tspan, y0, t_eval=x)
    return ss.y[0]

# Prior parameter distribution
#  zeta ~ N(0.5, 0.1)
#  w0 ~ N(2*pi, pi\8)

N_priors = 15
zeta_prior = np.random.normal(loc=0.5, scale=0.1, size=(N_priors))
# w0_prior = np.random.normal(loc=2*np.pi, scale=np.pi/8, size=(N_priors))
y_prior = np.zeros((N_priors, len(t_eval)))

for i in range(N_priors):
    obji = lambda t, y: damped_sho_rhs(t, y, zeta_prior[i], w0)
    soli = scipy.integrate.solve_ivp(obji, t_span, y0, t_eval=t_eval)
    y_prior[i] = soli.y[0]

ax.set_xlabel('x')
ax.set_ylabel('y, $\\eta(x, t)$')
ax.plot(soli.t, y_prior.T, color=(0.5, 0.5, 0.5, 0.5))
ax.plot(sol.t, sol.y[0, :], label='Exact', linewidth=2, color='r')
ax.errorbar(x, y, yerr=1.96*measure_sigma, fmt='k.',
    ecolor='k', label='Samples + 95%')
ax.legend()
ax.set_ylim([-0.5, 1.25])

# More meaty statistics: define sampling model for y, Likelihood of y
# given eta(theta)

def likelihood(y, eta):
    return np.exp(-0.5*np.linalg.norm(y - eta)/(measure_sigma**2))

def posterior(theta, y):
    eta_star = eta(theta)
    zeta_rv = scipy.stats.norm(loc=0.5, scale=0.1)
    prob_theta = zeta_rv.pdf(theta)
    return likelihood(y, eta_star)*prob_theta


def MH_step(theta_t):
    theta_star = np.random.normal(loc=0.5, scale=5*0.1)
    p_theta_t = posterior(theta_t, y)
    p_theta_s = posterior(theta_star, y)
    alpha = np.min([1, p_theta_s/p_theta_t])

    rng = np.random.default_rng()
    rv = rng.uniform()
    # print(rv)
    if rv<alpha:
        theta_new = theta_star
    else:
        theta_new = theta_t

    return theta_new

# print(MH_step(0.6))
MCMC_steps = int(1e4)
MCMC_samples = np.zeros(int(MCMC_steps))
theta = 0.5
MCMC_samples[0] = theta
for i in range(MCMC_steps):
    theta = MH_step(theta)
    MCMC_samples[i] = theta

fig2, ax2 = plt.subplots()
ax2.plot(MCMC_samples)

MCMC_subsample = MCMC_samples[int(5e3):]
fig3, ax3 = plt.subplots()
# ax3.hist(MCMC_subsample)

KDE = scipy.stats.gaussian_kde(MCMC_subsample)
pos = np.linspace(0, 1, 101)
pdist = KDE(pos)
ax3.plot(pos, pdist)

# Calculate 15th percentile
percentiles = np.arange(0.15, 0.9, 0.05)
# print(percentiles)
percentile_locs = np.zeros(percentiles.shape)
for i, p in enumerate(percentiles):
    min_obj = lambda x: (KDE.integrate_box_1d(0, x) - p)**2
    if p<0.5:
        x0 = 0.2
    elif p<0.8:
        x0 = 0.6
    else:
        x0 = 0.8

    res = scipy.optimize.minimize(min_obj, x0, method='SLSQP', tol=1e-6)
    # print(res.x)
    percentile_locs[i] = res.x
    ax3.axvline(res.x, color='k')

theta_percentiles = percentile_locs
# print(theta_percentiles)

fig4, ax4 = plt.subplots()
for k in range(len(percentiles)):
    object = lambda t, y: damped_sho_rhs(t, y, theta_percentiles[k], w0)
    y0 = np.array([1, w0])
    ss = scipy.integrate.solve_ivp(object, t_span, y0, t_eval=t_eval)
    yk = ss.y[0]

    ax4.plot(t_eval, yk, color=(0.5, 0.5, 0.5, 0.5))

ax4.errorbar(x, y, yerr=1.96*measure_sigma, fmt='k.',
    ecolor='k', label='Samples + 95%')
ax4.set_ylim([-0.5, 1.25])
plt.show()
