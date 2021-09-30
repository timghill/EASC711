import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate
import scipy.stats
import scipy.optimize

import MCMC

# ----------------------------------------------------------------------
# Parameters and distribution parameters
zeta = 0.625  # "True" parameter value
w0 = 2*np.pi  # Fixed value

# Measurement errors are ~ N(0, 0.1)
measure_sigma = 0.1

# Prior distribution is ~ N(prior_mean, prior_sd)
prior_mean = 0.6
prior_sd = 0.2

N_steps = 1e4
discard_frac = 0.1

percentiles = np.arange(0.05, 1, 0.05)

# Sample points
x = np.array([0.2, 0.3, 0.5, 0.8])
# ----------------------------------------------------------------------

# Define the physical system
def damped_sho_rhs(t, y, zeta, w0):
    return np.array([y[1], -w0**2*y[0] - 2*zeta*w0*y[1]])

def eta(theta, t_eval=x):
    """Calculate function eta at samples x with parameter value theta.
    """
    object = lambda t, y: damped_sho_rhs(t, y, theta, w0)
    tspan = [np.min(t_eval), np.max(t_eval)]
    y0 = np.array([1, w0])
    ss = scipy.integrate.solve_ivp(object, tspan, y0, t_eval=t_eval)
    return ss.y[0]

# ----------------------------------------------------------------------

# Plot the exact curve
t_span = [0, 1]
t_eval = np.linspace(0, 1, 49)
y0 = np.array([1, w0])
objective = lambda t, y: damped_sho_rhs(t, y, zeta, w0)
sol = scipy.integrate.solve_ivp(objective, t_span, y0, t_eval=t_eval)
fig, ax = plt.subplots()

# ----------------------------------------------------------------------
sol_exact = scipy.integrate.solve_ivp(objective, t_span, y0, t_eval=x)
y_exact = sol_exact.y[0]
# np.random.seed(2020)

# Add measurement error
y = y_exact + np.random.normal(scale=measure_sigma, size=x.shape)
# ----------------------------------------------------------------------

# Calculate priors to plot
prior_rv = scipy.stats.norm(loc=prior_mean, scale=prior_sd)
theta_prior = [prior_rv.ppf(p) for p in percentiles]
y_prior = np.zeros((len(percentiles), len(t_eval)))
for i in range(len(percentiles)):
    y_prior[i] = eta(theta_prior[i], t_eval=t_eval)

ax.set_xlabel('x')
ax.set_ylabel('y, $\\eta(x, t)$')
ax.plot(t_eval, y_prior.T, color=(0.5, 0.5, 0.5, 0.5))
ax.plot(sol.t, sol.y[0, :], label='Exact', linewidth=2, color='r')
ax.errorbar(x, y, yerr=1.96*measure_sigma, fmt='k.',
    ecolor='k', label='Samples + 95%')
ax.legend()
ax.set_ylim([-1, 1.5])

# More meaty statistics: define sampling model for y, Likelihood of y
# given eta(theta)

def likelihood(y, eta_vals):
    # Calculate likelihood of measurements y given simulator output eta
    return np.exp(-0.5*np.linalg.norm(y - eta_vals)**2/(measure_sigma**2))

def posterior(theta):
    eta_star = eta(theta)
    return likelihood(y, eta_star)*np.exp(-0.5*(theta - prior_mean)**2/prior_sd**2)

MCMC_model = MCMC.Model()
MCMC_model.jumping_model = lambda loc: scipy.stats.norm(loc, scale=0.3)
MCMC_model.sample_pdf = posterior
steps = int(N_steps)
discard = int(N_steps*discard_frac)
theta0 = 0.6
MCMC_subsample = MCMC_model.chain(theta0, steps=steps, discard=discard)

# -------------------------------------------------------------------------
# Figure 3: Prior and posterior distributions
fig3, ax3 = plt.subplots()

pos = np.linspace(0, 2, 101)
KDE = MCMC_model.calculate_pdf(MCMC_subsample)
pdist = KDE(pos)
ax3.plot(pos, pdist, label='MCMC Posterior')
zeta_rv = scipy.stats.norm(prior_mean, prior_sd)
ax3.plot(pos, zeta_rv.pdf(pos), label='Prior')
ax3.legend()

theta_percentiles = np.zeros(percentiles.shape)
emp_sigma = np.std(MCMC_subsample)
emp_mean = np.mean(MCMC_subsample)
emp_norm = scipy.stats.norm(loc=emp_mean, scale=emp_sigma)
print(emp_sigma)
print(emp_mean)
for i, p in enumerate(percentiles):
    min_obj = lambda x: np.abs(KDE.integrate_box_1d(0, x) - p)**2

    # Use normal percentiles to estimate locations
    x0 = emp_norm.ppf(p)
    res = scipy.optimize.minimize(min_obj, x0, tol=1e-12)

    theta_percentiles[i] = res.x
    ax3.axvline(res.x, color='k')

# Figure 4: Eta with posterior parameter values
fig4, ax4 = plt.subplots()
for k in range(len(percentiles)):
    object = lambda t, y: damped_sho_rhs(t, y, theta_percentiles[k], w0)
    y0 = np.array([1, w0])
    ss = scipy.integrate.solve_ivp(object, t_span, y0, t_eval=t_eval)
    yk = ss.y[0]

    ax4.plot(t_eval, yk, color=(0.5, 0.5, 0.5, 0.5))

y_mean = eta(emp_mean, t_eval=t_eval)
ax4.plot(t_eval, y_mean, 'r')

ax4.errorbar(x, y, yerr=1.96*measure_sigma, fmt='k.',
    ecolor='k', label='Samples + 95%')
ax4.set_ylim([-1, 1.5])
plt.show()
