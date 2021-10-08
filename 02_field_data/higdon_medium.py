"""
Medium-complexity method from Higdon et al. (2004) "COMBINING FIELD DATA
AND COMPUTER SIMULATIONS FOR CALIBRATION AND PREDICTION"

Assumes the computer model (simulator) is expensive, but that the simulator
sufficiently accurately represents the field data - e.g., this method
neglects any systematic error between the two ($\delta(x)$).

Applies the method to a damped, undriven simple harmonic oscillator to
calibrate the damping parameter (zeta).
"""


import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

# Third-party library that provides reasonable latin-hypercube sampling
import pyDOE

# ----------------------------------------------------------------------
# Parameters and distribution parameters
zeta = 0.625  # "True" parameter value
w0 = 2*np.pi  # Fixed value

# Measurement errors are ~ N(0, 0.1)
measure_sigma = 0.1

# Prior distribution is ~ N(prior_mean, prior_sd)
prior_mean = 0.6
prior_sd = 0.2

# Measurement points
x = np.array([0.2, 0.275, 0.489, 0.8])

# Simulator points
np.random.seed(199)
X = pyDOE.lhs(2, samples=15, criterion='cm', iterations=100)

# Scale second coordinate appropriately
theta_prior_rv = scipy.stats.norm(loc=prior_mean, scale=prior_sd)
X[:,1] = theta_prior_rv.ppf(X[:,1])


# MCMC parameters
MCMC_steps = int(5e3)
MCMC_discard = int(1e3)

nobs = len(x)
nsim = X.shape[0]

# ----------------------------------------------------------------------
# Define the physical system

def damped_sho_rhs(t, y, zeta, w0):
    return np.array([y[1], -w0**2*y[0] - 2*zeta*w0*y[1]])

def eta(theta, t_eval=x):
    """Calculate function eta at samples x with parameter value theta.
    """
    object = lambda t, y: damped_sho_rhs(t, y, theta, w0)
    tspan = [0, np.max(t_eval)]
    y0 = np.array([1, w0])
    ss = scipy.integrate.solve_ivp(object, tspan, y0, t_eval=t_eval)
    return ss.y[0]

# ----------------------------------------------------------------------
# Measured values
y_exact = eta(zeta, t_eval=x)
y = y_exact + np.random.normal(loc=0, scale=measure_sigma, size=x.shape)

# Simulator values
y_simulator = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    xi = [float(X[i,0])]
    thetai = X[i,1]
    y_simulator[i] = eta(thetai, t_eval=xi)

# STANDARDIZATION
sim_mean = np.mean(y_simulator)
y_simulator = y_simulator - sim_mean
y = y - sim_mean
y_exact = y_exact - sim_mean

sim_std = np.std(y_simulator)
y_simulator = y_simulator/sim_std
y = y/sim_std
y_exact = y_exact/sim_std

# Plot measurements and samples
fig, ax = plt.subplots()
ax.plot(x, y, 'bo')
ax.plot(X[:, 0], y_simulator, 'ko')
ax.set_title('Measurements and simulations')
fig.savefig('higdon_median_samples.png', dpi=600)

# ----------------------------------------------------------------------
# The next three functions (covariance, covariance_mat, L) provide the
# core statistical methods from Higdon et al. (2004)

def covariance(v1, v2, lambd, beta, alpha):
    """Calculate covariance between samples v1 and v2.

    Higdon et al. (2004) equation (3)
    """
    return (1/lambd)*np.exp(-np.sum(beta*np.abs(v2-v1)**alpha))

def covariance_mat(X, lambd, beta, alpha):
    """Calculate covariance matrix (see function covariance)
    """
    n = X.shape[0]
    Sigma = np.zeros((n, n))
    for i in range(n):
        Sigma[i, i] = covariance(X[i], X[i], lambd, beta, alpha)
        for j in range(i+1, n):
            v1 = X[i]
            v2 = X[j]
            covij = covariance(v1, v2, lambd, beta, alpha)

            Sigma[i,j] = covij
            Sigma[j,i] = covij
    # Ensure symmetry
    return 0.5*(Sigma + Sigma.T)

def L(theta, lambd, beta):
    """Likelihood: Higdon et al. (2004) Equation (4)
    """
    # Create the array of (x, theta) ffor field data and simulator
    Xobs = np.zeros((nsim + nobs, 2))
    Xobs[:nobs, 0] = x
    Xobs[:nobs, 1] = theta
    Xobs[nobs:] = X

    # Combined vector of field data and simulations
    Z = np.zeros((nsim + nobs, 1))
    Z[:nobs] = np.vstack(y)
    Z[nobs:] = np.vstack(y_simulator)

    # Measurement covariance matrix - IID normal
    Sigma_y = (measure_sigma**2)*np.eye(nobs)

    # Combined simulator + field data covariance matrix - this is the
    # GP assumption
    Sigma_eta = covariance_mat(Xobs, lambd, beta, 2)

    # Total covariance is sum of both components
    Sigma = Sigma_eta.copy()
    Sigma[:nobs, :nobs] = Sigma[:nobs, :nobs] + Sigma_y# + 1e-4*np.eye(Sigma_y.shape[0])

    # Calculate Likelihood
    Sigma_inv = np.linalg.inv(Sigma)
    detSigma = np.linalg.det(Sigma)
    dS = 1/np.sqrt(detSigma)
    mu = 0  # Since we have standardized - not always true!
    muvec = mu*np.ones((nsim + nobs, 1))
    arg = -0.5*np.matmul((Z - muvec).T, np.matmul(Sigma_inv, Z - muvec))

    likeli = float(dS*np.exp(arg))

    # Some checks to ensure smooth function
    if np.isnan(likeli) or np.isinf(likeli) or likeli>1e6 or np.isnan(detSigma):
        likeli = 0
    return likeli

# ----------------------------------------------------------------------
# Prior distributions (from Higdon et al. 2004)
def prior_theta(theta):
    """Prior $\theta$ distribution: N(prior_mean, prior_sd)
    """
    return scipy.stats.norm.pdf(theta, loc=prior_mean, scale=prior_sd)

def prior_lambd(lambd):
    """Prior $\lambda$ distribution
    """
    b = 5
    a = 5
    # return lambd**(a-1)*np.exp(-b*lambd)
    return scipy.stats.gamma.pdf(b*lambd, a)

def prior_beta(beta):
    """Prior $\beta$ distributions. Different for distance and zeta
    coefficients. These are found from empirically inspecting the
    Likelihood function
    """
    prior_b1 = scipy.stats.norm.pdf(beta[0], loc=5, scale=1)
    prior_b2 = scipy.stats.beta.pdf(beta[1], 2, 2)
    return float(prior_b1*prior_b2)

# ----------------------------------------------------------------------
# MCMC methods

def MCMC_prob(paravec):
    """Function to apply MCMC methods to. This is the Bayesian
    probability function: Likelihood * priors
    """
    theta = paravec[0]
    lambd = paravec[1]
    beta = paravec[2:]
    likeli = L(theta, lambd, beta)
    priors = prior_theta(theta)*prior_beta(beta)*prior_lambd(lambd)
    return likeli*priors

def jumping_model(para_vec):
    """Model used in selecting a new candidate parameter vector. In this
    case, multivariate normal distribution with zero cross-correlations,
    e.g. parameter candidates are independent.
    """
    theta = para_vec[0]
    lambd = para_vec[1]
    beta = para_vec[2:]
    diags = (0.25**2)*np.ones(4)
    diags[0] = (0.2)**2
    diags[1] = (0.2)**2

    jumping_cov = np.diag(diags)
    para_new = scipy.stats.multivariate_normal.rvs(para_vec, cov=jumping_cov)
    return para_new

# DO MCMC METHOD - METROPOLIS-HASTINGS
# Starting point
para_vec = np.array([0.6, 0.5, 6, 0.6])
mv_chain = np.zeros((MCMC_steps - MCMC_discard, len(para_vec)))
for i in range(MCMC_steps):
    para_prop = jumping_model(para_vec)
    p_theta = MCMC_prob(para_vec)
    p_theta_prop = MCMC_prob(para_prop)
    accep = min(1, p_theta_prop/p_theta)

    u = scipy.stats.uniform.rvs()
    if u<=accep:
        para_vec = para_prop
    else:
        para_vec = para_vec

    if i>MCMC_discard:
        mv_chain[i-MCMC_discard] = para_vec

# Estimate PDF
KDE_model = scipy.stats.gaussian_kde(mv_chain.T)

print('Mean values:', np.mean(mv_chain, axis=0))

# Plot pdfs
fig, ax = plt.subplots()
ax.plot(mv_chain[:, 0])
ax.plot(mv_chain[:, 1])
ax.plot(mv_chain[:, 2])
ax.plot(mv_chain[:, 3])
# plt.show()

fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
pp_theta = np.linspace(0, 2, 101)
pp_lambda = np.linspace(0, 2, 101)
pp_beta1 = np.linspace(1, 10, 101)
pp_beta2 = np.linspace(0, 1.5, 101)

post_theta = np.zeros(pp_theta.shape)
post_lambda = np.zeros(pp_theta.shape)
post_beta1 = np.zeros(pp_theta.shape)
post_beta2 = np.zeros(pp_theta.shape)

means = np.mean(mv_chain, axis=0)
for i in range(len(post_theta)):
    para1 = means.copy()
    para2 = means.copy()
    para3 = means.copy()
    para4 = means.copy()

    para1[0] = pp_theta[i]
    para2[1] = pp_lambda[i]
    para3[2] = pp_beta1[i]
    para4[3] = pp_beta2[i]

    post_theta[i] = KDE_model(para1)
    post_lambda[i] = KDE_model(para2)
    post_beta1[i] = KDE_model(para3)
    post_beta2[i] = KDE_model(para4)

axes[0][0].plot(pp_theta, prior_theta(pp_theta))
axes[0][0].plot(pp_theta, post_theta)
axes[0][0].set_xlabel('$\\theta$')

axes[0][1].plot(pp_lambda, prior_lambd(pp_lambda))
axes[0][1].plot(pp_lambda, post_lambda)
axes[0][1].set_xlabel('$\\lambda$')

# pp_beta = np.vstack(pp_beta1, pp_beta2)
# prior_beta_plot = prior_beta(pp_beta)
# axes[1][0].plot(pp_beta1, prior_beta_plot[:,0])
axes[1][0].plot(pp_beta1, post_beta1)
axes[1][0].set_xlabel('$\\beta_1$')

# axes[1][1].plot(pp_beta2, prior_beta_plot[:, 1])
axes[1][1].plot(pp_beta2, post_beta2)
axes[1][1].set_xlabel('$\\beta_2$')

fig.suptitle('Posterior distributions')
plt.show()
