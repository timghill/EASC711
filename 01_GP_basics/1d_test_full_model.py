"""
Test stochastic process model code one a one-dimensional example,
simultaneously estimating theta and p values
"""

import numpy as np
from matplotlib import pyplot as plt

import spm

## Data
x = np.array([-0.75, -0.5, -0.2, 0.2, 0.4, 0.75])
# x = np.array([-0.75, -0.6, -0.5, -0.2, 0, 0.2, 0.4, 0.75])

x = np.vstack(x)
n = x.shape[0]
func = lambda x: (x-1)*(x+1)*np.sin(2*np.pi*x)
y = func(x)

stoch_model = spm.solve_full_stochastic_model(x, y, x0=[20, 2], tol=1e-12)
mu_hat = stoch_model['mu']
sigma2_hat = stoch_model['sigma2']
theta_hat = stoch_model['theta']
p = stoch_model['p']
R_hat = stoch_model['R']
Rinv = np.linalg.inv(R_hat)

# For plotting: show likelihood on fine grid
thetas = np.linspace(0.1, 100, 101)
all_likeli = [spm.likelihood(x, y, t, p) for t in thetas]
fig, ax = plt.subplots()
ax.plot(thetas, all_likeli, label='Likelihood')
ax.axvline(x=theta_hat, color='r', label='$\\hat\\theta$')
ax.set_title('Likelihood function optimization')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Likelihood')
ax.legend()
fig.savefig('01_1d_full_likelihood.png', dpi=600)

# Calculate error on fine grid for plotting
xx = np.linspace(-1, 1, 201)
yp = np.zeros(xx.shape)
err = np.zeros(xx.shape)

for i in range(xx.shape[0]):
    [ystar, s2] = spm.predictor(xx[i], x, y, theta_hat, p, mu_hat, sigma2_hat, Rinv)
    s = np.sqrt(s2)
    yp[i] = ystar
    err[i] = s

fig, ax = plt.subplots()
ax.plot(xx, func(xx), 'k', label='True')
data=ax.plot(x, y, 'ko', label='Data')
pred=ax.plot(xx, yp, label='Predictor')
ax.fill_between(xx, yp-err, yp+err, alpha=0.2, label='Std error')

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y = f(x)')
ax.set_title('Predictor')
fig.savefig('01_1d_full_predictor.png', dpi=600)

## Cross-validation
# Vector of predictions at point x[i] calculated after removing the point from the model
y_cv = np.zeros(x.shape)
err_cv = np.zeros(x.shape)

for k in range(n):
    # Create cross-validation data arrays (xm, ym)
    xm = list(x)
    x_cv = xm.pop(k)
    xm = np.array(xm)
    # print(xm)

    ym = list(y)
    ym.pop(k)
    ym = np.array(ym)

    cv_model = spm.solve_full_stochastic_model(xm, ym, x0=[10, 2])
    mu_hat = cv_model['mu']
    sigma2_hat = cv_model['sigma2']
    theta_hat = cv_model['theta']
    p = cv_model['p']
    R_cv = cv_model['R']
    R_cv_inv = np.linalg.inv(R_cv)
    #
    # # Use same theta_hat, p values as in original regression
    # R_cv = R(xm, theta_hat, p)
    # def predictor(xs, y, theta, p, mu, sigma2, Rinv):
    prediction_cv = spm.predictor(x[k], xm, ym, theta_hat, p, mu_hat, sigma2_hat, R_cv_inv)
    y_cv[k] = prediction_cv[0]
    err_cv[k] = np.sqrt(np.abs(prediction_cv[1]))

    if k==4:
        R_plot = R_cv
        y_plot = ym.copy()
        x_plot = xm.copy()
        R_inv_plot = R_cv_inv

norm_err = (y_cv - y)/err_cv


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
ax1.scatter(y_cv, y)
ax1.set_xlabel('Predicted y')
ax1.set_ylabel('y')
ax1.plot([-1, 1], [-1, 1], 'k')

ax2.scatter(y_cv, norm_err)
ax2.set_xlabel('Predicted y')
ax2.set_ylabel('Norm error')
ax2.axhline(-1, color='k', linestyle='--')
ax2.axhline(1, color='k', linestyle='--')

ax3.plot(x, y, label='y')
ax3.plot(x, y_cv, label = 'Predicted y')
ax3.plot(xx, func(xx), color='k')
ax3.legend()
ax3.set_xlabel('x')
ax3.set_ylabel('y')

ypl = np.zeros(xx.shape)
for l in range(xx.shape[0]):
    ypl[l] = spm.predictor(xx[l], x_plot, y_plot, theta_hat, p, mu_hat, sigma2_hat, R_inv_plot)[0]

ax4.plot(xx, ypl)

fig2.savefig('01_1d_full_crossvalidation.png', dpi=600)
plt.show()
