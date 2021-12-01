import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

import pyDOE
import cmocean

import MCMC

# CONSTANTS
g = 9.81
n = 3
A = 2.4e-24

# PARAMETERS
f = 1.5
f_0 = 1.5
DDF_ice = 1
snow = 1.5

# Standard values
h_0 = 3e3           # Greenland max thickness (m)
area_0 = 1.71e12    # Greenland surface area (m2)
vol_0 = 2.86e15     # Greenland ice volume (m3)

def _scale(x, vmin, vmax):
    return (x - vmin)/(vmax - vmin)

def _inv_scale(x, vmin, vmax):
    return x*(vmax-vmin) + vmin

def model(f, snow, DDF_ice):
    """
    Very coarsely represent Greenland Ice Sheet max thickness, surface
    area, and volume as a function of flow enhancement factor f, snow
    accumulation snow, and ice melt DDF DDF_ice.
    """

    # Inverse scaling for input factors
    f_min = 0.5
    f_max = 1.5
    f_phys = _inv_scale(f, f_min, f_max)

    snow_min = 0.8
    snow_max = 1.2
    snow_phys = _inv_scale(snow, snow_min, snow_max)

    DDF_ice_min = 0.8
    DDF_ice_max = 1.2
    DDF_ice_phys = _inv_scale(DDF_ice, DDF_ice_min, DDF_ice_max)

    q_0 = h_0**(n+2)*snow_phys*(1 - 1/10*DDF_ice_phys**2)
    q_new_factor = f_phys

    h_new = (q_0/q_new_factor)**(1/(n+2))

    h_avg = (q_0/q_new_factor/DDF_ice_phys)**(1/(n+2))
    area_new = area_0*(1 +(1/25)*f_phys**2)*snow_phys/DDF_ice_phys
    vol_new = vol_0*(h_new/h_0)*(area_new/area_0)#*(DDF_ice_phys/snow_phys)
    return (h_new/h_0, area_new/area_0, vol_new/vol_0)


def proposal_dist(para_vec):
    jumping_cov = np.diag((0.1)**2*np.ones(3))
    para_new = scipy.stats.multivariate_normal.rvs(para_vec, cov=jumping_cov)
    return para_new

def ism_bayes(figname, measurements=np.array([0, 1, 2])):
    prior_h = scipy.stats.norm(0.5, 0.25)
    prior_a = scipy.stats.norm(0.5, 0.25)
    prior_v = scipy.stats.norm(0.5, 0.25)

    # prior_mu = scipy.stats.norm(1, 0.25)
    # prior_b1 = scipy.stats.norm(0.1, 0.025)
    # prior_b2 = scipy.stats.norm(0.1, 0.025)
    # prior_b3 = scipy.stats.norm(0.1, 0.025)

    prior_xx = np.linspace(0, 1, 101)

    # MCMC Model
    measure_sigma = 0.05

    def likelihood(para):
        y_sim = np.array(model(*para))[measurements]
        y_obs = np.array(model(0.5, 0.5, 0.5))[measurements]
        return np.exp(-0.5*np.sum(y_obs - y_sim)**2/measure_sigma**2)

    def posterior(para):
        L = likelihood(para)
        return L*prior_h.pdf(para[0])*prior_a.pdf(para[1])*prior_v.pdf(para[2])

    MCMC_model = MCMC.Model(sample_pdf=posterior)
    MCMC_model.jumping_model = lambda theta: proposal_dist(theta)
    steps = 10000
    discard = 5000
    theta0 = np.array([0.49, 0.52, 0.51])
    samples = MCMC_model.mv_chain(theta0, steps=steps, discard=discard)

    means = np.tile(np.mean(samples, axis=0), (samples.shape[0], 1))

    [xx, yy] = np.meshgrid(prior_xx, prior_xx)
    val_arr = np.array([xx.flatten(), yy.flatten()])

    priors = [prior_h, prior_a, prior_v]
    para_labels = ['F', 'S', 'D']
    fig, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3, sharex=True, sharey=True)
    for i in range(3):
        for j in range(i):
            kde = scipy.stats.gaussian_kde(samples[:, [j, i]].T)
            kde_vals = kde(val_arr).reshape(xx.shape)
            ax[i][j].pcolor(xx, yy, kde_vals, cmap=cmocean.cm.dense)


        # Calculate prior PDF
        prior_dist = priors[i].pdf(prior_xx)

        # Calculate posterior PDF
        uni_samples = means.copy()
        uni_samples[:, i] = samples[:, i]
        kde_uni = scipy.stats.gaussian_kde(samples.T)


        prior_samples = means[:len(prior_xx)].copy()
        prior_samples[:, i] = prior_xx
        post_dist = kde_uni(prior_samples.T)
        dx = prior_xx[1] - prior_xx[0]
        post_area = np.sum(dx*post_dist)
        post_dist = post_dist/post_area

        all_vals = np.array([prior_dist, post_dist])
        PDF_NORM = np.max(all_vals)

        ax[i][i].plot(prior_xx, prior_dist/PDF_NORM)
        ax[i][i].plot(prior_xx, post_dist/PDF_NORM)

        ax[i][i].set_xlim([0, 1])
        ax[i][i].set_ylim([0, 1])
        ax[i][i].spines['right'].set_visible(False)
        ax[i][i].spines['top'].set_visible(False)

        ax[i][0].set_ylabel(para_labels[i])
        ax[-1][i].set_xlabel(para_labels[i])

    plt.tight_layout()

    fig.savefig(figname, dpi=600)

if __name__ == '__main__':
    # Test simple ice sheet model
    ffs = np.linspace(0, 1, 101)
    snows = np.linspace(0, 1, 101)
    DDF_ices = np.linspace(0, 1, 101)

    # Plot ice thickness, area, and volume for varying flow enhancement factor,
    # snow accumulation, and ice DDF
    fig, ax = plt.subplots(ncols=3, figsize=(9, 3), sharey=True)

    h_f, a_f, v_f = model(ffs, 0.5, 0.5)
    ax[0].plot(ffs, h_f, label='Max thick')
    ax[0].plot(ffs, a_f, label = 'Area')
    ax[0].plot(ffs, v_f, label = 'Volume')
    ax[0].set_xlabel('Flow enhancement factor')
    ax[0].legend()
    ax[0].grid()

    h_s, a_s, v_s = model(0.5, snows, 0.5)
    ax[1].plot(snows, h_s, label='Max thick')
    ax[1].plot(snows, a_s, label='Area')
    ax[1].plot(snows, v_s, label='Volume')
    ax[1].set_xlabel('Snow accumulation')
    ax[1].grid()

    h_i, a_i, v_i = model(0.5, 0.5, DDF_ices)
    ax[2].plot(DDF_ices, h_i, label='Max thick')
    ax[2].plot(DDF_ices, a_i, label='Area')
    ax[2].plot(DDF_ices, v_i, label='Volume')
    ax[2].set_xlabel('DDF ice')
    ax[2].grid()

    plt.tight_layout()
    fig.savefig('ism_model.png', dpi=600)

    np.random.seed(101)
    X_obs = pyDOE.lhs(3, 5**3, 'cm', 50)

    h_obs, area_obs, vol_obs = model(X_obs[:,0], X_obs[:,1], X_obs[:,2])

    fig_obs, ax_obs = plt.subplots(figsize=(8, 3), ncols=3, sharey=True)

    xlabels = ['Flow enhancement', 'Snow', 'DDF ice']
    for i in range(3):
        ax_obs[i].scatter(X_obs[:, i], h_obs)
        ax_obs[i].scatter(X_obs[:, i], area_obs)
        ax_obs[i].scatter(X_obs[:, i], vol_obs)

        ax_obs[i].grid()
        ax_obs[i].set_xlim([0, 1])
        ax_obs[i].set_xlabel(xlabels[i])

    plt.tight_layout()
    fig_obs.savefig('ism_basic_sensitivity.png', dpi=600)

    # ism_bayes('ism_simple_bayes.png')
    ism_bayes('ism_simple_bayes_2.png', measurements=2)

    plt.show()
