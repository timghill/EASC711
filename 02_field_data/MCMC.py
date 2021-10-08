"""
Homemade Markov Chain Monte Carlo methods module

Built on top of numpy and scipy.stats libraries, using Metropolis-
Hastings algorithm
"""

import numpy as np
import scipy.stats

class Model:
    """
    MCMC model representation

    Class functions step, chain, and calculate_pdf make working with
    MCMC methods easy and intuitive
    """
    _default_pdf = scipy.stats.norm().rvs
    _default_jdf = scipy.stats.norm().pdf
    def __init__(self, jumping_model=_default_jdf, sample_pdf=_default_pdf):
        """
        Initialize Model instance

        Model attributes control the details of the MCMC method:
         * `jumping_model` (jumping distribution function): Jumping
            distribution used to draw new samples. Function with call
            signature:

            ```python
            theta_prop = jumping_model(theta)
            ```

            where `theta` is the current step in the Metropolis-Hastings
            algorithm

         * `sample_pdf`: Sample probability density function. Function with
            call signature:

            ```python
            prob_theta = sample_pdf(theta)
            ```

        At each step in the MCMC chain, let the current variable value be
        theta. Then, a new proposed value is drawn from the jumping
        distribution:

        ```python
        theta_prop = self.jumping_model()
        ```

        The probability of the original and proposed values are calculated:

        ```python
        p_theta = self.prob_model(theta)
        p_prop = self.prob_model(theta_prop)

        accep = min(1, p_prop/p_theta)
        ```

        And the new value (`theta_prop`) is accepted with a probability
        equal to the acceptance ratio
        """
        # self.jumping_model = jumping_model
        self.sample_pdf = sample_pdf
        self._uniform = scipy.stats.uniform()

    def step(self, theta):
        """Take one step with Metropolis-Hastings algorithm

        Arguments:

         * `theta`: Current step

        Generates a random proposed candidate from the `self.jumping_model`
        distribution, and accepts the candidate according to its
        acceptance ratio (ratio of probability of new and current
        steps)

        Returns new accepted value
        """
        # Generate random proposed candidate
        # theta_rv = self.jumping_model(theta)
        # theta_prop = theta_rv.rvs()
        theta_prop = self.jumping_model(theta)

        # Calculate probability of old and new guesses
        p_theta = self.sample_pdf(theta)
        p_theta_prop = self.sample_pdf(theta_prop)

        # Calculate acceptance ratio
        accep = min(1, p_theta_prop/p_theta)

        # Accept with a probability alpha
        u = self._uniform.rvs()
        if u<= accep:
            theta_new = theta_prop
        else:
            theta_new = theta

        # Return accepted theta value
        return theta_new

    def chain(self, theta0, steps=1e3, discard=0):
        """Generate a complete Markov Chain using Metropolis-Hastings
        algorithm

        Arguments:

         * `theta0`: Starting sample
         * `steps`: Total number of steps to take
         * `discard`: Number of steps to discard from the beginning of
            the chain

        Returns:

         * `samples`: `steps - discard`-length array containing
            samples in the chain
        """
        # Enforce type int
        steps = int(steps); discard = int(discard)

        # samples = np.zeros((steps - discard, len(theta0)))
        samples = np.zeros(steps - discard)
        # print(samples.shape)
        theta = theta0
        # Steps where we forget the result
        for i in range(discard):
            theta = self.step(theta)
        # Save the result for the rest of the steps
        for j in range(discard, steps):
            theta = self.step(theta)
            samples[j-discard] = theta

        return samples

    def calculate_pdf(self, samples, **kde_params):
        """Estimate probability density function from MCMC chain.

        This is a simple wrapper to `scipy.stats.gaussian_KDE` function

        Arguments:

         * `samples`: MCMC samples (e.g. from `chain` function)
         * `kde_params`: Additional parameters to pass to
            `scipy.stats.gaussian_KDE function`
        """
        kde = scipy.stats.gaussian_kde(samples, **kde_params)
        return kde
