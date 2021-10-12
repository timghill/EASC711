# Incorporating field data
Implements the simplest (fast simulator) and medium (expensive but accurate simulator) class of methods from Higdon et al. (2004). Applies method to a simple one-dimensional example: dampled simple harmonic motion (underdamped case).

Module `MCMC.py` implements slightly generalized Metropolis-Hastings MCMC algorithm which should be applicable to future examples. At the moment in only handles the univariate case.

`higdon_simple.py` implements the fast-simulator case, and `hidgon_medium.py` implements the expensive simulator case (with no discrepancy).
