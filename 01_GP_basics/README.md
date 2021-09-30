# Stochastic process models
Recreating the stochastic process model from "Efficient Global Optimization of Expensive Black-Box Functions" (Jones et al.).

The statistical methods are implemented in the module `spm.py`.

The examples are as follows:

 * `1d_test_simple.py`: Applies GP model to a simple, smooth one-dimensional function, assuming the covariance exponents are p=2. Includes cross-validation, which does not work too well with only a few sample points in one-dimension
 * `1d_test_full_model.py`: Same as above, but includes covariance exponents in optimization
 * `branin.py': Recreate Branin test function example from J-S-W paper, assuming covariance exponents p=2. Includes cross-validation and metrics
 * `branin_full.py`: Same as above, but including exponents in optimization
