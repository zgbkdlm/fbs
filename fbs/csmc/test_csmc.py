



# LGSSM: test mean, cov, and bivariate cov C(X_t, X_{t+1} | Y_{0:T}) for T = 25


# Non-linear case:
# If x is distributed according to the prior dynamics, then sample y | x, then use CSMC to sample x | y
# This is a Gibbs sampler that keeps the prior distribution of x intact.
# p(x) Gaussian, p(y | x) something weird.