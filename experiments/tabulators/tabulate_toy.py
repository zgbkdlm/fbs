"""
Print the error statistics of the toy experiment.

Run this script under the folder `./experiments`
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from fbs.utils import kl
from functools import partial

jax.config.update("jax_enable_x64", True)

sde = 'const'
nparticles = 10

methods = [f'filter-{sde}-{nparticles}',
           f'gibbs-eb-{sde}-{nparticles}',
           f'pmcmc-0.005-{sde}-{nparticles}',
           f'twisted-{sde}-{nparticles}']
max_mcs = 100
q = 0.95

errs_m = np.zeros((max_mcs,))
errs_vars = np.zeros((max_mcs,))  # marginal variances
errs_kl = np.zeros((max_mcs,))
errs_skew = np.zeros((max_mcs,))
errs_kurt = np.zeros((max_mcs,))

# Tabulate
for method in methods:
    for mc_id in range(max_mcs):

        # Load
        filename = f'./toy/results/{method}-{mc_id}.npz'
        results = np.load(filename)
        samples, gp_mean, gp_cov = results['samples'], results['gp_mean'], results['gp_cov']

        if any([s in method for s in ['gibbs', 'pmcmc']]):
            approx_means = jax.vmap(partial(jnp.mean, axis=0))(samples)
            approx_covs = jax.vmap(partial(jnp.cov, rowvar=False))(samples)

            err_m = jnp.mean(jnp.abs(approx_means - gp_mean[None, :]))
            err_var = jnp.mean(jnp.abs(jnp.diagonal(approx_covs - gp_cov[None, :, :], axis1=1, axis2=2)))
            err_kl = jnp.mean(jax.vmap(kl, in_axes=[None, None, 0, 0])(gp_mean, gp_cov, approx_means, approx_covs))
            err_skew = jnp.mean(jnp.abs(scipy.stats.skew(samples, axis=1)))
            err_kurt = jnp.mean(jnp.abs(scipy.stats.kurtosis(samples, axis=1, fisher=True)))
        else:
            approx_mean = jnp.mean(samples, axis=0)
            approx_cov = jnp.cov(samples, rowvar=False)

            err_m = jnp.mean(jnp.abs(approx_mean - gp_mean))
            err_var = jnp.mean(jnp.abs(jnp.diag(approx_cov) - jnp.diag(gp_cov)))
            err_kl = kl(gp_mean, gp_cov, approx_mean, approx_cov)
            err_skew = jnp.mean(jnp.abs(scipy.stats.skew(samples, axis=0)))
            err_kurt = jnp.mean(jnp.abs(scipy.stats.kurtosis(samples, axis=0, fisher=True)))

        errs_m[mc_id] = err_m
        errs_vars[mc_id] = err_var
        errs_kl[mc_id] = err_kl
        errs_skew[mc_id] = err_skew
        errs_kurt[mc_id] = err_kurt

    print(f'Method {method} | '
          f'Mean {jnp.mean(errs_m):.4f} {jnp.std(errs_m):.4f} | '
          f'Var {jnp.mean(errs_vars):.4f} {jnp.std(errs_vars):.4f} | '
          f'KL | {jnp.mean(errs_kl):.4f} {jnp.std(errs_kl):.4f} | '
          f'Skew | {jnp.mean(errs_skew):.4f} {jnp.std(errs_skew):.4f} | '
          f'Kurt | {jnp.mean(errs_kurt):.4f} {jnp.std(errs_kurt):.4f}')

