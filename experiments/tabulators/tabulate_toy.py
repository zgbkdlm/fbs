"""
Print the error statistics of the toy experiment.

Run this script under the folder `./experiments`
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from fbs.utils import kl, bures_dist
from functools import partial

jax.config.update("jax_enable_x64", True)

sde = 'const'
nparticles = 100

methods = [f'filter-{sde}-{nparticles}',
           f'gibbs-eb-{sde}-{nparticles}',
           f'pmcmc-0.005-{sde}-{nparticles}',
           f'pmcmc-0.001-{sde}-{nparticles}',
           f'twisted-{sde}-{nparticles}',
           f'csgm-{sde}']
max_mcs = 100

errs_m = np.zeros((max_mcs,))
errs_vars = np.zeros((max_mcs,))  # marginal variances
errs_kl = np.zeros((max_mcs,))
errs_bures = np.zeros((max_mcs,))
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

            err_m = np.mean(jnp.abs(approx_means - gp_mean[None, :]))
            err_var = np.mean(jnp.abs(jnp.diagonal(approx_covs - gp_cov[None, :, :], axis1=1, axis2=2)))
            err_kl = np.mean(jax.vmap(kl, in_axes=[None, None, 0, 0])(gp_mean, gp_cov, approx_means, approx_covs))
            err_bures = np.mean(jax.vmap(bures_dist, in_axes=[None, None, 0, 0])(gp_mean, gp_cov,
                                                                                 approx_means, approx_covs))
            err_skew = np.mean(jnp.abs(scipy.stats.skew(samples, axis=1)))
            err_kurt = np.mean(jnp.abs(scipy.stats.kurtosis(samples, axis=1, fisher=True)))
        else:
            approx_mean = np.mean(samples, axis=0)
            approx_cov = np.cov(samples, rowvar=False)

            err_m = np.mean(jnp.abs(approx_mean - gp_mean))
            err_var = np.mean(jnp.abs(jnp.diag(approx_cov) - jnp.diag(gp_cov)))
            err_kl = kl(gp_mean, gp_cov, approx_mean, approx_cov)
            err_bures = bures_dist(gp_mean, gp_cov, approx_mean, approx_cov)
            err_skew = np.mean(jnp.abs(scipy.stats.skew(samples, axis=0)))
            err_kurt = np.mean(jnp.abs(scipy.stats.kurtosis(samples, axis=0, fisher=True)))

        errs_m[mc_id] = err_m
        errs_vars[mc_id] = err_var
        errs_kl[mc_id] = err_kl
        errs_bures[mc_id] = err_bures
        errs_skew[mc_id] = err_skew
        errs_kurt[mc_id] = err_kurt

    print(f'Method {method} | '
          f'KL | {np.mean(errs_kl):.4f} {np.std(errs_kl):.4f} | ' 
          f'Bures | {np.mean(errs_bures):.4f} {np.std(errs_bures):.4f} | '
          f'Mean {np.mean(errs_m):.4f} {np.std(errs_m):.4f} | '
          f'Var {np.mean(errs_vars):.4f} {np.std(errs_vars):.4f}')
