"""
Print the error statistics of the toy experiment.

Run this script under the folder `./experiments`
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npr
import scipy
import matplotlib.pyplot as plt
from fbs.utils import kl

jax.config.update("jax_enable_x64", True)

methods = ['filter', 'gibbs-eb', 'pmcmc-0.01', 'twisted']
method_label = ['', '', '']
max_mcs = 50
max_lags = 100
q = 0.95

errs_kl = np.zeros((max_mcs,))
errs_skew = np.zeros((max_mcs,))
errs_kurt = np.zeros((max_mcs,))
autocorrs = np.zeros((max_mcs, max_lags))

for method in methods:
    for mc_id in range(max_mcs):
        # Load
        filename = f'./toy/results/{method}-{mc_id}.npz'
        results = np.load(filename)
        samples, gp_mean, gp_cov = results['samples'], results['gp_mean'], results['gp_cov']

        # Compute errors
        approx_mean = jnp.mean(samples, axis=0)
        approx_cov = jnp.cov(samples, rowvar=False)

        err_kl = kl(gp_mean, gp_cov, approx_mean, approx_cov)
        err_skew = jnp.mean(jnp.abs(scipy.stats.skew(samples, axis=0)))
        err_kurt = jnp.mean(jnp.abs(scipy.stats.kurtosis(samples, axis=0, fisher=True)))

        errs_kl[mc_id] = err_kl
        errs_skew[mc_id] = err_skew
        errs_kurt[mc_id] = err_kurt

        # Autocorrelation
        if method != 'filter':
            autocorrs[mc_id] = np.quantile(npr.diagnostics.autocorrelation(samples, axis=0)[:max_lags],
                                           q=q, axis=1)

    # Tabulate
    print(f'Method {method} | '
          f'KL | {jnp.mean(errs_kl):.4f} {jnp.std(errs_kl):.4f} | '
          f'Skew | {jnp.mean(errs_skew):.4f} {jnp.std(errs_skew):.4f} | '
          f'Kurt | {jnp.mean(errs_kurt):.4f} {jnp.std(errs_kurt):.4f}')

    # Autocorrelation
    plt.plot(jnp.arange(max_lags), jnp.mean(autocorrs, axis=0), label=method)

plt.legend()
plt.show()
