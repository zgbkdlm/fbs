import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from fbs.utils import kl, bures_dist

jax.config.update("jax_enable_x64", True)
bures_dist = jax.jit(bures_dist)

methods = ['filter-proper',
           'filter-heuristic',
           'gibbs-eb']
nparticles_used = [4, 8, 16, 32, 64]
max_mcs = 100

errs_filter_proper = np.zeros((max_mcs, len(nparticles_used),))
errs_filter_heuristic = np.zeros((max_mcs, len(nparticles_used),))
errs_filter_gibbs = np.zeros((max_mcs, len(nparticles_used),))

for method in methods:
    for mc_id in range(max_mcs):
        for p, nparticles in enumerate(nparticles_used):
            # Load
            filename = f'./sb/results/{method}-{nparticles}-{mc_id}.npz'
            results = np.load(filename)
            samples, gp_mean, gp_cov = results['samples'], results['gp_mean'], results['gp_cov']

            approx_mean = np.mean(samples, axis=0)
            approx_cov = np.cov(samples, rowvar=False)

            err_bures = bures_dist(gp_mean, gp_cov, approx_mean, approx_cov)

            if method == 'filter-proper':
                errs_filter_proper[mc_id, p] = err_bures
            elif method == 'filter-heuristic':
                errs_filter_heuristic[mc_id, p] = err_bures
            elif method == 'gibbs-eb':
                errs_filter_gibbs[mc_id, p] = err_bures
            else:
                raise ValueError(f'Invalid method "{method}"')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, ax = plt.subplots()

ax.plot(nparticles_used, np.mean(errs_filter_proper, axis=0),
        c='black', linewidth=2, alpha=0.5, label='PF (proper)')
ax.fill_between(nparticles_used,
                np.mean(errs_filter_proper, axis=0) - 1.96 * np.std(errs_filter_proper, axis=0),
                np.mean(errs_filter_proper, axis=0) + 1.96 * np.std(errs_filter_proper, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.plot(nparticles_used, np.mean(errs_filter_heuristic, axis=0),
        c='black', linewidth=2, linestyle='--', alpha=0.5, label='PF (heuristic)')
ax.fill_between(nparticles_used,
                np.mean(errs_filter_heuristic, axis=0) - 1.96 * np.std(errs_filter_heuristic, axis=0),
                np.mean(errs_filter_heuristic, axis=0) + 1.96 * np.std(errs_filter_heuristic, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.plot(nparticles_used, np.mean(errs_filter_gibbs, axis=0),
        c='black', linewidth=2, alpha=0.5, label='Gibbs-CSMC')
ax.fill_between(nparticles_used,
                np.mean(errs_filter_gibbs, axis=0) - 1.96 * np.std(errs_filter_gibbs, axis=0),
                np.mean(errs_filter_gibbs, axis=0) + 1.96 * np.std(errs_filter_gibbs, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.set_scale('log')
ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_xlabel('Number of particles')
ax.set_ylabel('Wasserstein--Bures distance')
plt.tight_layout(pad=0.1)
plt.legend()
plt.show()
