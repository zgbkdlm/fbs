import numpy as np
import jax
import matplotlib.pyplot as plt
from fbs.utils import kl, bures_dist

jax.config.update("jax_enable_x64", True)

measure = 'bures'

if measure == 'kl':
    measure_fn = jax.jit(kl)
else:
    measure_fn = jax.jit(bures_dist)

methods = ['filter-proper',
           'filter-heuristic',
           'gibbs-eb']
nparticles_used = [2, 4, 8, 16, 32, 64]
max_mcs = 100
q = 0.95

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

            err_bures = measure_fn(gp_mean, gp_cov, approx_mean, approx_cov)

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
        c='black', linewidth=2, marker='o', markerfacecolor='none', markersize=10,
        alpha=0.5, label='PF (ideal)')
ax.fill_between(nparticles_used,
                np.quantile(errs_filter_proper, q=0.05, axis=0),
                np.quantile(errs_filter_proper, q=0.95, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.plot(nparticles_used, np.mean(errs_filter_heuristic, axis=0),
        c='black', linewidth=2, linestyle='--', marker='x', markerfacecolor='none', markersize=10,
        alpha=0.5, label='PF (approximate)')
ax.fill_between(nparticles_used,
                np.quantile(errs_filter_heuristic, q=0.05, axis=0),
                np.quantile(errs_filter_heuristic, q=0.95, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.plot(nparticles_used, np.mean(errs_filter_gibbs, axis=0),
        c='black', linewidth=2, marker='*', markerfacecolor='none', markersize=12,
        alpha=0.5, label='Gibbs-CSMC')
ax.fill_between(nparticles_used,
                np.quantile(errs_filter_gibbs, q=0.05, axis=0),
                np.quantile(errs_filter_gibbs, q=0.95, axis=0),
                alpha=0.2, color='black', edgecolor='none')

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_xlabel('Number of particles')
ax.set_ylabel(f'{"KL divergence" if measure == "kl" else "Wasserstein--Bures distance"}')
plt.tight_layout(pad=0.1)
plt.legend()
plt.savefig('figs/gaussian-sb.pdf', transparent=True)
plt.show()
