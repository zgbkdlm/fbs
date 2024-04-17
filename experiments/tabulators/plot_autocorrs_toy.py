import jax
import jax.numpy as jnp
import numpy as np
import numpyro as npr
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, ax = plt.subplots()

jax.config.update("jax_enable_x64", True)

sde = 'const'
nparticles = 100

methods = [f'gibbs-eb-{sde}-10',
           f'gibbs-eb-{sde}-100',
           f'pmcmc-0.005-{sde}-100',
           f'pmcmc-0.005-{sde}-10',
           f'pmcmc-0.01-{sde}-10',
           f'pmcmc-0.001-{sde}-10',]
method_labels = ['Gibbs-CSMC-10',
                 'Gibbs-CSMC-100',
                 'PMCMC-0.005-100',
                 'PMCMC-0.005-10',
                 'PMCMC-0.01-10',
                 'PMCMC-0.001-10']
method_markers = ['*', '*', 'o', 'o', 'o', 'o']  # distinguish methods
method_line_styles = ['--', '-', '-', '--', '--', '--']  # distinguish nparticles
method_line_alphas = [1., 1., 0.5, 0.5, 0.1, 1.]  # distinguish deltas
max_mcs = 100
max_lags = 100
q = 0.95


def autocorr_over_chains(chains):
    return jnp.mean(np.vectorize(npr.diagnostics.autocorrelation, signature='(m,n)->(m,n)')(chains), axis=0)


autocorrs = np.zeros((max_mcs, max_lags))

for method, label, marker, style, alpha in zip(methods, method_labels, method_markers, method_line_styles,
                                               method_line_alphas):
    for mc_id in range(max_mcs):
        # Load
        filename = f'./toy/results/{method}-{mc_id}.npz'
        results = np.load(filename)
        samples = results['samples']

        acs = autocorr_over_chains(samples)[:max_lags, :]
        autocorrs[mc_id] = np.quantile(acs, q=q, axis=-1)

    autocorr_mean = jnp.mean(autocorrs, axis=0)
    autocorr_std = jnp.std(autocorrs, axis=0)
    ax.plot(jnp.arange(max_lags), autocorr_mean, c='black', linewidth=2,
            marker=marker, markerfacecolor='none', markevery=20, markersize=10,
            linestyle=style, alpha=alpha, label=label)
    # ax.fill_between(jnp.arange(max_lags),
    #                 autocorr_mean - 1.96 * autocorr_std,
    #                 autocorr_mean + 1.96 * autocorr_std,
    #                 alpha=0.1, color='black', edgecolor='none')

ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
plt.tight_layout(pad=0.1)
plt.legend()
plt.savefig('figs/autocorrs.pdf', transparent=True)
plt.show()
