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
nparticles = 10

methods = [f'gibbs-eb-{sde}-{nparticles}',
           f'pmcmc-0.005-{sde}-{nparticles}']
method_labels = ['Gibbs-CSMC', 'pMCMC']
max_mcs = 100
max_lags = 100
q = 0.95


def autocorr_over_chains(chains):
    return jnp.mean(np.vectorize(npr.diagnostics.autocorrelation, signature='(m,n)->(m,n)')(chains), axis=0)


autocorrs = np.zeros((max_mcs, max_lags))

for method, method_label in zip(methods, method_labels):
    for mc_id in range(max_mcs):
        # Load
        filename = f'./toy/results/{method}-{mc_id}.npz'
        results = np.load(filename)
        samples = results['samples']

        acs = autocorr_over_chains(samples)[:max_lags, :]
        autocorrs[mc_id] = np.quantile(acs, q=q, axis=-1)

    autocorr_mean = jnp.mean(autocorrs, axis=0)
    autocorr_std = jnp.std(autocorrs, axis=0)
    ax.plot(jnp.arange(max_lags), autocorr_mean, c='black', label=method_label)
    ax.fill_between(jnp.arange(max_lags),
                    autocorr_mean - 1.96 * autocorr_std,
                    autocorr_mean + 1.96 * autocorr_std,
                    alpha=0.3, color='black', edgecolor='none')

ax.grid(linestyle='--', alpha=0.3, which='both')
plt.tight_layout(pad=0.1)
plt.legend()
plt.show()
