"""
Plot the trace of the PMCMC chain.
"""
import jax
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axes = plt.subplots(ncols=3, sharey='row', figsize=(12, 3))

jax.config.update("jax_enable_x64", True)

sde = 'const'
nparticles = 10
mc_id = 0
max_mcs = 100
chain_id = 0
dim_id = 0
steps = np.arange(10000)
down_sampling_rate = 2

for i, delta in enumerate([0.01, 0.005, 0.001]):

    filename = f'./toy/results/pmcmc-{delta}-{sde}-{nparticles}-{mc_id}.npz'
    results = np.load(filename)
    samples = results['samples'][chain_id, :, dim_id][::down_sampling_rate]

    axes[i].plot(steps[::down_sampling_rate], samples, c='black', linewidth=1)
    axes[i].set_title(rf'$\delta={delta}$')

    axes[i].set_xlabel('PMCMC chain iteration')
    if i == 0:
        axes[i].set_ylabel('Marginal sample')
    axes[i].grid(linestyle='--', alpha=0.3, which='both')

plt.subplots_adjust(wspace=0.)
plt.tight_layout(pad=0.1)
plt.savefig('figs/pmcmc-trace.pdf', transparent=True)
plt.show()
