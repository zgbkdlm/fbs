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

dataset = 'mnist'
task = 'supr-4'
rnd_mask = False
sde = 'lin'
delta = 0.005
nparticles = 10
ny0s = 10
max_mcs = 100
max_lags = 10

methods = [f'gibbs-eb-{sde}-{nparticles}',
           f'pmcmc-{delta}-{sde}-{nparticles}']
method_labels = ['Gibbs-CSMC', f'pMCMC-{delta}']
q = 0.05

img_size = 28 * 28 if 'mnist' in dataset else 64 * 64 * 3
restored_imgs = np.zeros((max_mcs, img_size))
i = 1

for method, method_label in zip(methods, method_labels):
    path_head = f'./imgs/results_{task.split("-")[0]}/arrs/{dataset}-{task.split("-")[1]}'
    path_head = path_head + '-rm' if 'supr' in task and rnd_mask else path_head
    path_head = path_head + f'-{sde}-{nparticles}-'
    for k in range(max_mcs):
        restored_imgs[k] = np.load(path_head + f'{i}-{method}-{k}.npy').reshape((img_size, ))

    acs = npr.diagnostics.autocorrelation(restored_imgs, axis=0)[:max_lags, :]
    autocorrs = np.quantile(acs, q=q, axis=-1)

    autocorr_mean = jnp.mean(autocorrs, axis=0)
    autocorr_std = jnp.std(autocorrs, axis=0)
    ax.plot(jnp.arange(max_lags), autocorr_mean, c='black', label=method_label)
    ax.fill_between(jnp.arange(max_lags),
                    autocorr_mean - 1.96 * autocorr_std,
                    autocorr_mean + 1.96 * autocorr_std,
                    alpha=0.1, color='black', edgecolor='none')

ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
plt.tight_layout(pad=0.1)
plt.legend()
plt.savefig('figs/autocorrs.pdf', transparent=True)
plt.show()
