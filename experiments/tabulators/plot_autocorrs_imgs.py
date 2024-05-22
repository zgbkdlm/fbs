"""
Plot the autocorrelation of MNIST images.
This generates Figure 8.
"""
import numpy as np
import numpyro as npr
import matplotlib.pyplot as plt

nsamples = 100
max_mcs = 100
max_lags = 20
rnd_mask = False
sde = 'lin'


def ac_fn(samples_):
    acs_ = npr.diagnostics.autocorrelation(samples_, axis=0)[:max_lags]
    acs_[acs_ < 0] = 0.
    return acs_


plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axes = plt.subplots(ncols=4, sharey='row', figsize=(16, 4))

for col, dataset, task, title in [(0, 'mnist', 'inpainting-15', 'MNIST inpainting'),
                                  (1, 'mnist', 'supr-4', 'MNIST super-resolution'),
                                  (2, 'celeba-64', 'inpainting-32', 'CelebA inpainting'),
                                  (3, 'celeba-64', 'supr-2', 'CelebA super-resolution')]:

    methods = [f'gibbs-eb-ef-{100 if "mnist" in dataset else 10}',
               f'gibbs-eb-ef-{10 if "mnist" in dataset else 2}',
               f'pmcmc-0.005-{100 if "mnist" in dataset else 10}',
               f'pmcmc-0.005-{10 if "mnist" in dataset else 2}']
    method_labels = [f'Gibbs-CSMC-{100 if "mnist" in dataset else 10}',
                     f'Gibbs-CSMC-{10 if "mnist" in dataset else 2}',
                     f'PMCMC-0.005-{100 if "mnist" in dataset else 10}',
                     f'PMCMC-0.005-{10 if "mnist" in dataset else 2}']
    method_markers = ['*', 'o', '*', 'o']  # distinguish nparticles
    method_line_styles = ['-', '-', '--', '--']  # distinguish method

    autocorrs = np.zeros((max_mcs, max_lags))

    for method, label, marker, style in zip(methods, method_labels, method_markers, method_line_styles):
        nparticles = method.split('-')[-1]

        for mc_id in range(max_mcs):
            if 'inpainting' in task:
                path_head = f'./imgs/results_inpainting/arrs/{dataset}-{task.split("-")[-1]}-{sde}-{nparticles}-{mc_id}'
            else:
                path_head = f'./imgs/results_supr/arrs/{dataset}-{task.split("-")[-1]}{"-rm" if rnd_mask else ""}-{sde}-{nparticles}-{mc_id}'

            filename = path_head + f'-{"-".join(method.split("-")[:-1])}.npy'
            samples = np.load(filename).reshape(nsamples, -1)
            acs = np.nanmin(ac_fn(samples), axis=-1)
            autocorrs[mc_id] = acs

        axes[col].plot(np.arange(max_lags), np.mean(autocorrs, axis=0), c='black', linewidth=2,
                       marker=marker, markerfacecolor='none', markevery=max_lags // 5, markersize=10,
                       linestyle=style, label=label)

    axes[col].grid(linestyle='--', alpha=0.3, which='both')
    axes[col].set_xlabel('Lag')
    axes[col].set_title(title)
    # axes[col].set_yscale('log')

    if col == 0:
        axes[col].set_ylabel('Autocorrelation (best variable)')

    axes[col].legend()

plt.tight_layout(pad=0.1)
plt.savefig('figs/appendix-autocorrs-imgs.pdf', transparent=True)
plt.show()
