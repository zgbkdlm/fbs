"""
Plot some contours of the Gaussian marginals.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

sde = 'const'

method_labels = ['CSGM',
                 'PF',
                 'Gibbs-CSMC',
                 'PMCMC-0.005',
                 'PMCMC-0.001',
                 'TPF']

mc_id = 66
which_marginals = jnp.array([60, 66])
ngrids = 200
level_lines = np.insert(np.linspace(0.15, 0.90, 6), 0, 0.05)


@partial(jax.vmap, in_axes=[0, None, None])
@partial(jax.vmap, in_axes=[0, None, None])
def pdf_mvn(x, m, cov):
    return jax.scipy.stats.multivariate_normal.pdf(x, m, cov)


plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 18})

fig, axes = plt.subplots(nrows=2, ncols=6, sharey='row', sharex='col', figsize=(20, 8))

for row, nparticles in enumerate([10, 100]):
    methods = [f'csgm-{sde}',
               f'filter-{sde}-{nparticles}',
               f'gibbs-eb-{sde}-{nparticles}',
               f'pmcmc-0.005-{sde}-{nparticles}',
               f'pmcmc-0.001-{sde}-{nparticles}',
               f'twisted-{sde}-{nparticles}']

    for col, method, label in zip(list(range(len(methods))), methods, method_labels):

        # Load
        filename = f'./toy/results/{method}-{mc_id}.npz'
        results = np.load(filename)
        samples, gp_mean, gp_cov = results['samples'], results['gp_mean'], results['gp_cov']

        if any([s in method for s in ['gibbs', 'pmcmc']]):
            approx_mean = jnp.mean(jax.vmap(partial(jnp.mean, axis=0))(samples), axis=0)
            approx_cov = jnp.mean(jax.vmap(partial(jnp.cov, rowvar=False))(samples), axis=0)

        else:
            approx_mean = np.mean(samples, axis=0)
            approx_cov = np.cov(samples, rowvar=False)

        gp_mean = gp_mean[which_marginals]
        gp_cov = jnp.array(
            [[gp_cov[which_marginals[0], which_marginals[0]], gp_cov[which_marginals[0], which_marginals[1]]],
             [gp_cov[which_marginals[1], which_marginals[0]], gp_cov[which_marginals[1], which_marginals[1]]]])

        approx_mean = approx_mean[which_marginals]
        approx_cov = jnp.array(
            [[approx_cov[which_marginals[0], which_marginals[0]], approx_cov[which_marginals[0], which_marginals[1]]],
             [approx_cov[which_marginals[1], which_marginals[0]], approx_cov[which_marginals[1], which_marginals[1]]]])

        # Plot the PDF contours
        grid_x1 = np.linspace(gp_mean[0] - 3 * np.sqrt(gp_cov[0, 0]), gp_mean[0] + 3 * np.sqrt(gp_cov[0, 0]), ngrids)
        grid_x2 = np.linspace(gp_mean[1] - 3 * np.sqrt(gp_cov[1, 1]), gp_mean[1] + 3 * np.sqrt(gp_cov[1, 1]), ngrids)
        meshes_plot = np.meshgrid(grid_x1, grid_x2)
        meshes = np.dstack(meshes_plot)

        pdfs_gp = pdf_mvn(meshes, gp_mean, gp_cov)
        pdfs_approx = pdf_mvn(meshes, approx_mean, approx_cov)

        con = axes[row, col].contour(*meshes_plot, pdfs_gp, levels=level_lines,
                                     linewidths=2, cmap=plt.cm.binary)
        axes[row, col].contour(*meshes_plot, pdfs_approx, levels=level_lines,
                               linewidths=2, linestyles='dashed', cmap=plt.cm.binary)

        if row == 1:
            axes[row, col].set_xlabel('$x_1$')

        if row == 0:
            axes[row, col].set_title(label)

        if col == 0:
            axes[row, col].set_ylabel('$x_2$')
            axes[row, col].clabel(con)

            if row == 0:
                # Create ghost line for legend
                axes[row, col].plot([], [], c='black', linewidth=2, alpha=0.5, label='Truth')
                axes[row, col].plot([], [], c='black', linewidth=2, linestyle='--', alpha=0.5, label='Approximation')
                axes[row, col].legend()

        axes[row, col].grid(linestyle='--', alpha=0.3, which='both')

plt.tight_layout(pad=0.1)
plt.savefig(f'figs/toy-contours-{mc_id}.pdf', transparent=True)
plt.show()
