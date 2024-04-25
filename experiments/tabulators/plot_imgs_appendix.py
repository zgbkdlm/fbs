"""
Plot the MNIST and Celeba images in the appendix.

Layout being like:
            ===============(task)=========================
method |    corrupt img | true img | sample0 | sample1 | sample2 | ...
...
"""
import numpy as np
import matplotlib.pyplot as plt
from fbs.data.images import normalise
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

np.random.seed(666)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 14})


def to_img(img):
    img = np.asarray(normalise(img, method='clip'))
    return img[..., 0] if dataset == 'mnist' else img


dataset = 'celeba-64'
task = 'supr-2'
rnd_mask = False
sde = 'lin'
methods = ['filter', 'gibbs-eb-ef', 'pmcmc-0.005', 'twisted', 'csgm']
nparticles = 10
nsamples = 100
nexamples = 12
sample_inds = np.random.choice(np.arange(nsamples), nexamples, replace=False)

img_hw = (28, 28) if dataset == 'mnist' else (64, 64)

for y0_id in range(100):

    fig = plt.figure(figsize=(16, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(methods), nexamples + 2), axes_pad=0.)

    path_head = f'./imgs/results_{task.split("-")[0]}/imgs/{dataset}-{task.split("-")[1]}'
    path_head = path_head + '-rm' if 'supr' in task and rnd_mask else path_head
    path_head = path_head + f'-{sde}-'
    filename = path_head + f'{y0_id}-{"corrupt-lr" if "supr" in task else "corrupt"}.png'
    img_corrupt = np.asarray(Image.open(filename).resize(img_hw, resample=Image.Resampling.NEAREST))
    filename = path_head + f'{y0_id}-true.png'
    img_true = np.asarray(Image.open(filename))

    for row in range(len(methods)):
        for col in range(nexamples + 2):
            axes_idx = row * (nexamples + 2) + col
            if col == 0:
                grid[axes_idx].imshow(img_corrupt)
            elif col == 1:
                grid[axes_idx].imshow(img_true)
            else:
                method = methods[row]
                if method == 'csgm':
                    filename = path_head + f'{y0_id}-{method}-{sample_inds[col - 2]}.png'
                else:
                    filename = path_head + f'{nparticles}-{y0_id}-{method}-{sample_inds[col - 2]}.png'
                img_restored = np.asarray(Image.open(filename))
                grid[axes_idx].imshow(img_restored)

            if row == 0 and col == 0:
                grid[axes_idx].set_title('corrupt')
            elif row == 0 and col == 1:
                grid[axes_idx].set_title('true')
            elif row == 0 and col > 1:
                grid[axes_idx].set_title(f'sample {col - 2}')

            grid[axes_idx].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(f'./tmp_figs/appendix-imgs-{dataset}-{task}-{nparticles}-{y0_id}.png', transparent=True)
    plt.show()
