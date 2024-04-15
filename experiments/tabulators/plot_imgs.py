import numpy as np
import matplotlib.pyplot as plt
from fbs.data.images import normalise
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image


def to_img(img):
    img = np.asarray(normalise(img, method='clip'))
    return img[..., 0] if dataset == 'mnist' else img


dataset = 'celeba-64'
task = 'supr-2'
rnd_mask = False
sde = 'lin'
nparticles = 10
ny0s = 20
nsamples = 100

img_hw = (28, 28) if dataset == 'mnist' else (64, 64)

for method in ['filter', 'gibbs-eb-ef', 'pmcmc-0.005', 'twisted', 'csgm']:
    np.random.seed(666)
    path_head = f'./imgs/results_{task.split("-")[0]}/imgs/{dataset}-{task.split("-")[1]}'
    path_head = path_head + '-rm' if 'supr' in task and rnd_mask else path_head
    path_head = path_head + f'-{sde}-'

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.)
    y0_inds = np.random.choice(np.arange(ny0s), 4, replace=False)

    for row in range(4):
        filename = path_head + f'{y0_inds[row]}-{"corrupt-lr" if "supr" in task else "corrupt"}.png'
        img_corrupt = np.asarray(Image.open(filename).resize(img_hw, resample=Image.Resampling.NEAREST))
        for col in range(4):
            idx = row * 4 + col
            if col == 0:
                grid[idx].imshow(img_corrupt)
            else:
                if method == 'csgm':
                    filename = path_head + f'{y0_inds[row]}-{method}-{col * 30}.png'
                else:
                    filename = path_head + f'{nparticles}-{y0_inds[row]}-{method}-{col * 30}.png'
                img_restored = np.asarray(Image.open(filename))
                grid[idx].imshow(img_restored)

            grid[idx].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(f'./tmp_figs/imgs-{dataset}-{task}-{method}-{nparticles}.png', transparent=True)
    plt.show()
