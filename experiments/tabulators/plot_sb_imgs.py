import numpy as np
import matplotlib.pyplot as plt
from fbs.data.images import normalise
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 17})


def to_img(img):
    img = np.asarray(normalise(img, method='clip'))
    return img[..., 0] if dataset == 'mnist' else img


# Load data
dataset = 'mnist'
sde = 'lin'
methods = ['filter', 'gibbs-eb-ef']
nparticles = 100
max_mcs = 100
y0_id = 10

path_head_arr = f'./sb_imgs/results/{dataset}-4-{sde}-{nparticles}-{y0_id}'

true_img = to_img(np.load(path_head_arr + '-true.npz')['test_img'])
corrupt_img = np.asarray(Image.open(path_head_arr + '-corrupt-lr.png').resize((28, 28),
                                                                              resample=Image.Resampling.NEAREST))
filter_random_imgs = to_img(np.load(path_head_arr + '-filter-random.npy'))
filter_blank_imgs = to_img(np.load(path_head_arr + '-filter-blank.npy'))
filter_interp_imgs = to_img(np.load(path_head_arr + '-filter-interp.npy'))
gibbs_random_imgs = to_img(np.load(path_head_arr + '-gibbs-eb-ef-random.npy'))
gibbs_blank_imgs = to_img(np.load(path_head_arr + '-gibbs-eb-ef-blank.npy'))
gibbs_interp_imgs = to_img(np.load(path_head_arr + '-gibbs-eb-ef-interp.npy'))
all_restored_imgs = [filter_random_imgs, filter_blank_imgs, filter_interp_imgs, gibbs_blank_imgs]

# Plot four examples to show in the paper main body
fig = plt.figure(figsize=(8, 9))
axes = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.4)

axes[0].imshow(corrupt_img, cmap='gray')
axes[0].set_title('Input')
axes[0].axis('off')

axes[1].imshow(filter_random_imgs[0], cmap='gray')
axes[1].set_title('PF (random $X_0$)')
axes[1].axis('off')

axes[2].imshow(filter_interp_imgs[1], cmap='gray')
axes[2].set_title('PF (interpolation $X_0$)')
axes[2].axis('off')

axes[3].imshow(gibbs_blank_imgs[50], cmap='gray')
axes[3].set_title('Gibbs-CSMC')
axes[3].axis('off')

plt.tight_layout(pad=0.1)
plt.savefig(f'figs/sb-imgs-examples-{y0_id}.pdf', transparent=True)
plt.show()

# Plot more in the appendix
fig = plt.figure(figsize=(24, 8))
nexamples = 10
per = max_mcs // nexamples
axes = ImageGrid(fig, 111, nrows_ncols=(4, nexamples + 2), axes_pad=0.)

for row in range(len(all_restored_imgs)):
    for col in range(nexamples + 2):
        axes_idx = row * (nexamples + 2) + col
        if col == 0:
            axes[axes_idx].imshow(corrupt_img, cmap='gray')
        elif col == 1:
            axes[axes_idx].imshow(true_img, cmap='gray')
        else:
            restored_imgs = all_restored_imgs[row]
            axes[axes_idx].imshow(restored_imgs[(col - 2) * per], cmap='gray')

        if row == 0 and col == 0:
            axes[axes_idx].set_title('corrupt')
        elif row == 0 and col == 1:
            axes[axes_idx].set_title('true')
        elif row == 0 and col > 1:
            axes[axes_idx].set_title(f'sample {col - 2}')

        # if col == 0 and row == 0:
        #     axes[axes_idx].set_ylabel('PF (random)')
        # elif col == 1 and row == 0:
        #     axes[axes_idx].set_ylabel('PF (zeros)')
        # elif col == 2 and row == 0:
        #     axes[axes_idx].set_ylabel('PF (interpolation)')
        # elif col == 3 and row == 0:
        #     axes[axes_idx].set_ylabel('Gibbs-CSMC')

        axes[axes_idx].axis('off')

plt.tight_layout(pad=0.1)
plt.savefig(f'figs/sb-imgs-appendix-{y0_id}.pdf', transparent=True)
plt.show()
