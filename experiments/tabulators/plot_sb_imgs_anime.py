"""
Plot an animation of the super-resolution images under a Schrodinger bridge model.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fbs.data.images import normalise
from fbs.data import MNISTRestore
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
from PIL import Image

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 17})


def to_img(img):
    img = np.asarray(normalise(img, method='clip'))
    return img[..., 0] if dataset_name == 'mnist' else img


# Load data
key = jax.random.PRNGKey(666)
key, data_key = jax.random.split(key)
key, subkey = jax.random.split(key)
dataset_name = 'mnist'
dataset = MNISTRestore(subkey, '../datasets/mnist.npz', task=f'supr-4', test=True)
dataset.sr_random = False
x_shape = dataset.unobs_shape

sde = 'lin'
nparticles = 100
max_nsamples = 30
y0_id = 9

path_head_arr = f'./sb_imgs/results/{dataset_name}-4-{sde}-{nparticles}-{y0_id}'

true_img = to_img(np.load(path_head_arr + '-true.npz')['test_img'])
corrupt_img = np.asarray(Image.open(path_head_arr + '-corrupt-lr.png').resize((28, 28),
                                                                              resample=Image.Resampling.NEAREST))
x0_type = 'blank'
filter_random_imgs = to_img(np.load(path_head_arr + f'-filter-{x0_type}.npy'))
gibbs_random_imgs = to_img(np.load(path_head_arr + f'-gibbs-eb-ef-{x0_type}.npy'))
gibbs_random_imgs = jnp.concatenate([to_img(np.load(path_head_arr + '-gibbs-init.npy'))[None, :, :],
                                     gibbs_random_imgs], axis=0)

data_key, subkey = jax.random.split(data_key)
for _ in range(y0_id):
    data_key, subkey = jax.random.split(data_key)
test_img, test_y0, mask = dataset.sampler(subkey)

key, subkey = jax.random.split(key)
if x0_type == 'random':
    x0_init = jax.random.uniform(subkey, x_shape, minval=0., maxval=1.)
elif x0_type == 'blank':
    x0_init = jnp.zeros(x_shape)
else:
    raise ValueError('')
init_img = dataset.concat(x0_init, test_y0, mask)

# Plot four examples to show in the paper main body
delay_frames = 5
fps = 1

fig = plt.figure(figsize=(11, 3))
axes = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.)

axes[0].imshow(corrupt_img, cmap='gray')
axes[0].set_title('Input')
axes[0].axis('off')

axes[1].imshow(true_img, cmap='gray')
axes[1].set_title('Truth')
axes[1].axis('off')

im_pf = axes[2].imshow(init_img, cmap='gray')
axes[2].set_title('PF sample 0')
axes[2].axis('off')

im_gibbs = axes[3].imshow(init_img, cmap='gray')
axes[3].set_title('Gibbs chain sample 0')
axes[3].axis('off')


def anime_init():
    return im_pf, im_gibbs


def update(frame):
    if frame < delay_frames:
        return im_pf, im_gibbs
    im_pf.set_data(filter_random_imgs[frame - delay_frames])
    im_gibbs.set_data(gibbs_random_imgs[frame - delay_frames])
    axes[2].set_title(f'PF sample {frame - delay_frames}')
    axes[3].set_title(f'Gibbs chain sample {frame - delay_frames}')
    return im_pf, im_gibbs


ani = FuncAnimation(fig, update, frames=np.arange(max_nsamples + delay_frames), interval=1000 // fps,
                    init_func=anime_init, blit=False)
plt.tight_layout(pad=0.1)
plt.subplots_adjust(top=0.905)
ani.save(f'./figs/sb-imgs-anime-{y0_id}.gif', fps=fps)

plt.show()
