"""Tabulate errors of SB imges with Gibbs-MH
"""
import jax
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from fbs.data.images import normalise

jax.config.update("jax_enable_x64", True)


def to_img(img):
    img = normalise(img, method='clip')
    return img[..., 0] if dataset == 'mnist' else img


dataset = 'mnist'
task = 'supr-4'
rnd_mask = False
sde = 'lin'
nparticles = 10
ny0s = 100
nsamples = 100
x0_init = 'interp'
use_mh = True

methods = [f'gibbs-eb-ef-{x0_init}{"-mh" if use_mh else ""}']

ssims = np.zeros((ny0s, nsamples))
psnrs = np.zeros((ny0s, nsamples))
mh_accs_all = np.zeros((ny0s, nsamples))

for method in methods:
    path_head = f'./sb_imgs_mh/arrs/{dataset}-{task.split("-")[1]}'
    path_head = path_head + '-rm' if 'supr' in task and rnd_mask else path_head
    path_head = path_head + f'-{sde}-{nparticles}-'

    for i in range(ny0s):
        true_img = np.asarray(to_img(np.load(path_head + f'{i}-true.npz')['test_img']))
        filename = path_head + f'{i}-{method}.npz'
        result_data = np.load(filename)
        restored_imgs = np.asarray(jax.vmap(to_img)(result_data['restored_imgs']))
        mh_accs = result_data['mh_acc']
        mh_accs_all[i, :] = mh_accs

        for k in range(nsamples):
            psnr = peak_signal_noise_ratio(true_img, restored_imgs[k], data_range=1)
            ssim = structural_similarity(true_img, restored_imgs[k], data_range=1,
                                         channel_axis=None if dataset == 'mnist' else -1)
            psnrs[i, k] = psnr
            ssims[i, k] = ssim

    print(
        f'{method} | PSNR: {np.mean(psnrs):.4f} {np.std(psnrs):.4f} | SSIM: {np.mean(ssims):.4f} {np.std(ssims):.4f} | '
        f'ACC: {np.mean(mh_accs_all):.2f} {np.std(mh_accs_all):.2f}')
