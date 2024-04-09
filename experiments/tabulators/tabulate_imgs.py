"""Tabulate errors of image restoration.

FID/LPIPS are not computed, as they are not compatible for the MCMC method for image restoration. See, e.g.,
https://arxiv.org/pdf/2403.11407.pdf, pp. 8.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from fbs.data.images import normalise


def to_img(img):
    img = normalise(img, method='clip')
    return img[..., 0] if dataset == 'mnist' else img


dataset = 'mnist'
task = 'supr-4'
ny0s = 10
nsamples = 100

path_head = f'./imgs/results_{task.split("-")[0]}/arrs/{dataset}-{task.split("-")[1]}-'

methods = [f'filter',
           f'gibbs-eb-ef',
           f'pmcmc-0.005',
           f'twisted',
           f'csgm']
q = 0.95

ssims = np.zeros((ny0s, nsamples))
psnrs = np.zeros((ny0s, nsamples))

for method in methods:
    for i in range(ny0s):
        true_img = np.load(path_head + f'{i}-true.npy')
        for k in range(nsamples):
            restored_img = to_img(np.load(path_head + f'{i}-{method}-{k}.npy'))
            psnr = peak_signal_noise_ratio(true_img, restored_img, data_range=1)
            ssim, *_ = structural_similarity(true_img, restored_img, data_range=1,
                                             channel_axis=None if dataset == 'mnist' else -1)
            psnrs[i, k] = psnr
            ssims[i, k] = ssim
    print(f'{method} | PSNR: {np.mean(psnrs):.4f} | SSIM: {np.mean(ssims):.4f}')
