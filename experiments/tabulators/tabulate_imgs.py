"""Tabulate errors of image restoration.

FID is not computed, as it is not compatible for the MCMC method for image restoration. See, e.g.,
https://arxiv.org/pdf/2403.11407.pdf, pp. 8.
"""
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from fbs.data.images import normalise


def to_img(img):
    img = np.asarray(normalise(img, method='clip'))
    return img[..., 0] if dataset == 'mnist' else img


def to_torch_tensor(img):
    """img: (h, w, c) to torch tensor (1, c, h, w), and scale to [-1, 1].
    """
    if dataset == 'mnist':
        raise AssertionError('MNIST is not compatable with LPIPS')
    img = np.expand_dims(np.swapaxes(img, -1, 0), 0) * 2 - 1
    return torch.Tensor(img)


dataset = 'celeba-64'
task = 'supr-2'
rnd_mask = False
sde = 'lin'
nparticles = 10
ny0s = 10
nsamples = 100

methods = [f'filter',
           f'gibbs-eb-ef',
           f'pmcmc-0.005',
           f'twisted',
           f'csgm']
q = 0.95

ssims = np.zeros((ny0s, nsamples))
psnrs = np.zeros((ny0s, nsamples))
lpipss = np.zeros((ny0s, nsamples))

loss_fn = lpips.LPIPS(net='alex')

for method in methods:
    path_head = f'./imgs/results_{task.split("-")[0]}/arrs/{dataset}-{task.split("-")[1]}'
    path_head = path_head + '-rm' if 'supr' in task and rnd_mask else path_head
    path_head = path_head + f'-{sde}-'

    for i in range(ny0s):
        true_img = to_img(np.load(path_head + f'{i}-true.npz')['test_img'])
        for k in range(nsamples):
            if 'csgm' in method:
                filename = path_head + f'{i}-{method}-{k}.npy'
            else:
                filename = path_head + f'{nparticles}-{i}-{method}-{k}.npy'
            restored_img = to_img(np.load(filename))
            psnr = peak_signal_noise_ratio(true_img, restored_img, data_range=1)
            ssim = structural_similarity(true_img, restored_img, data_range=1,
                                         channel_axis=None if dataset == 'mnist' else -1)
            psnrs[i, k] = psnr
            ssims[i, k] = ssim

            if dataset != 'mnist':
                # Compute LPIPS (not for MNIST, as it is not compatible with LPIPS)
                tensor0 = to_torch_tensor(true_img)
                tensor1 = to_torch_tensor(restored_img)
                lpipss[i, k] = loss_fn.forward(tensor0, tensor1)

    print(f'{method} | PSNR: {np.mean(psnrs):.4f} {np.std(psnrs):.4f} | SSIM: {np.mean(ssims):.4f} {np.std(ssims):.4f} '
          f'| LPIPS {np.mean(lpipss):.4f} {np.std(lpipss):.4f}')
