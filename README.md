# Conditioning diffusions via explicit forward-backward bridging
[![UnitTest](https://github.com/zgbkdlm/fbs/actions/workflows/unittest.yml/badge.svg)](https://github.com/zgbkdlm/fbs/actions/workflows/unittest.yml)

This repository is concerned with Markov chain Monte Carlo (MCMC) method for conditional sampling with generative diffusions, 
see, https://arxiv.org/placeholder.
More specifically, our core contributions are as follows. 

1. We develop new and efficient (particle Gibbs and particle pseudo-marginal) MCMC samplers for conditional sampling in diffusion models. 
2. Our proposed method is not only consistent but is also asymptotically exact, even when 1) using a finite number of particles, and 2) no access to the reference distribution. We show the performance on synthetic and real datasets.
3. Our method is also applicable to Schrödinger bridges, and hence the merits (e.g., low time steps) of SBs are automatically inherited here.

To quickly see what our method can do while others cannot, please check the two animations below 
(you may wait for seconds for the animations to start). 

<img src="./docs/sb-imgs-anime-2.gif" style="width: 80%; height: auto; display: block; margin-left: auto; margin-right: auto">
<img src="./docs/sb-imgs-anime-9.gif" style="width: 80%; height: auto; display: block; margin-left: auto; margin-right: auto">

In the two animations above, we see that our Gibbs sampler progressively burns into the target distribution in a few iterations, and then become stationary, 
while the peer method (i.e., a standard particle filter approach) can give unrealistic results due to its 
inherent statistical biases (plural!).

# Install
1. `git clone git@github.com:zgbkdlm/fbs.git`
2. `cd fbs`
3. `python venv ./venv && source venv/bin/activate` Please not use your base environment, as it may corrupt your existing package versions.
4. Install JAX in GPU/CPU environment according to this official guidance https://github.com/google/jax?tab=readme-ov-file#installation.
5. `pip install -r requirements.txt`
6. `pip install -e .`

# How to reproduce the results
All the experiments-related scripts are in `./experiments`. 
Originally, our experiments are done in a Slurm-based server (i.e., Berzlius i Linköpings universitet), and hence you 
may need to adapt the bash files in the folder to your local environment.

The scripts in `./experiments` are explained as follows.

1. `./experiments/bashes`. This folder contains the bash files that are submitted to the server for running the experiments. You can find the exact parameters that we use.
2. `./experiments/checkpoints`. This folder contains the trained models. You can download them from https://huggingface.co/zgbkdlm/fbs.
3. `./experiments/datasets`. This folder contains the MNIST and CelebA-HQ datasets. Please see its README.md for details.
4. `./experiments/imgs`. This folder contains scripts for inpainting and super-resolution in MNIST and CelebA.
5. `./experiments/sb`. This folder is concerned with the Gaussian Schrödinger bridge experiment.
6. `./experiments/sb_imgs`. This folder is concerned with the Schrödinger bridge experiments on MNIST super-resolution.
7. `./experiments/tabulators`. This folder contains the scripts for producing the tables and figures in our paper.
8. `./experiments/toy`. This folder is concerned with the Gaussian synthetic experiments.

The trained models are available at https://huggingface.co/zgbkdlm/fbs. 
Download them and copy to the folder `./experiments/checkpoints`. 
If you cannot download them, run the training scripts in `./experiments`, and you should get the exact models as we have. 

After you have run all the experiments, results will be saved in their corresponding directories. 
Then, simply run any file in `./experiments/tabulators` to produce the tables and figures in our paper.

# Citation
Please cite our paper as follows. 

```bibtex
@article{corenflos2024FBS,
    title={Conditioning diffusion models by explicit forward-backward bridging},
    author={Corenflos, Adrien and Zhao, Zheng and S\"{a}rkk\"{a}, Simo and Sj\"{o}lund, Jens and Sch\"{o}n, Thomas B.},
    journal={arXiv preprint xxx},
    year={2024}
}
```

# License
The Apache License 2.0.

# Contact
Zheng Zhao (https://zz.zabemon.com) and Adrien Corenflos (https://adriencorenflos.github.io/). 
