import math
import jax.numpy as jnp
import jax.random
from fbs.data.base import DataSet
from fbs.typings import JArray, JKey
from typing import Tuple


class Crescent(DataSet):
    r"""
    X ~ N(m, cov),
    Y | X ~ N(X_1 / \psi + 0.5 \, (X_0 ** 2 + \psi_0 ** 2), 1.)
    """

    def __init__(self, n: int = 10, psi: float = 1.):
        self.n = n
        self.psi = psi
        self.m = jnp.array([0., 0.])
        self.cov = jnp.array([[2., 0.],
                              [0., 1.]])
        self.cov_is_diag = True

    def sampler(self, key: JKey, batch_size: int) -> Tuple[JArray, JArray]:
        key, subkey = jax.random.split(key)
        xs = self.m + jax.random.normal(subkey, (batch_size, 2)) @ jnp.linalg.cholesky(self.cov)

        key, subkey = jax.random.split(key)
        ys = jax.vmap(self.emission, in_axes=[0, None])(xs, self.psi) + 1. * jax.random.normal(subkey, (batch_size,))
        return xs, ys

    @staticmethod
    def emission(x, psi):
        return x[1] / psi + 0.5 * (x[0] ** 2 + psi ** 2)

    def log_prior_pdf(self, phi):
        if self.cov_is_diag:
            return jnp.sum(jax.scipy.stats.norm.logpdf(phi, self.m, jnp.diag(self.cov)))
        else:
            return jax.scipy.stats.multivariate_normal.logpdf(phi, self.m, self.cov)

    def log_cond_pdf_likelihood(self, y, phi):
        return jnp.sum(jax.scipy.stats.norm.logpdf(y, self.emission(phi, self.psi), 1.))

    def posterior(self, phi_mesh: JArray, y: JArray):
        def energy(phi): return jnp.exp(self.log_prior_pdf(phi) + self.log_cond_pdf_likelihood(y, phi))

        evals = jax.vmap(jax.vmap(energy, in_axes=[0]), in_axes=[0])(phi_mesh)
        z = jax.scipy.integrate.trapezoid(jax.scipy.integrate.trapezoid(evals, phi_mesh[0, :, 0], axis=0),
                                          phi_mesh[:, 0, 1])
        return evals / z
