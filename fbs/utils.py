import jax
import jax.numpy as jnp
from fbs.typings import JArray, FloatScalar
from typing import Tuple


def discretise_lti_sde(A: JArray, gamma: JArray, dt: FloatScalar) -> Tuple[JArray, JArray]:
    """dX(t) = A X(t) dt + B dW(t),
    where gamma = B @ B.T
    """
    d = A.shape[0]

    F = jax.scipy.linalg.expm(A * dt)
    phi = jnp.vstack([jnp.hstack([A, gamma]), jnp.hstack([jnp.zeros_like(A), -A.T])])
    AB = jax.scipy.linalg.expm(phi * dt) @ jnp.vstack([jnp.zeros_like(A), jnp.eye(d)])
    cov = AB[0:d, :] @ F.T
    return F, cov


def bures_dist(m0, cov0, m1, cov1):
    """The Wasserstein distance between two Gaussians.
    """
    chol = jnp.linalg.cholesky(cov0)
    A = cov0 + cov1 - 2 * jnp.linalg.cholesky(chol @ cov1 @ chol)
    return jnp.sum((m0 - m1) ** 2) + jnp.trace(A)


def kl(m0, cov0, m1, cov1):
    d = m0.shape[-1]
    chol = jax.scipy.linalg.cho_factor(cov1)
    return (jnp.trace(jax.scipy.linalg.cho_solve(chol, cov0))
            - d + jnp.dot(m1 - m0, jax.scipy.linalg.cho_solve(chol, m1 - m0))
            + jnp.log(jnp.linalg.det(cov1) / jnp.linalg.det(cov0)))
