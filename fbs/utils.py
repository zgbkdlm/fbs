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


def bures_dist(m1, cov1, m2, cov2):
    """The Wasserstein distance between two Gaussians.
    """
    chol = jnp.linalg.cholesky(cov1)
    A = cov1 + cov2 - 2 * jnp.linalg.cholesky(chol @ cov2 @ chol)
    return jnp.sum((m1 - m2) ** 2) + jnp.trace(A)
