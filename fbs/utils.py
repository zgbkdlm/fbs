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


def _log_det(chol):
    return 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))


def kl(m0, cov0, m1, cov1):
    d = m0.shape[-1]
    chol0 = jax.scipy.linalg.cho_factor(cov0)
    chol1 = jax.scipy.linalg.cho_factor(cov1)
    log_det0 = _log_det(chol0[0])
    log_det1 = _log_det(chol1[0])
    return (jnp.trace(jax.scipy.linalg.cho_solve(chol1, cov0))
            - d + jnp.dot(m1 - m0, jax.scipy.linalg.cho_solve(chol1, m1 - m0))
            + log_det1 - log_det0)
