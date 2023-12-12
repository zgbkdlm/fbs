from typing import Optional

import jax
import jax.numpy as jnp

Array = jax.typing.ArrayLike
PRNGKey = jax.typing.ArrayLike


def multinomial(key: PRNGKey, weights: Array, i: Optional[int] = 0, j: Optional[int] = 0,
                conditional: bool = True) -> Array:
    """
    Multinomial resampling. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
        Only used if conditional is True.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling
    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    N = weights.shape[0]
    indices = jax.random.choice(key, N, p=weights, shape=(N,), replace=True)
    if conditional:
        indices = indices.at[j].set(i)
    return indices


def killing(key: PRNGKey, weights: Array, i: Optional[int] = 0, j: Optional[int] = 0,
            conditional: bool = True) -> Array:
    """
    Killing resampling. The weights are assumed to be normalised already.
    Compared to the multinomial resampling, this algorithm does not move the indices when the weights are uniform.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
        Only used if conditional is True.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling

    Returns
    -------
    indices:
        Indices of the resampled particles.
    """

    # unconditional killing
    key_1, key_2, key_3 = jax.random.split(key, 3)

    N = weights.shape[0]
    w_max = weights.max()

    killed = (jax.random.uniform(key_1, (N,)) * w_max >= weights)
    idx = jnp.arange(N)
    idx = jnp.where(~killed, idx,
                    jax.random.choice(key_2, N, (N,), p=weights))
    if not conditional:
        return idx
    # Random permutation
    # TODO: logspace ?
    J_prob = (1. - weights / w_max) / N
    J_prob = J_prob.at[i].set(0.)
    J_prob_i = jnp.maximum(1 - jnp.sum(J_prob), 0.)
    J_prob = J_prob.at[i].set(J_prob_i)

    J = jax.random.choice(key_3, N, (), p=J_prob)
    idx = jnp.roll(idx, j - J)
    idx = idx.at[j].set(i)

    return idx


def systematic(key: PRNGKey, weights: Array, i: Optional[int] = 0, j: Optional[int] = 0,
               conditional: bool = True) -> Array:
    """
    Systematic resampling. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
        Only used if conditional is True.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling

    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    if conditional:
        return _conditional_systematic(key, weights, i, j)
    else:
        return _standard_systematic(key, weights)


def _standard_systematic(key: PRNGKey, weights: Array) -> Array:
    N = weights.shape[0]
    u = (jnp.arange(N) + jax.random.uniform(key)) / N
    cum_weights = jnp.cumsum(weights)
    indices = jnp.searchsorted(cum_weights, u)
    return indices


def _conditional_systematic(key: PRNGKey, weights: Array, i, j) -> Array:
    N = weights.shape[0]

    tmp = N * weights[i]
    tmp_floor = jnp.floor(tmp)

    U, V, W = jax.random.uniform(key, (3,))

    def _otherwise():
        rem = tmp - tmp_floor
        p_cond = rem * (tmp_floor + 1) / tmp
        return jax.lax.select(V < p_cond,
                              rem * U,
                              rem + (1. - rem) * U)

    uniform = jax.lax.cond(tmp <= 1,
                           lambda: tmp * U,
                           _otherwise)

    linspace = (jnp.arange(N, dtype=weights.dtype) + uniform) / N
    idx = jnp.searchsorted(jnp.cumsum(weights), linspace)

    n_i = jnp.sum(idx == i)
    zero_loc = jnp.flatnonzero(idx == i, size=N, fill_value=-1)
    roll_idx = jnp.floor(n_i * W).astype(int)

    idx = jnp.roll(idx, j - zero_loc[roll_idx])
    return jnp.clip(idx, 0, N - 1)



