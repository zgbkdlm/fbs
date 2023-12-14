"""
Implements the random walk cSMC kernel from Finke and Thiery (2023).

Due to Adrien Corenflos.
"""
import math
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from typing import Callable, Union, Any, Tuple
from fbs.typings import JArray, JKey, FloatScalar


def csmc(key: JKey,
         us_star: JArray, bs_star: JArray,
         vs: JArray, ts: JArray,
         init_sampler: Callable[[JKey, int], JArray],
         init_likelihood_logpdf: Callable[[JArray, JArray], JArray],
         transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
         transition_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
         likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
         cond_resampling: Callable,
         nsamples,
         niters: int,
         backward: bool = False):
    """
    Generic cSMC kernel.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    us_star : JArray (K + 1, du)
        Reference trajectory :math:`u_0^*, u_1^*, u_2^*, \ldots, u_K^*` to update.
    bs_star : JArray (K + 1, )
        Indices of the reference trajectory at times :math:`t_1, t_2, \ldots, t_K`.
    vs : JArray (K + 1, dv)
        Measurements :math:`v_0, v_1, \ldots, v_K`.
    ts : JArray (K + 1, )
        The times :math:`t_0, t_1, \ldots, t_K`.
    init_sampler : JKey, int -> (n, du)
    init_likelihood_logpdf : (dv, ), (n, du) -> (n, )
    transition_sampler : (n, du), (dv, ), float, key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
        That is, :math:`p(u_k | u_{k-1}, v_{k-1})`.
    transition_logpdf : (du, ), (du, ), (dv, ), float -> (n, )
    likelihood_logpdf : (dv, ), (n, du), (dv, ), float -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}. The fourth argument is for the time.
    cond_resampling :
        Resampling scheme to use.
    key : JKey
        Random number generator key.
    nsamples : int
        Number of particles to use (N+1, if we include the reference trajectory).

    Returns
    -------
    xs : JArray (niters, K + 1, d)
        Particles.
    """
    keys = jax.random.split(key, niters)

    def scan_body(carry, elem):
        us, bs = carry
        key_ = elem

        us, bs_next = csmc_kernel(key_,
                                  us, bs, vs, ts,
                                  init_sampler, init_likelihood_logpdf,
                                  transition_sampler, transition_logpdf,
                                  likelihood_logpdf,
                                  cond_resampling,
                                  nsamples,
                                  backward)
        accepted = bs_next != bs
        return (us, bs), (us, accepted)

    return jax.lax.scan(scan_body, (us_star, bs_star), keys)[1]


def csmc_kernel(key: JKey,
                us_star: JArray, bs_star: JArray,
                vs: JArray, ts: JArray,
                init_sampler: Callable[[JKey, int], JArray],
                init_likelihood_logpdf: Callable[[JArray, JArray], JArray],
                transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                transition_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                measurement_cond_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                cond_resampling: Callable,
                nsamples: int,
                backward: bool = False):
    """
    Generic cSMC kernel.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    us_star : JArray (K + 1, du)
        Reference trajectory :math:`u_0^*, u_1^*, u_2^*, \ldots, u_K^*` to update.
    bs_star : JArray (K + 1, )
        Indices of the reference trajectory at times :math:`t_1, t_2, \ldots, t_K`.
    vs : JArray (K + 1, dv)
        Measurements :math:`v_0, v_1, \ldots, v_K`.
    ts : JArray (K + 1, )
        The times :math:`t_0, t_1, \ldots, t_K`.
    init_sampler : JKey, int -> (n, du)
    init_likelihood_logpdf : (dv, ), (n, du) -> (n, )
    transition_sampler : (n, du), (dv, ), float, key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
        That is, :math:`p(u_k | u_{k-1}, v_{k-1})`.
    transition_logpdf : (du, ), (du, ), (dv, ), float -> (n, )
    measurement_cond_logpdf : (dv, ), (n, du), (dv, ), float -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}. The fourth argument is for the time.
    cond_resampling :
        Resampling scheme to use.
    nsamples : int
        Number of particles to use (N+1, if we include the reference trajectory).
    backward : bool, default=False
        Whether to run the backward sampling kernel.

    Returns
    -------
    xs_star : JArray (K + 1, du)
        Particles.
    bs_star : JArray (K + 1, )
        Indices of the ancestors.
    """
    key_fwd, key_bwd = jax.random.split(key, 2)

    As, log_ws, xss = forward_pass(key_fwd,
                                   us_star, bs_star,
                                   vs, ts,
                                   init_sampler, init_likelihood_logpdf,
                                   transition_sampler, measurement_cond_logpdf, cond_resampling, nsamples)
    if backward:
        xs_star, bs_star = backward_sampling_pass(key_bwd, transition_logpdf, vs, ts, xss, log_ws)
    else:
        xs_star, bs_star = backward_scanning_pass(key_bwd, As, xss, log_ws[-1])
    return xs_star, bs_star


def forward_pass(key: JKey,
                 us_star: JArray, bs_star: JArray,
                 vs: JArray, ts: JArray,
                 init_sampler: Callable[[JKey, int], JArray],
                 init_likelihood_logpdf: Callable[[JArray, JArray], JArray],
                 transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                 likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                 cond_resampling: Callable,
                 nsamples: int) -> Tuple[JArray, JArray, JArray]:
    r"""
    Forward pass of the cSMC kernel.

    u0, u1, ..., uK,
    v0, v1, ..., vK,

    u_{0:K}^*

    Parameters
    ----------
    us_star : JArray (K + 1, du)
        Reference trajectory :math:`u_0^*, u_1^*, u_2^*, \ldots, u_K^*` to update.
    bs_star : JArray (K + 1, )
        Indices of the reference trajectory at times :math:`t_1, t_2, \ldots, t_K`.
    vs : JArray (K + 1, dv)
        Measurements :math:`v_0, v_1, \ldots, v_K`.
    ts : JArray (K + 1, )
        The times :math:`t_0, t_1, \ldots, t_K`.
    init_sampler : JKey, int -> (n, du)
    init_likelihood_logpdf : (dv, ), (n, du) -> (n, )
    transition_sampler : (n, du), (dv, ), float, key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
        That is, :math:`p(u_k | u_{k-1}, v_{k-1})`.
    likelihood_logpdf : (dv, ), (n, du), (dv, ), float -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}. The fourth argument is for the time.
    cond_resampling :
        Resampling scheme to use.
    key : JKey
        Random number generator key.
    nsamples : int
        Number of particles to use (N + 1, if we include the reference trajectory).

    Returns
    -------
    JArray (K, n), JArray (K, n + 1), JArray (K, n + 1, du)
        The collections of ancestors, log weights, and smoothing samples.
    """
    K_plus_one = us_star.shape[0]
    nsteps = K_plus_one - 1

    def scan_body(carry, inp):
        log_ws, us = carry
        v, v_prev, t_prev, b_star_prev, b_star, key_, u_star = inp
        key_resampling, key_transition = jax.random.split(key_, num=2)

        # Conditional resampling
        A = cond_resampling(key_resampling, jnp.exp(log_ws), b_star_prev, b_star, True)
        us = jnp.take(us, A, axis=0)

        us = transition_sampler(us, v_prev, t_prev, key_transition)
        us = us.at[b_star].set(u_star)

        log_ws = likelihood_logpdf(v, us, v_prev, t_prev)
        log_ws = normalise(log_ws, log_space=True)

        return (log_ws, us), (log_ws, A, us)

    key_init, key_scan = jax.random.split(key, num=2)
    us0 = init_sampler(key_init, nsamples + 1)
    us0 = us0.at[bs_star[0]].set(us_star[0])

    log_ws0 = init_likelihood_logpdf(vs[0], us0)
    log_ws0 = normalise(log_ws0, log_space=True)

    keys = jax.random.split(key_scan, nsteps)
    inputs = (vs[1:], vs[:-1], ts[:-1], bs_star[:-1], bs_star[1:], keys, us_star[1:])
    _, (log_wss, As, uss) = jax.lax.scan(scan_body, (log_ws0, us0), inputs)

    log_wss = jnp.insert(log_wss, 0, log_ws0, axis=0)
    uss = jnp.insert(uss, 0, us0, axis=0)

    return As, log_wss, uss


def backward_sampling_pass(key, transition_logpdf, vs, ts, uss, log_ws):
    """
    Backward sampling pass for the cSMC kernel.

    Parameters
    ----------
    key:
        Random number generator key.
    transition_logpdf:
        The transition logpdf. p(x_k | x_{k-1})
    uss:
        JArray of particles.
    log_ws:
        JArray of log-weights for the filtering solution.

    Returns
    -------
    xs:
        JArray of particles.
    Bs:
        JArray of indices of the last ancestor.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################

    K_plus_one, n, *_ = uss.shape
    keys = jax.random.split(key, K_plus_one)

    ###############################
    #        BACKWARD PASS        #
    ###############################
    # Select last ancestor
    W_T = normalise(log_ws[-1], )
    B_T = barker_move(keys[-1], W_T)
    x_T = uss[-1, B_T]

    def body(x_t, inp):
        op_key, xs_t_m_1, log_w_t_m_1, v, t = inp
        Gamma_log_w = transition_logpdf(x_t, xs_t_m_1, v, t)  # I swapped the order
        Gamma_log_w -= jnp.max(Gamma_log_w)
        log_w = Gamma_log_w + log_w_t_m_1
        w = normalise(log_w)
        B_t_m_1 = jax.random.choice(op_key, w.shape[0], p=w, shape=())
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return x_t_m_1, (x_t_m_1, B_t_m_1)

    # Reverse arrays, ideally, should use jax.lax.scan(reverse=True) but it is simpler this way due to insertions.
    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable... Same for log_ws.
    # vs[-1:0:-1] means the reverse of vs[1:]
    inps = keys[:-1], uss[-2::-1], log_ws[-2::-1], vs[-1:0:-1], ts[-1:0:-1]

    # Run backward pass
    _, (uss, Bs) = jax.lax.scan(body, x_T, inps)

    # Insert last ancestor and particle
    uss = jnp.insert(uss, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)

    return uss[::-1], Bs[::-1]


def backward_scanning_pass(key, As, xss, log_w_T):
    """
    Backward scanning pass for the cSMC kernel.

    Parameters
    ----------
    key:
        Random number generator key.
    As:
        JArray of indices of the ancestors.
    xss:
        JArray of particles.
    log_w_T:
        Log-weight of the last ancestor.

    Returns
    -------
    xs:
        JArray of particles.
    Bs:
        JArray of indices of the star trajectory.
    """

    ###############################
    #        BACKWARD PASS        #
    ###############################
    # Select last ancestor
    B_T = barker_move(key, normalise(log_w_T))
    x_T = xss[-1, B_T]

    def body(B_t, inp):
        xs_t_m_1, A_t = inp
        B_t_m_1 = A_t[B_t]
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return B_t_m_1, (x_t_m_1, B_t_m_1)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable...
    _, (xss, Bs) = jax.lax.scan(body, B_T, (xss[-2::-1], As[::-1]))
    xss = jnp.insert(xss, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xss[::-1], Bs[::-1]


def normalise(log_weights: JArray, log_space=False) -> JArray:
    """
    Normalize log weights to obtain unnormalised weights.

    Parameters
    ----------
    log_weights:
        Log weights to normalize.
    log_space:
        If True, the output is in log space. Otherwise, the output is in natural space.

    Returns
    -------
    log_weights/weights:
        Unnormalised weights.
    """
    log_weights -= logsumexp(log_weights)
    if log_space:
        return log_weights
    return jnp.exp(log_weights)


def barker_move(key, ws):
    n = ws.shape[0]
    return jax.random.choice(key, n, (), p=ws)
