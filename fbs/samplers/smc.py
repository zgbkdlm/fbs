import math
import jax
import jax.numpy as jnp
from fbs.samplers.common import MCMCState
from fbs.typings import JArray, JFloat, JKey, FloatScalar
from typing import Callable, Tuple, Optional


def bootstrap_filter(transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                     measurement_cond_pdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                     vs: JArray,
                     ts: JArray,
                     init_sampler: Callable[[JArray, JArray, int], JArray],
                     key: JKey,
                     nparticles: int,
                     resampling: Callable[[JArray, JArray], JArray],
                     log: bool = True,
                     return_last: bool = True,
                     **kwargs) -> Tuple[JArray, JFloat]:
    r"""Bootstrap particle filter, using the notations in the paper.

    Parameters
    ----------
    transition_sampler : (n, du), (dv, ), float, key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
    measurement_cond_pdf : (dv, ), (n, du), (dv, ), float -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}. The fourth argument is for the time.
    vs : JArray (K + 1, dv)
        Measurements :math:`v_0, v_1, \ldots, v_K`.
    ts : JArray (K + 1, )
        The times :math:`t_0, t_1, \ldots, t_K`.
    init_sampler : JKey, (dv, ), int -> (n, du)
        Sampler for :math:`p(u0 | v0)`.
    key : JKey
        PRNGKey.
    nparticles : int
        Number of samples/particles n.
    resampling : (n, ), key -> (n, )
        Resample method.
    log : bool, default=True
        Whether run the particle filter in the log domain. If True, then the function `measurement_cond_pdf` should
        provide the log pdf.
    return_last : bool, default=True
        Whether return the particle samples at the last time step only.

    Returns
    -------
    JArray (K, n, du), JFloat
        The particle filtering samples, and the negative log likelihood.

    Notes
    -----
    Lazy: using resampling at every step.
    """

    def scan_body(carry, elem):
        us_prev, log_nell = carry
        v, v_prev, t_prev, key_ = elem

        us = transition_sampler(us_prev, v_prev, t_prev, key_, **kwargs)

        if log:
            log_weights = measurement_cond_pdf(v, us_prev, v_prev, t_prev, **kwargs)
            _c = jax.scipy.special.logsumexp(log_weights)
            log_nell -= _c - math.log(nparticles)
            log_weights = log_weights - _c
            weights = jnp.exp(log_weights)
        else:
            weights = measurement_cond_pdf(v, us_prev, v_prev, t_prev, **kwargs)
            log_nell -= jnp.log(jnp.mean(weights))
            weights = weights / jnp.sum(weights)

        _, subkey_ = jax.random.split(key_)
        us = us[resampling(weights, subkey_), ...]

        return (us, log_nell), None if return_last else us

    nsteps = vs.shape[0] - 1
    key_init, key_steps = jax.random.split(key)
    init_samples = init_sampler(key_init, vs[0], nparticles)
    keys = jax.random.split(key_steps, num=nsteps)

    (last_samples, nell_ys), filtering_samples = jax.lax.scan(scan_body,
                                                              (init_samples, 0.),
                                                              (vs[1:], vs[:-1], ts[:-1], keys))
    if return_last:
        return last_samples, nell_ys
    else:
        filtering_samples = jnp.concatenate([jnp.expand_dims(init_samples, axis=0), filtering_samples], axis=0)
        return filtering_samples, nell_ys


def bootstrap_backward_smoother(key: JKey,
                                filter_us: JArray, vs: JArray, ts: JArray,
                                transition_logpdf: Callable,
                                *args, **kwargs) -> JArray:
    """Backward particle smoother by using the results from a bootstrap filter.
    """

    def scan_body(carry, elem):
        u_kp1 = carry
        uf_k, v_k, t_k, key_ = elem

        log_ws = transition_logpdf(u_kp1, uf_k, v_k, t_k, *args, **kwargs)
        log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)
        # jax.debug.print('{}', jnp.exp(log_ws))
        u_k = jax.random.choice(key_, uf_k, axis=0, p=jnp.exp(log_ws))
        return u_k, u_k

    nsteps = filter_us.shape[0] - 1
    key_last, key_smoother = jax.random.split(key, num=2)
    uT = jax.random.choice(key, filter_us[-1], axis=0)
    traj = jax.lax.scan(scan_body, uT, (filter_us[-2::-1], vs[-2::-1], ts[-2::-1],
                                        jax.random.split(key_smoother, num=nsteps)))[1][::-1]
    return jnp.concatenate([traj, jnp.expand_dims(uT, axis=0)], axis=0)


def pmcmc_filter_step(key: JKey, vs_bridge: JArray, u0s: JArray, ts: JArray,
                      transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                      likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray], resampling: Callable,
                      nparticles: int, **kwargs) -> Tuple[JArray, JFloat]:
    """Particle MCMC sampling of p(x | y) for separable forward process.

    Parameters
    ----------
    key
    vs_bridge
    ts
    u0s
    transition_sampler
    likelihood_logpdf
    resampling
    nparticles

    Returns
    -------

    """
    log_ell0 = 0.

    def scan_body(carry, elem):
        us_prev, log_ell = carry
        v, v_prev, t_prev, key_ = elem

        key_proposal, key_resampling = jax.random.split(key_)
        us = transition_sampler(us_prev, v_prev, t_prev, key_proposal, **kwargs)

        log_ws = likelihood_logpdf(v, us_prev, v_prev, t_prev, **kwargs)
        _c = jax.scipy.special.logsumexp(log_ws)
        log_ell = log_ell - math.log(nparticles) + _c
        log_ws = log_ws - _c

        us = us[resampling(jnp.exp(log_ws), key_resampling), ...]

        return (us, log_ell), None

    keys = jax.random.split(key, num=ts.shape[0] - 1)
    (uT, log_ellT), *_ = jax.lax.scan(scan_body,
                                      (u0s, log_ell0),
                                      (vs_bridge[1:], vs_bridge[:-1], ts[:-1], keys))
    return uT, log_ellT


def pcn_proposal(key, delta: float, x: JArray, mean: JArray, sampler):
    beta = 2 / (2 + delta)
    key_rnds = jax.random.split(key, num=2)
    rnds = jax.vmap(sampler)(key_rnds)
    p = x + math.sqrt(delta / 2) * (rnds[0] - mean)
    return beta * p + (1 - beta) * mean + math.sqrt(1 - beta) * (rnds[1] - mean)


def pmcmc_kernel(key: JKey,
                 uT, log_ell, ys,
                 y0: JArray,
                 ts: JArray,
                 fwd_ys_sampler,
                 sde,
                 ref_sampler: Callable,
                 transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                 likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                 resampling: Callable,
                 nparticles: int,
                 delta: float = None,
                 which_u: int = 0,
                 **kwargs) -> Tuple[JArray, JFloat, JArray, JArray, MCMCState]:
    r"""A particle MCMC kernel for variables (uT, log_ell, yT, xT) targeting at p(uT | vT = y0)

    Parameters
    ----------
    key : JKey
        A JAX random key.
    uT : JArray (du, )
        MCMC sample for uT.
    log_ell : JFloat
        MCMC sample for log_ell.
    yT : JArray (dv, )
        MCMC sample for yT.
    xT : JArray (du, )
        MCMC sample for xT.
    ts : JArray (K + 1, )
        Time steps :math`t_0, t_1, \ldots, t_K`.
    y0 : JArray (dv, )
    fwd_ys_sampler : JKey, (dv, ), (K + 1, ) -> (K + 1, dv)
        A sampler for the forward process :math:`y_0, y_1, \ldots, y_K`.
    ref_sampler : JKey, int -> (n, du)
        Sampling the reference distribution for xT (or u0). This should return a Dirac.
    ref_logpdf : JArray (du, ) -> JFloat
        The log PDF of the reference measure.
    transition_sampler : (n, du), (dv, ), float, key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
    likelihood_logpdf : (dv, ), (n, du), (dv, ), float -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}. The fourth argument is for the time.
    resampling : (n, ), key -> (n, )
        Resample method.
    nparticles : int
        The number of particles.
    which_u : int, default=0
        Which particle to choose.

    Returns
    -------
    JArray (du, ), JFloat, JArray (dv, ), JArray (du, )
        An MCMC sample tuple for uT, log_ell, yT, and xT.

    Notes
    -----
    For the time being, let's assume that we can store the trajectory of Y. Implementing its online backward bridge is
    annoying.
    """
    key_prop, key_u0, key_filter, key_mh = jax.random.split(key, num=4)

    if delta is None:
        prop_ys = fwd_ys_sampler(key_prop, y0)
    else:
        mean = jax.vmap(sde.mean, in_axes=[0, None, None])(ts, ts[0], y0)
        prop_ys = pcn_proposal(key_prop, delta, ys, mean, lambda key_: fwd_ys_sampler(key_, y0))

    vs = prop_ys[::-1]

    u0s = ref_sampler(key_u0, vs[0], nparticles)  # p(u0 | y0)
    prop_uTs, prop_log_ell = pmcmc_filter_step(key_filter, vs, u0s, ts, transition_sampler, likelihood_logpdf,
                                               resampling, nparticles, **kwargs)
    prop_uT = prop_uTs[which_u]

    log_acc_prob = jnp.minimum(0., prop_log_ell - log_ell)

    z = jax.random.uniform(key_mh)
    acc_flag = jnp.log(z) < log_acc_prob

    mcmc_state = MCMCState(acceptance_prob=jnp.exp(log_acc_prob),
                           is_accepted=acc_flag,
                           prop_log_ell=prop_log_ell,
                           log_ell=log_ell)
    return jax.lax.cond(acc_flag,
                        lambda _: (prop_uT, prop_log_ell, prop_ys, mcmc_state),
                        lambda _: (uT, log_ell, ys, mcmc_state),
                        None)


def twisted_smc(key: JKey, y: JArray, ts: JArray,
                init_sampler: Callable[[JKey, int], JArray],
                transition_logpdf: Callable[[JArray, JArray, JArray], JArray],
                twisting_logpdf: Callable[[JArray, JArray, FloatScalar], JArray],
                twisting_prop_sampler: Callable[[JKey, JArray, FloatScalar, JArray], JArray],
                twisting_prop_logpdf: Callable[[JArray, JArray, FloatScalar, JArray], JArray],
                resampling: Callable,
                nparticles: int) -> Tuple[JArray, JArray]:
    """Implementation of the twisted SMC sampler.

    Notes
    -----
    Algorithm 1, https://arxiv.org/pdf/2306.17775.pdf.
    """

    def scan_body(carry, elem):
        xs_prev, log_ps_prev, log_ws = carry
        key_, t_prev = elem

        key_resampling, key_prop = jax.random.split(key_)

        # Resampling
        resampling_inds = resampling(jnp.exp(log_ws), key_resampling)
        xs_prev = xs_prev[resampling_inds, ...]
        log_ps_prev = log_ps_prev[resampling_inds, ...]

        # Proposal
        xs = twisting_prop_sampler(key_prop, xs_prev, t_prev, y)

        # Weights
        log_ps = twisting_logpdf(y, xs, t_prev)
        log_ws = (transition_logpdf(xs, xs_prev, t_prev) + log_ps
                  - twisting_prop_logpdf(xs, xs_prev, t_prev, y) - log_ps_prev)
        log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

        return (xs, log_ps, log_ws), None

    nsteps = ts.shape[0] - 1
    key_init, key_filter = jax.random.split(key, num=2)
    keys = jax.random.split(key_filter, num=nsteps)

    init_xs = init_sampler(key_init, nparticles)
    init_log_ps = twisting_logpdf(y, init_xs, ts[0])
    init_log_ws = init_log_ps - jax.scipy.special.logsumexp(init_log_ps)

    (samples, _, log_weights), _ = jax.lax.scan(scan_body,
                                                (init_xs, init_log_ps, init_log_ws),
                                                (keys, ts[1:]))
    return samples, log_weights
