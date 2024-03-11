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
                     return_last: bool = True) -> Tuple[JArray, JFloat]:
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

        us = transition_sampler(us_prev, v_prev, t_prev, key_)

        if log:
            log_weights = measurement_cond_pdf(v, us_prev, v_prev, t_prev)
            _c = jax.scipy.special.logsumexp(log_weights)
            log_nell -= _c - math.log(nparticles)
            log_weights = log_weights - _c
            weights = jnp.exp(log_weights)
        else:
            weights = measurement_cond_pdf(v, us_prev, v_prev, t_prev)
            log_nell -= jnp.log(jnp.mean(weights))
            weights = weights / jnp.sum(weights)

        _, subkey_ = jax.random.split(key_)
        us = us[resampling(weights, subkey_), ...]

        return (us, log_nell), None if return_last else us

    nsteps = vs.shape[0] - 1
    init_samples = init_sampler(key, vs[0], nparticles)
    keys = jax.random.split(key, num=nsteps)

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
                                transition_logpdf: Callable) -> JArray:
    """Backward particle smoother by using the results from a bootstrap filter.
    """
    def scan_body(carry, elem):
        u_kp1 = carry
        uf_k, v_k, t_k, key_ = elem

        log_ws = transition_logpdf(u_kp1, uf_k, v_k, t_k)
        log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)
        u_k = jax.random.choice(key_, uf_k, axis=0, p=jnp.exp(log_ws))
        return u_k, u_k

    nsteps = filter_us.shape[0] - 1
    key_last, key_smoother = jax.random.split(key, num=2)
    uT = jax.random.choice(key, filter_us[-1], axis=0)
    traj = jax.lax.scan(scan_body, uT, (filter_us[-2::-1], vs[-2::-1], ts[-2::-1],
                                        jax.random.split(key_smoother, num=nsteps)))[1][::-1]
    return jnp.concatenate([traj, jnp.expand_dims(uT, axis=0)], axis=0)


def pf_temp(key: JKey,
            vs: JArray, ts: JArray,
            u0: JArray,
            transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
            likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
            resampling: Callable,
            nparticles: int) -> JArray:
    """Particle MCMC sampling of p(x | y) for separable forward process.

    Parameters
    ----------
    key
    vs
    ts
    u0
    transition_sampler
    likelihood_logpdf
    resampling
    nparticles

    Returns
    -------

    """

    def scan_body(carry, elem):
        us_prev = carry
        v, v_prev, t_prev, key_ = elem

        key_proposal, key_resampling = jax.random.split(key_)
        us = transition_sampler(us_prev, v_prev, t_prev, key_proposal)

        log_ws = likelihood_logpdf(v, us_prev, v_prev, t_prev)
        _c = jax.scipy.special.logsumexp(log_ws)
        log_ws = log_ws - _c

        us = us[resampling(jnp.exp(log_ws), key_resampling), ...]

        return us, None

    keys = jax.random.split(key, num=ts.shape[0] - 1)

    u0s = u0 * jnp.ones((nparticles, u0.shape[0]))
    return jax.lax.scan(scan_body, u0s, (vs[1:], vs[:-1], ts[:-1], keys))[0]


def pmcmc_step(key: JKey,
               vs_bridge: JArray, ts: JArray,
               u0s: JArray,
               transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
               likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
               resampling: Callable,
               nparticles: int) -> Tuple[JArray, JFloat]:
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
        us, log_ell = carry
        v, v_prev, t_prev, key_ = elem

        key_proposal, key_resampling = jax.random.split(key_)
        us = transition_sampler(us, v_prev, t_prev, key_proposal)

        log_ws = likelihood_logpdf(v, us, v_prev, t_prev)
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


def pmcmc_kernel(key: JKey,
                 uT, log_ell, yT, xT,
                 ts: JArray,
                 y0: JArray,
                 fwd_ys_sampler: Callable[[JKey, JArray, JArray], JArray],
                 ref_sampler: Callable[[JKey, int, Optional[JArray]], JArray],
                 ref_logpdf: Callable[[JArray, Optional[JArray]], JArray],
                 transition_sampler: Callable[[JArray, JArray, FloatScalar, JKey], JArray],
                 likelihood_logpdf: Callable[[JArray, JArray, JArray, FloatScalar], JArray],
                 resampling: Callable,
                 nparticles: int,
                 which_u: int = 0) -> Tuple[JArray, JFloat, JArray, JArray, MCMCState]:
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
    key_fwd_ys, key_u0, key_pmcmc, key_mh = jax.random.split(key, num=4)
    ys = fwd_ys_sampler(key_fwd_ys, y0, ts)
    vs = ys[::-1]
    prop_yT = ys[-1]

    u0s = ref_sampler(key_u0, nparticles, prop_yT)
    prop_uTs, prop_log_ell = pmcmc_step(key_pmcmc,
                                        vs, ts,
                                        u0s,
                                        transition_sampler,
                                        likelihood_logpdf,
                                        resampling,
                                        nparticles)
    prop_uT = prop_uTs[which_u]
    prop_xT = u0s[which_u]

    # log_acc_prob = jnp.minimum(0.,
    #                            ref_logpdf(prop_xT, prop_yT) - ref_logpdf(xT, prop_yT)
    #                            + prop_log_ell - log_ell)
    # log_acc_prob = jnp.minimum(0., prop_log_ell - log_ell)
    #
    # z = jax.random.uniform(key_mh)
    # acc_flag = jnp.log(z) < log_acc_prob
    #
    # mcmc_state = MCMCState(acceptance_prob=jnp.exp(log_acc_prob), is_accepted=acc_flag)
    acc_prob = jnp.minimum(1., jnp.exp(prop_log_ell - log_ell))

    z = jax.random.uniform(key_mh)
    acc_flag = z < acc_prob

    mcmc_state = MCMCState(acceptance_prob=acc_prob, is_accepted=acc_flag)
    return jax.lax.cond(acc_flag,
                        lambda _: (prop_uT, prop_log_ell, prop_yT, prop_xT, mcmc_state),
                        lambda _: (uT, log_ell, yT, xT, mcmc_state),
                        None)
