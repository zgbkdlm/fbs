import jax
import jax.numpy as jnp
from fbs.typings import JArray, JFloat, JKey
from typing import Callable, Tuple


def bootstrap_filter(transition_sampler: Callable[[JArray, JArray, JKey], JArray],
                     measurement_cond_pdf: Callable[[JArray, JArray, JArray], JArray],
                     vs: JArray,
                     init_sampler: Callable[[JArray, JArray, int], JArray],
                     key: JKey,
                     nsamples: int,
                     resampling: Callable[[JArray, JArray], JArray],
                     log: bool = True) -> Tuple[JArray, JFloat]:
    r"""Bootstrap particle filter, using the notations in the paper.

    Parameters
    ----------
    transition_sampler : (n, du), (dv, ), key -> (n, du)
        Draw n new samples conditioned on the previous samples and :math:`v_{k-1}`.
    measurement_cond_pdf : (dv, ), (n, du), (dv, ) -> (n, )
        The measurement conditional PDF :math:`p(v_k | u_{k-1}, v_{k-1})`.
        The first function argument is for v. The second argument is for u, which accepts an array of samples
        and output an array of evaluations. The third argument is for v_{k-1}.
    vs : JArray (K + 1, dv)
        Measurements :math:`v_0, v_1, \ldots, v_K`.
    init_sampler : JKey, (dv, ), int -> (n, du)
        Sampler for :math:`p(u0 | v0)`.
    key : JKey
        PRNGKey.
    nsamples : int
        Number of samples/particles n.
    resampling : (n, ), key -> (n, )
        Resample method.
    log : bool, default=True
        Whether run the particle filter in the log domain. If True, then the function `measurement_cond_pdf` should
        provide the log pdf.

    Returns
    -------
    JArray (K, n, du), JFloat
        The particle filtering samples, and the negative log likelihood.
    """

    def scan_body(carry, elem):
        samples, log_nell = carry
        v, v_prev, key_ = elem

        samples = transition_sampler(samples, v_prev, key_)

        if log:
            log_weights = measurement_cond_pdf(v, samples, v_prev) - jnp.log(nsamples)
            _c = jax.scipy.special.logsumexp(log_weights)
            log_nell -= _c
            log_weights = log_weights - _c

            _, subkey_ = jax.random.split(key_)
            samples = samples[resampling(jnp.exp(log_weights), subkey_), ...]
        else:
            weights = measurement_cond_pdf(v, samples, v_prev) / nsamples
            log_nell -= jnp.log(jnp.sum(weights))
            weights = weights / jnp.sum(weights)

            _, subkey_ = jax.random.split(key_)
            samples = samples[resampling(weights, subkey_), ...]

        return (samples, log_nell), samples

    nsteps = vs.shape[0] - 1

    init_samples = init_sampler(key, vs[0], nsamples)
    keys = jax.random.split(key, num=nsteps)

    (*_, nell_ys), filtering_samples = jax.lax.scan(scan_body, (init_samples, 0.), (vs[1:], vs[:-1], keys))
    return filtering_samples, nell_ys
