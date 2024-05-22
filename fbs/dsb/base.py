import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, JFloat, FloatScalar
from typing import Callable


def ipf_loss_disc(param: JArray,
                  simulator_param: JArray,
                  x0s: JArray,
                  ks: JArray,
                  gammas: FloatScalar,
                  parametric_fn: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_fn: Callable[[JArray, FloatScalar, JArray], JArray],
                  key: JKey) -> JFloat:
    """The iterative proportional fitting (discretised version) loss used in Schrödinger bridge.
    """
    nsamples, d = x0s.shape
    nsteps = ks.shape[0] - 1

    def scan_body(carry, elem):
        x, err = carry
        k, k_next, gamma, rnd = elem

        x_next = simulator_fn(x, k, simulator_param) + jnp.sqrt(gamma) * rnd
        err = err + jnp.mean((parametric_fn(x_next, k_next, param) - (
                x_next + simulator_fn(x, k, simulator_param) - simulator_fn(x_next, k, simulator_param))) ** 2)
        return (x_next, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, nsamples, d))
    (_, err_final), _ = jax.lax.scan(scan_body, (x0s, 0.), (ks[:-1], ks[1:], gammas, rnds))
    return jnp.mean(err_final)


def ipf_loss_cont(key: JKey,
                  param: JArray,
                  simulator_param: JArray,
                  init_samples: JArray,
                  ts: JArray,
                  parametric_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  dispersion: Callable) -> JFloat:
    r"""The iterative proportional fitting (discretised version) loss used in Schrödinger bridge.
    Proposition 29, de Bortoli et al., 2021.

    Forward
    .. math::

        X_{k+1} = X_k + f(k, X_k) \delta_k / 2 + \xi_k.

    Backward
    .. math::

        X_k = X_{k+1} - b(k+1, X_{k+1}) \delta_k / 2 + \zeta_k,

    where :math:`\delta_k = \lvert t_{k+1} - t_k \rvert`.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    param : JArray
        The parameter of the parametric drift function that you wish to learn.
    simulator_param : JArray
        The parameter of the simulation process drift function.
    init_samples : JArray (m, n, ...)
        Samples from the initial distribution (i.e., either the target or the reference depending on if you are
        learning the forward or the backward process).
    ts : JArray (n, )
        Either the forward times `t_0, t_1, ..., t_n` or its reversal, depending on if you are using this function
        to learn the backward or forward process.
    parametric_drift : Callable
        The parametric drift function whose signature is `f(x, t, param)`.
    simulator_drift : Callable
        The simulator process' drift function whose signature is `g(x, t, simulator_param)`.
    dispersion : Callable
        The dispersion function, a function of time.

    Returns
    -------
    JFloat
        The loss.

    Notes
    -----
    When using this function to learn the backward process `target <- ref`,
    simulate the forward process defined by `simulator_drift` at forward `ts`.

    When using this function to learn the forward process `target -> ref`,
    simulate the backward process defined by `simulator_drift` at backward `ts`.
    """
    nsteps = ts.shape[0] - 1
    fn = lambda x, t, dt: x + simulator_drift(x, t, simulator_param) * dt

    def scan_body(carry, elem):
        x, err = carry
        t, t_next, rnd = elem

        dt = jnp.abs(t_next - t)
        x_next = x + simulator_drift(x, t, simulator_param) * dt + jnp.sqrt(dt) * dispersion(t) * rnd
        err = err + jnp.mean(
            (parametric_drift(x_next, t_next, param) * dt - (fn(x, t, dt) - fn(x_next, t, dt))) ** 2)
        return (x_next, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, *init_samples.shape))
    (_, err_final), _ = jax.lax.scan(scan_body, (init_samples, 0.), (ts[:-1], ts[1:], rnds))
    return jnp.mean(err_final / nsteps)


def ipf_loss_cont_v(key: JKey,
                    param: JArray,
                    simulator_param: JArray,
                    init_samples: JArray,
                    ts: JArray,
                    parametric_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                    simulator_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                    dispersion: Callable) -> JFloat:
    """This is identical to `ipf_loss_cont` but implemented via vmap for faster speed.
    """
    nsteps = ts.shape[0] - 1
    fn = lambda x, t, dt: x + simulator_drift(x, t, simulator_param) * dt

    def scan_body(carry, elem):
        x = carry
        t, t_next, rnd = elem

        dt = jnp.abs(t_next - t)
        x = x + simulator_drift(x, t, simulator_param) * dt + jnp.sqrt(dt) * dispersion(t) * rnd
        return x, x

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, *init_samples.shape))
    _, trajs = jax.lax.scan(scan_body, init_samples, (ts[:-1], ts[1:], rnds))
    trajs = jnp.concatenate([jnp.expand_dims(init_samples, axis=0), trajs], axis=0)  # (nsteps + 1, batch, d)

    dts = jnp.expand_dims(jnp.abs(jnp.diff(ts)), axis=list([i + 1 for i in range(init_samples.ndim)]))
    errs = jax.vmap(parametric_drift, in_axes=[0, 0, None])(trajs[1:], ts[1:], param) * dts - (
            jax.vmap(fn, in_axes=[0, 0, 0])(trajs[:-1], ts[:-1], dts) - jax.vmap(fn, in_axes=[0, 0, 0])(trajs[1:],
                                                                                                        ts[:-1],
                                                                                                        dts))
    return jnp.mean(errs ** 2)
