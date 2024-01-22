import jax
import jax.numpy as jnp
from fbs.dsb import ipf_loss
from fbs.typings import JArray, JKey, FloatScalar
from typing import NamedTuple


class LinearSDE:
    pass


class StationaryConstLinearSDE(LinearSDE):
    """dX(t) = a X(t) dt + b dW(t), where
        - `b^2 / a = 2 sigma^2`
    """
    a: FloatScalar
    b: FloatScalar

    def __init__(self, a: FloatScalar, b: FloatScalar):
        self.a, self.b = a, b

    def drift(self, x, t):
        return self.a * x

    def dispersion(self, t):
        return self.b


class StationaryLinLinearSDE(LinearSDE):
    """dX(t) = a(t) X(t) dt + b(t) dW(t), where
        - `b(t)^2 / a(t) = 2 sigma^2`
        - `a(t) = a t` and `b(t) = b sqrt(t)`, where `a` and `b` are fixed.
    """
    a: FloatScalar
    b: FloatScalar

    def __init__(self, a: FloatScalar, b: FloatScalar):
        self.a, self.b = a, b

    def drift(self, x, t):
        return self.a * t * x

    def dispersion(self, t):
        return self.b * jnp.sqrt(t)


class StationaryExpLinearSDE(LinearSDE):
    """dX(t) = a(t) X(t) dt + b(t) dW(t), where
        - `b(t)^2 / a(t) = 2 sigma^2`
        - `a(t) = a exp(c (t - z))` and `b(t) = b exp(c (t - z) / 2)`, where `a`, `b`, and `z` are fixed.
    """
    a: FloatScalar
    b: FloatScalar
    c: FloatScalar
    z: FloatScalar

    def __init__(self, a: FloatScalar, b: FloatScalar, c: FloatScalar, z: FloatScalar):
        self.a, self.b, self.c, self.z = a, b, c, z

    def drift(self, x, t):
        return self.a * jnp.exp(self.c * (t - self.z)) * x

    def dispersion(self, t):
        return self.b * jnp.exp(self.c * (t - self.z) / 2)


def make_ou_sde(a, b):
    """Independent OU SDEs of the form `dX = a X dt + b dW`.
    """

    def discretise_ou_sde(t):
        return jnp.exp(a * t), b ** 2 / (2 * a) * (jnp.exp(2 * a * t) - 1)

    def cond_score_t_0(x: JArray, t, x0: JArray):
        F, Q = discretise_ou_sde(t)
        return -(x - F * x0) / Q

    def simulate_cond_forward(key: JKey, x0: JArray, ts: JArray, keep_path: bool = True) -> JArray:
        """Simulate a path of the OU process at ts starting from x0.

        Parameters
        ----------
        key : JKey
        x0 : JArray (d, )
        ts : JArray (nsteps + 1, )
            t_0, t_1, ..., t_nsteps.
        keep_path : bool, default=True
            Let it be true will make the returned sample a valid sample path from the SDE. Othwerwise, the return are
            independent samples at each time point marginally.

        Returns
        -------
        JArray (nsteps + 1, d)
            X_0, X_1, ..., X_nsteps.
        """

        def scan_body(carry, elem):
            x = carry
            dt, rnd = elem

            F, Q = discretise_ou_sde(dt)
            x = F * x + jnp.sqrt(Q) * rnd
            return x, x

        if keep_path:
            dts = jnp.diff(ts)
            rnds = jax.random.normal(key, (dts.shape[0], x0.shape[0]))
            return jnp.concatenate([x0[None, :], jax.lax.scan(scan_body, x0, (dts, rnds))[1]], axis=0)
        else:
            Fs, Qs = discretise_ou_sde(ts)
            rnds = jax.random.normal(key, (ts.shape[0], x0.shape[0]))
            return Fs[:, None] * x0[None, :] + jnp.sqrt(Qs)[:, None] * rnds

    return discretise_ou_sde, cond_score_t_0, simulate_cond_forward


def make_linear_sde(sde: LinearSDE):
    """Discretisation of linear SDEs of the form dX(t) = a(t) X(t) dt + b(t) dW(t).
    """

    def discretise_linear_sde(t, s):
        if isinstance(sde, StationaryLinLinearSDE):
            a, b = sde.a, sde.b
            r = a * 0.5 * (t ** 2 - s ** 2)
            stationary_variance = -b ** 2 / (2 * a)
            return jnp.exp(r), stationary_variance * (1 - jnp.exp(2 * r))
        elif isinstance(sde, StationaryConstLinearSDE):
            a, b = sde.a, sde.b
            return jnp.exp(a * (t - s)), b ** 2 / (2 * a) * (jnp.exp(2 * a * (t - s)) - 1)
        elif isinstance(sde, StationaryExpLinearSDE):
            a, b, c, z = sde.a, sde.b, sde.c, sde.z
            stationary_variance = -b ** 2 / (2 * a)
            r = a * (jnp.exp(c * (t - z)) - jnp.exp(c * (s - z))) / c
            return jnp.exp(r), stationary_variance * (1 - jnp.exp(2 * r))
        else:
            raise NotImplementedError('...')

    def cond_score_t_0(x: JArray, t, x0: JArray, s):
        F, Q = discretise_linear_sde(t, s)
        return -(x - F * x0) / Q

    def simulate_cond_forward(key: JKey, x0: JArray, ts: JArray, keep_path: bool = True) -> JArray:
        """Simulate a path of the OU process at ts starting from x0.

        Parameters
        ----------
        key : JKey
        x0 : JArray (d, )
        ts : JArray (nsteps + 1, )
            t_0, t_1, ..., t_nsteps.
        keep_path : bool, default=True
            Let it be true will make the returned sample a valid sample path from the SDE. Othwerwise, the return are
            independent samples at each time point marginally.

        Returns
        -------
        JArray (nsteps + 1, d)
            X_0, X_1, ..., X_nsteps.
        """

        def scan_body(carry, elem):
            x = carry
            t, t_prev, rnd = elem

            F, Q = discretise_linear_sde(t, t_prev)
            x = F * x + jnp.sqrt(Q) * rnd
            return x, x

        if keep_path:
            rnds = jax.random.normal(key, (ts.shape[0] - 1, x0.shape[0]))
            return jnp.concatenate([x0[None, :], jax.lax.scan(scan_body, x0, (ts[1:], ts[:-1], rnds))[1]], axis=0)
        else:
            Fs, Qs = discretise_linear_sde(ts, ts[0])
            rnds = jax.random.normal(key, (ts.shape[0], x0.shape[0]))
            return Fs[:, None] * x0[None, :] + jnp.sqrt(Qs)[:, None] * rnds

    return discretise_linear_sde, cond_score_t_0, simulate_cond_forward


def make_linear_sde_score_matching_loss(sde: LinearSDE, nn_score,
                                        t0=0., T=2., nsteps: int = 100,
                                        random_times: bool = True,
                                        keep_path: bool = True):
    discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

    def score_scale(t, s):
        return discretise_linear_sde(t, s)[1]

    def loss_fn(param, key, x0s):
        nsamples = x0s.shape[0]
        key_ts, key_fwd = jax.random.split(key, num=2)

        if random_times:
            ts = jnp.hstack([t0,
                             jnp.sort(jax.random.uniform(key_ts, (nsteps - 1,), minval=t0, maxval=T)),
                             T])
        else:
            ts = jnp.linspace(t0, T, nsteps + 1)
        scales = jax.vmap(score_scale, in_axes=[0, 0])(ts[1:], ts[:-1])

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, x0s, ts)  # (n, nsteps + 1, d)
        nn_evals = jax.vmap(nn_score,
                            in_axes=[1, 0, None],
                            out_axes=1)(fwd_paths[:, 1:], ts[1:], param)  # (n, nsteps, d)
        cond_score_evals = jax.vmap(cond_score_t_0,
                                    in_axes=[1, 0, None, None],
                                    out_axes=1)(fwd_paths[:, 1:], ts[1:], fwd_paths[:, 0], ts[0])  # (n, nsteps, d)
        return jnp.mean(jnp.mean((nn_evals - cond_score_evals) ** 2, axis=-1) * scales[None, :])

    return loss_fn


def make_ou_score_matching_loss(a, b, nn_score, t0=0., T=2., nsteps: int = 100, random_times: bool = True):
    discretise_ou_sde, cond_score_t_0, simulate_cond_forward = make_ou_sde(a, b)

    def score_scale(t):
        return discretise_ou_sde(t)[1]

    def loss_fn(param, key, x0s):
        nsamples = x0s.shape[0]
        key_ts, key_fwd = jax.random.split(key, num=2)

        if random_times:
            ts = jnp.hstack([t0,
                             jnp.sort(jax.random.uniform(key_ts, (nsteps - 1,), minval=t0, maxval=T)),
                             T])
        else:
            ts = jnp.linspace(t0, T, nsteps + 1)
        scales = score_scale(ts[1:])

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, x0s, ts)
        nn_evals = jax.vmap(nn_score, in_axes=[1, 0, None], out_axes=1)(fwd_paths[:, 1:], ts[1:], param)
        cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
                                             in_axes=[0, 0, None]),
                                    in_axes=[0, None, 0])(fwd_paths[:, 1:], ts[1:], fwd_paths[:, 0])
        return jnp.mean(jnp.mean((nn_evals - cond_score_evals) ** 2, axis=-1) * scales[None, :])

    return loss_fn


def make_ou_ipf_loss(a, b, bwd_fn, t0=0., T=2., nsteps: int = 100, random_times: bool = True):
    discretise_ou_sde, cond_score_t_0, simulate_cond_forward = make_ou_sde(a, b)

    def fwd_fn(x, k, ts):
        pass

    def loss_fn(param, key, x0s):
        nsamples = x0s.shape[0]
        key_ts, key_fwd = jax.random.split(key, num=2)

        if random_times:
            ts = jnp.hstack([t0,
                             jnp.sort(jax.random.uniform(key_ts, (nsteps - 1,), minval=t0, maxval=T)),
                             T])
        else:
            ts = jnp.linspace(t0, T, nsteps + 1)

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, x0s, ts)  # (n, nsteps + 1, d)
