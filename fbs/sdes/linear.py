import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, FloatScalar
from functools import partial
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
    r"""dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t), where
        - `\beta(t) = (beta_max - beta_min) / (T - t0) t + (beta_min T - beta_max t0) / (T - t0)`
    """
    beta_min: FloatScalar
    beta_max: FloatScalar
    t0: FloatScalar
    T: FloatScalar

    def __init__(self, beta_min: FloatScalar, beta_max: FloatScalar, t0: FloatScalar, T: FloatScalar):
        self.beta_min, self.beta_max, self.t0, self.T = beta_min, beta_max, t0, T

    def beta(self, t):
        beta_min, beta_max, t0, T = self.beta_min, self.beta_max, self.t0, self.T
        return (beta_max - beta_min) / (T - t0) * t + (beta_min * T - beta_max * t0) / (T - t0)

    def beta_integral(self, t, s):
        beta_min, beta_max, t0, T = self.beta_min, self.beta_max, self.t0, self.T
        return 0.5 * (t - s) * ((beta_max - beta_min) / (T - t0) * (t + s)
                                + 2 * (beta_min * T - beta_max * t0) / (T - t0))

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    def dispersion(self, t):
        return jnp.sqrt(self.beta(t))


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
            r = sde.beta_integral(t, s)
            return jnp.exp(-0.5 * r), 1 - jnp.exp(-r)
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


def make_linear_sde_law_loss(sde: LinearSDE, nn_fn,
                             t0=0., T=2., nsteps: int = 100,
                             random_times: bool = True,
                             keep_path: bool = True,
                             loss_type: str = 'score',
                             save_mem: bool = False):
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
        scales = score_scale(ts[1:], ts[0])

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(simulate_cond_forward, in_axes=[0, 0, None])(keys, x0s, ts)  # (n, nsteps + 1, d)
        nn_evals = jax.vmap(nn_fn,
                            in_axes=[1, 0, None],
                            out_axes=1)(fwd_paths[:, 1:], ts[1:], param)  # (n, nsteps, d)

        if loss_type == 'score':
            cond_score_evals = jax.vmap(cond_score_t_0,
                                        in_axes=[1, 0, None, None],
                                        out_axes=1)(fwd_paths[:, 1:], ts[1:], fwd_paths[:, 0], ts[0])  # (n, nsteps, d)
            return jnp.mean(jnp.mean((nn_evals - cond_score_evals) ** 2, axis=-1) * scales[None, :])
        elif loss_type == 'ipf':
            @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            def fwd_transition(x, t, s):
                return discretise_linear_sde(t, s)[0] * x

            fwd_evals1 = fwd_transition(fwd_paths[:, :-1], ts[1:], ts[:-1])
            fwd_evals2 = fwd_transition(fwd_paths[:, 1:], ts[1:], ts[:-1])
            return jnp.mean((nn_evals - (fwd_paths[:, 1:] + fwd_evals1 - fwd_evals2)) ** 2)
        elif loss_type == 'ipf-score':
            @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            def f(x, t, t_prev):
                return x + sde.drift(x, t_prev) * (t - t_prev)

            return jnp.mean(((nn_evals - f(fwd_paths[:, :-1], ts[1:], ts[:-1])) * (ts[None, 1:, None] - ts[None, :-1, None])
                             + fwd_paths[:, 1:] - fwd_paths[:, :-1]) ** 2)
            # return jnp.mean((nn_evals - (f(fwd_paths[:, :-1], ts[1:], ts[:-1]) - fwd_paths[:, 1:]) / (
            #             ts[None, 1:, None] - ts[None, :-1, None])) ** 2)
        else:
            raise NotImplementedError(f'Loss {loss_type} not implemented.')

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
