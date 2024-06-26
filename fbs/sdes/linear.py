import jax
import jax.numpy as jnp
from fbs.utils import sqrtm
from fbs.typings import JArray, JKey, FloatScalar
from typing import Callable, Tuple
from functools import partial


class LinearSDE:
    pass


class StationaryConstLinearSDE(LinearSDE):
    """dX(t) = a X(t) dt + b dW(t), where `b^2 / a = 2 sigma^2`.
    """
    a: FloatScalar
    b: FloatScalar

    def __init__(self, a: FloatScalar, b: FloatScalar):
        self.a, self.b = a, b

    def drift(self, x, t):
        return self.a * x

    def dispersion(self, t):
        return self.b

    def mean(self, t, s, m0):
        return m0 * jnp.exp(self.a * (t - s))

    def variance(self, t, s):
        """Marginal variance.
        """
        return self.b ** 2 / (2 * self.a) * (jnp.exp(2 * self.a * (t - s)) - 1)

    def bridge_drift(self, x, t, target, T):
        """The corresponding Doobs transform SDE drift.
        """

        def log_h(a, b): return jnp.sum(jax.scipy.stats.norm.logpdf(a,
                                                                    self.mean(T, t, b),
                                                                    jnp.sqrt(self.variance(T, t))))

        score_h = jax.grad(log_h, argnums=1)(target, x)
        return self.drift(x, t) + self.dispersion(t) ** 2 * score_h


class StationaryLinLinearSDE(LinearSDE):
    r"""dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t), where
    `\beta(t) = (beta_max - beta_min) / (T - t0) t + (beta_min T - beta_max t0) / (T - t0)`
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

    def mean(self, t, s, m0):
        return m0 * jnp.exp(-0.5 * self.beta_integral(t, s))

    def variance(self, t, s):
        """Marginal variance.
        """
        return 1 - jnp.exp(-self.beta_integral(t, s))

    def bridge_drift(self, x, t, target, T):
        """The corresponding Doobs transform SDE drift.
        """

        def log_h(a, b): return jnp.sum(jax.scipy.stats.norm.logpdf(a,
                                                                    self.mean(T, t, b),
                                                                    jnp.sqrt(self.variance(T, t))))

        score_h = jax.grad(log_h, argnums=1)(target, x)
        return self.drift(x, t) + self.dispersion(t) ** 2 * score_h


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
    """Generate functions (e.g., discretisation and simulator) of linear SDEs that we use for experiments.
    """

    def discretise_linear_sde(t, s):
        """Discretisation of linear SDEs of the form dX(t) = a(t) X(t) dt + b(t) dW(t).
        """
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

    def simulate_cond_forward(key: JKey, x0: JArray, ts: JArray, t0: float = None, keep_path: bool = True) -> JArray:
        """Simulate a path of the OU process at ts starting from x0.

        Parameters
        ----------
        key : JKey
        x0 : JArray (..., )
        ts : JArray (nsteps + 1, )
            t_0, t_1, ..., t_nsteps.
        t0: float, default=None
            The initial time. If None, the initial time is given by ts[0].
        keep_path : bool, default=True
            Let it be true will make the returned sample a valid sample path from the SDE. Otherwise, the return is
            independent samples at each time point marginally.

        Returns
        -------
        JArray (nsteps + 1, ...), JArray (nsteps, ...)
            X_0, X_1, ..., X_nsteps.
        """

        def scan_body(carry, elem):
            x = carry
            t, t_prev, rnd = elem

            F, Q = discretise_linear_sde(t, t_prev)
            x = F * x + jnp.sqrt(Q) * rnd
            return x, x

        if keep_path:
            rnds = jax.random.normal(key, (ts.shape[0] - 1, *x0.shape))
            return jnp.concatenate([x0[jnp.newaxis], jax.lax.scan(scan_body, x0, (ts[1:], ts[:-1], rnds))[1]], axis=0)
        else:
            Fs, Qs = discretise_linear_sde(ts, t0)
            rnds = jax.random.normal(key, (*ts.shape, *x0.shape))
            return Fs * x0 + jnp.sqrt(Qs) * rnds

    return discretise_linear_sde, cond_score_t_0, simulate_cond_forward


def make_linear_sde_law_loss(sde: LinearSDE,
                             nn_fn,
                             t0=0., T=2.,
                             nsteps: int = 100,
                             random_times: bool = True,
                             loss_type: str = 'score',
                             save_mem: bool = False) -> Callable:
    """Generate the loss function to learn the backward-time SDE.

    Parameters
    ----------
    sde : LinearSDE
        A LinearSDE instance.
    nn_fn : Callable (..., d), (), (p, ) -> (..., )
        A neural network function that takes, x, t, and parameter as input, and outputs the approximate score.
    t0 : float
        The initial time.
    T : float
        The terminal time.
    nsteps : int
        The number of time steps (at the training time).
    random_times : bool, default=True
        Whether randomise the times for training.
    loss_type : str, default='score'
        What type of loss function to use. Default is the classical denoising score matching loss.
    save_mem : bool, default=False
        Enabling this option will restrict the number of time steps and batch size to be the same, to save memory cost.

    Returns
    -------
    Callable
        The loss function.
    """
    discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)
    eps = 1e-5  # the minimum I can get for numerical stability when using float32

    def score_scale(t, s):
        return discretise_linear_sde(t, s)[1]

    def loss_fn(param, key, x0s):
        nsamples = x0s.shape[0]
        state_shape = x0s.shape[1:]
        key_ts, key_fwd = jax.random.split(key, num=2)

        if random_times:
            ts = jnp.hstack([t0,
                             jnp.sort(jax.random.uniform(key_ts, (nsteps - 1,), minval=t0 + eps, maxval=T)),
                             T])
        else:
            ts = jnp.linspace(t0, T, nsteps + 1)
        scales = score_scale(ts[1:], ts[0])

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(partial(simulate_cond_forward, keep_path=True),
                             in_axes=[0, 0, None])(keys, x0s, ts)  # (n, nsteps + 1, ...)
        nn_evals = jax.vmap(nn_fn,
                            in_axes=[1, 0, None],
                            out_axes=1)(fwd_paths[:, 1:], ts[1:], param)  # (n, nsteps, ...)

        if loss_type == 'score':
            cond_score_evals = jax.vmap(cond_score_t_0,
                                        in_axes=[1, 0, None, None],
                                        out_axes=1)(fwd_paths[:, 1:], ts[1:], fwd_paths[:, 0], ts[0])  # (n, nsteps,...)
            return jnp.mean(jnp.mean((nn_evals - cond_score_evals) ** 2,
                                     axis=list(range(2, 2 + len(state_shape)))) * scales[None, :])
        elif loss_type == 'ipf':
            @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            def fwd_transition(x, t, s):
                return discretise_linear_sde(t, s)[0] * x

            fwd_evals1 = fwd_transition(fwd_paths[:, :-1], ts[1:], ts[:-1])
            fwd_evals2 = fwd_transition(fwd_paths[:, 1:], ts[1:], ts[:-1])
            return jnp.mean((nn_evals - (fwd_paths[:, 1:] + fwd_evals1 - fwd_evals2)) ** 2)
        elif loss_type == 'ipf-score':
            # @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            # def f(x, t, t_prev):
            #     return discretise_linear_sde(t, t_prev)[0] * x
            #
            # return jnp.mean(((nn_evals - f(fwd_paths[:, :-1], ts[1:], ts[:-1])) * (ts[None, 1:, None] - ts[None, :-1, None])
            #                  + fwd_paths[:, 1:] - fwd_paths[:, :-1]) ** 2)
            # return jnp.mean((nn_evals - (f(fwd_paths[:, :-1], ts[1:], ts[:-1]) - fwd_paths[:, 1:]) / (
            #             ts[None, 1:, None] - ts[None, :-1, None])) ** 2)
            cond_score_evals = jax.vmap(cond_score_t_0,
                                        in_axes=[1, 0, 1, 0],
                                        out_axes=1)(fwd_paths[:, 1:], ts[1:], fwd_paths[:, :-1], ts[:-1])
            return jnp.mean((nn_evals - cond_score_evals) ** 2)
        else:
            raise NotImplementedError(f'Loss {loss_type} not implemented.')

    def loss_fn_save_mem(param, key, x0s):
        nsamples = x0s.shape[0]
        state_shape = x0s.shape[1:]
        key_ts, key_fwd = jax.random.split(key, num=2)

        if random_times:
            ts = jnp.hstack([jnp.sort(jax.random.uniform(key_ts, (nsamples - 1,), minval=t0 + eps, maxval=T)),
                             T])  # (n, )
        else:
            dt = (T - t0) / nsamples
            ts = jnp.linspace(t0 + dt, T, nsamples)
        scales = score_scale(ts, t0)

        keys = jax.random.split(key_fwd, num=nsamples)
        fwd_paths = jax.vmap(partial(simulate_cond_forward, t0=t0, keep_path=False),
                             in_axes=[0, 0, 0])(keys, x0s, ts)  # (n, ...)
        nn_evals = nn_fn(fwd_paths, ts, param)  # (n, ...)

        if loss_type == 'score':
            cond_score_evals = jax.vmap(cond_score_t_0, in_axes=[0, 0, 0, None])(fwd_paths, ts, x0s, t0)  # (n, ...)
            return jnp.mean(jnp.mean((nn_evals - cond_score_evals) ** 2,
                                     axis=list(range(1, 1 + len(state_shape)))) * scales)
        elif loss_type == 'ipf':
            @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            def fwd_transition(x, t, s):
                return discretise_linear_sde(t, s)[0] * x

            fwd_evals1 = fwd_transition(fwd_paths[:, :-1], ts[1:], ts[:-1])
            fwd_evals2 = fwd_transition(fwd_paths[:, 1:], ts[1:], ts[:-1])
            return jnp.mean((nn_evals - (fwd_paths[:, 1:] + fwd_evals1 - fwd_evals2)) ** 2)
        elif loss_type == 'ipf-score':
            # @partial(jax.vmap, in_axes=[1, 0, 0], out_axes=1)
            # def f(x, t, t_prev):
            #     return discretise_linear_sde(t, t_prev)[0] * x
            #
            # return jnp.mean(((nn_evals - f(fwd_paths[:, :-1], ts[1:], ts[:-1])) * (ts[None, 1:, None] - ts[None, :-1, None])
            #                  + fwd_paths[:, 1:] - fwd_paths[:, :-1]) ** 2)
            # return jnp.mean((nn_evals - (f(fwd_paths[:, :-1], ts[1:], ts[:-1]) - fwd_paths[:, 1:]) / (
            #             ts[None, 1:, None] - ts[None, :-1, None])) ** 2)
            cond_score_evals = jax.vmap(cond_score_t_0,
                                        in_axes=[1, 0, 1, 0],
                                        out_axes=1)(fwd_paths[:, 1:], ts[1:], fwd_paths[:, :-1], ts[:-1])
            return jnp.mean((nn_evals - cond_score_evals) ** 2)
        else:
            raise NotImplementedError(f'Loss {loss_type} not implemented.')

    return loss_fn_save_mem if save_mem else loss_fn


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


def make_gaussian_bw_sb(mean0, cov0, mean1, cov1, sig: float = 1.) -> Tuple[Callable, Callable, Callable]:
    """Generate a Gaussian Schrodinger bridge with a Brownian motion reference at time interval [0, 1].

    Parameters
    ----------
    mean0 : JArray (d, )
        The mean of the initial Gaussian distribution.
    cov0 : JArray (d, d)
        The covariance of the initial Gaussian distribution.
    mean1 : JArray (d, )
        The mean of the terminal Gaussian distribution.
    cov1 : JArray (d, d)
        The covariance of the terminal Gaussian distribution.
    sig : float, default=1.
        The Brownian motion's diffusion coefficient.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        The marginal mean, marginal covariance, and drift functions.

    Notes
    -----
    Table 1, The Schrödinger Bridge between Gaussian Measures has a Closed Form, 2023.
    """
    d = mean0.shape[0]
    sqrt0 = sqrtm(cov0)

    D_sig = sqrtm(4 * sqrt0 @ cov1 @ sqrt0 + sig ** 4 * jnp.eye(d))
    C_sig = 0.5 * (sqrt0 @ jnp.linalg.solve(sqrt0.T, D_sig.T).T - sig ** 2 * jnp.eye(d))

    def kappa(t, _):
        return t * sig ** 2

    def r(t):
        return t

    def r_bar(t):
        return 1 - t

    def rho(t):
        return t

    def marginal_mean(t):
        return r_bar(t) * mean0 + r(t) * mean1

    def marginal_cov(t):
        return r_bar(t) ** 2 * cov0 + r(t) ** 2 * cov1 + r(t) * r_bar(t) * (C_sig + C_sig.T) + kappa(t, t) * (
                1 - rho(t)) * jnp.eye(d)

    def s(t):
        pt = r(t) * cov1 + r_bar(t) * C_sig
        qt = r_bar(t) * cov0 + r(t) * C_sig
        return pt - qt.T - sig ** 2 * rho(t) * jnp.eye(d)

    def drift(x, t):
        mt = marginal_mean(t)
        chol_t = jax.scipy.linalg.cho_factor(marginal_cov(t))
        return s(t).T @ jax.scipy.linalg.cho_solve(chol_t, x - mt) - mean0 + mean1

    return marginal_mean, marginal_cov, drift
