import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey


def make_ou_sde(a, b):
    """Independent OU SDEs of the form `dX = a X dt + b dW`.
    """

    def discretise_ou_sde(t):
        return jnp.exp(a * t), b ** 2 / (2 * a) * (jnp.exp(2 * a * t) - 1)

    def cond_score_t_0(x: JArray, t, x0: JArray):
        F, Q = discretise_ou_sde(t)
        return -(x - F * x0) / Q

    def simulate_cond_forward(key: JKey, x0: JArray, ts: JArray) -> JArray:
        """Simulate a path of the OU process at ts starting from x0.

        Parameters
        ----------
        key : JKey
        x0 : JArray (d, )
        ts : JArray (nsteps + 1, )
            t_0, t_1, ..., t_nsteps.

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

        dts = jnp.diff(ts)
        rnds = jax.random.normal(key, (dts.shape[0], x0.shape[0]))
        return jnp.concatenate([x0[None, :], jax.lax.scan(scan_body, x0, (dts, rnds))[1]], axis=0)
        # Fs, _ = discretise_ou_sde(ts)
        # dts = jnp.diff(ts, prepend=ts[None, 0])
        # _, Qs = discretise_ou_sde(dts)
        # return Fs * x0 + jnp.cumsum(jnp.sqrt(Qs) * jax.random.normal(key, Qs.shape))

    return discretise_ou_sde, cond_score_t_0, simulate_cond_forward


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
        nn_evals = jax.vmap(jax.vmap(nn_score,
                                     in_axes=[0, 0, None]),
                            in_axes=[0, None, None])(fwd_paths[:, 1:], ts[1:], param)
        cond_score_evals = jax.vmap(jax.vmap(cond_score_t_0,
                                             in_axes=[0, 0, None]),
                                    in_axes=[0, None, 0])(fwd_paths[:, 1:], ts[1:], fwd_paths[:, 0])
        return jnp.mean(jnp.sum((nn_evals - cond_score_evals) ** 2, axis=-1) * scales[None, :])

    return loss_fn
