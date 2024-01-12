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

    return discretise_ou_sde, cond_score_t_0, simulate_cond_forward
