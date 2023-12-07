import math
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.config import config
from fbs.filters.smc import bootstrap_filter
from fbs.filters.resampling import stratified

np.random.seed(666)
config.update("jax_enable_x64", True)


def test_particle_filter():
    """
    x_k = F x_{k-1} + y_{k-1} + q_k,
    y_k = H x_k + y_{k-1} + r_k
    """
    F, trans_var = 0.1, 0.1
    H, meas_var = 1., 1.
    y0 = 0.
    m0, v0 = y0, 1.
    key = jax.random.PRNGKey(666)
    nsteps = 20
    ts = jnp.linspace(0, 1, nsteps + 1)

    def scan_simulate(carry, elem):
        x, y = carry
        q, r = elem
        x = F * x + y + q
        y = H * x + y + r
        return (x, y), (x, y)

    key, subkey = jax.random.split(key)
    x0 = m0 + jnp.sqrt(v0) * jax.random.normal(subkey)

    key, subkey = jax.random.split(key)
    qs = jnp.sqrt(trans_var) * jax.random.normal(subkey, (nsteps,))
    key, subkey = jax.random.split(key)
    rs = jnp.sqrt(meas_var) * jax.random.normal(subkey, (nsteps,))

    _, (xs, ys) = jax.lax.scan(scan_simulate, (x0, y0), (qs, rs))
    ys = jnp.concatenate([jnp.array([y0]), ys])

    def scan_kf(carry, elem):
        mf, vf, nell = carry
        y, y_prev = elem

        mp = F * mf + y_prev
        vp = F * vf * F + trans_var

        s = vp * H ** 2 + meas_var
        gain = vp * H / s
        pred_y = H * mp + y_prev
        mf = mp + gain * (y - pred_y)
        vf = vp - vp * H * gain
        nell -= jax.scipy.stats.norm.logpdf(y, pred_y, jnp.sqrt(s))
        return (mf, vf, nell), (mf, vf)

    (*_, kf_nell), (mfs, vfs) = jax.lax.scan(scan_kf,
                                             (m0, v0, 0.),
                                             (ys[1:], ys[:-1]))

    def transition_sampler(x, y_prev, t, key_):
        return F * x + y_prev + jnp.sqrt(trans_var) * jax.random.normal(key_, (x.shape[0],))

    def measurement_cond_logpdf(y, x, y_prev, t):
        return jax.scipy.stats.norm.logpdf(y, H * x + y_prev, jnp.sqrt(meas_var))

    def measurement_cond_pdf(y, x, y_prev, t):
        return jax.scipy.stats.norm.pdf(y, H * x + y_prev, jnp.sqrt(meas_var))

    def init_sampler(key_, y, nsamples_):
        return y + jnp.sqrt(v0) * jax.random.normal(key_, (nsamples_,))

    nsamples = 10_000
    key, subkey = jax.random.split(key)

    pf_samples, pf_nell = bootstrap_filter(transition_sampler, measurement_cond_logpdf,
                                           ys, ts, init_sampler, subkey, nsamples, stratified,
                                           log=True, return_last=False)

    npt.assert_allclose(jnp.mean(pf_samples, axis=1), mfs, rtol=1e-2)
    npt.assert_allclose(jnp.var(pf_samples, axis=1), vfs, rtol=1e-1)
    npt.assert_allclose(pf_nell, kf_nell, rtol=1e-3)

    pf_samples, pf_nell = bootstrap_filter(transition_sampler, measurement_cond_pdf,
                                           ys, ts, init_sampler, subkey, nsamples, stratified,
                                           log=True, return_last=False)

    npt.assert_allclose(jnp.mean(pf_samples, axis=1), mfs, rtol=1e-2)
    npt.assert_allclose(jnp.var(pf_samples, axis=1), vfs, rtol=1e-1)
    npt.assert_allclose(pf_nell, kf_nell, rtol=1e-3)
