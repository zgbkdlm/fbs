import math
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from fbs.samplers.smc import bootstrap_filter, bootstrap_backward_smoother
from fbs.samplers.resampling import stratified
from fbs.utils import discretise_lti_sde

np.random.seed(666)
jax.config.update("jax_enable_x64", True)


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

    nsamples = 1_000
    key, subkey = jax.random.split(key)

    pf_samples, _ = bootstrap_filter(transition_sampler, measurement_cond_logpdf,
                                     ys, ts, init_sampler, subkey, nsamples, stratified,
                                     log=True, return_last=False)
    pf_samples = pf_samples[3:]
    mfs = mfs[2:]
    vfs = vfs[2:]

    npt.assert_allclose(jnp.mean(pf_samples, axis=1), mfs, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(jnp.var(pf_samples, axis=1), vfs, rtol=1e-1, atol=1e-1)


def test_particle_smoother():
    def gp_cov(t1, t2):
        return sigma ** 2 * jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell)

    ell, sigma = 1., 1.
    a, b = -1 / ell, math.sqrt(2 / ell) * sigma

    T = 1
    nsteps = 100
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    F, Q = discretise_lti_sde(a * jnp.eye(1), b ** 2 * jnp.eye(1), dt)
    F, Q = jnp.squeeze(F), jnp.squeeze(Q)
    chol_Q = jnp.sqrt(Q)
    R = 1.

    key = jax.random.PRNGKey(666)
    xs = jnp.linalg.cholesky(gp_cov(ts, ts)) @ jax.random.normal(key, (nsteps + 1,))

    key, subkey = jax.random.split(key)
    ys = xs + math.sqrt(R) * jax.random.normal(subkey, (nsteps + 1,))

    # Solve GP regression
    cov_ = gp_cov(ts, ts)
    gain = cov_ + R * jnp.eye(nsteps + 1)
    chol_gain = jax.scipy.linalg.cho_factor(gain)
    posterior_mean = cov_ @ jax.scipy.linalg.cho_solve(chol_gain, ys)
    posterior_cov = cov_ - cov_ @ jax.scipy.linalg.cho_solve(chol_gain, cov_)

    def init_sampler(key_, _, nsamples_):
        return posterior_mean[0] + jnp.sqrt(posterior_cov[0, 0]) * jax.random.normal(key_, (nsamples_,))

    def transition_sampler(xs_prev, v_prev, t_prev, key_):
        return xs_prev * F + jax.random.normal(key_, xs_prev.shape) * chol_Q

    def transition_logpdf(x, x_prev, v_pref, t_prev):
        return jax.scipy.stats.norm.logpdf(x, x_prev * F, chol_Q)

    def likelihood_logpdf(y, x_prev, y_prev, t_prev):
        return jax.scipy.stats.norm.logpdf(y, x_prev, math.sqrt(R))

    key, subkey = jax.random.split(key)
    filtering_samples = bootstrap_filter(transition_sampler, likelihood_logpdf, ys, ts,
                                         init_sampler, subkey, 10_000, stratified,
                                         log=True, return_last=False)[0]

    def smoother(key_):
        return bootstrap_backward_smoother(key_, filtering_samples, ys, ts, transition_logpdf)

    key, subkey = jax.random.split(key)
    trajs = jax.vmap(smoother)(jax.random.split(subkey, 1000))

    npt.assert_allclose(jnp.mean(trajs, axis=0), posterior_mean, rtol=2e-1)
