"""
Test particle MCMC for the conditional sampling of a Gaussian model.
"""
import jax
import jax.numpy as jnp
import math
import numpy as np
import numpy.testing as npt
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE, StationaryLinLinearSDE
from fbs.utils import discretise_lti_sde
from fbs.samplers.smc import pmcmc_kernel, pcn_proposal
from fbs.samplers import stratified
from functools import partial

jax.config.update("jax_enable_x64", True)


def test_pcn_proposal():
    T = 2
    nsteps = 500
    ts = jnp.linspace(0, T, nsteps + 1)

    for sde in (StationaryConstLinearSDE(a=-0.5, b=1.),
                StationaryLinLinearSDE(beta_min=0.02, beta_max=5., t0=0., T=T)):

        discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

        delta = 5.  # Check it
        y0 = jnp.array(2.)

        def fwd_sampler(key_):
            return simulate_cond_forward(key_, y0, ts)

        def proposal(key_, ys):
            return pcn_proposal(key_, delta, ys, sde.mean(ts, ts[0], y0), fwd_sampler)

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num=10000)
        yss = jax.vmap(fwd_sampler)(keys)

        key, _ = jax.random.split(key)
        keys = jax.random.split(key, num=10000)
        prop_yss = jax.vmap(proposal, in_axes=[0, 0])(keys, yss)

        npt.assert_allclose(jnp.mean(prop_yss, axis=0), jnp.mean(yss, axis=0), rtol=5e-2)
        npt.assert_allclose(jnp.var(prop_yss, axis=0), jnp.var(yss, axis=0), rtol=5e-2)


def test_pmcmc():
    key = jax.random.PRNGKey(666)

    nparticles = 100
    nsamples = 1000
    burn_in = 100

    T = 3
    nsteps = 1000
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    # Target distribution \pi(x, y)
    m0 = jnp.array([1., -1.])
    cov0 = jnp.array([[2., 0.5],
                      [0.5, 1.2]])

    y0 = jnp.array(4.)
    true_cond_m = m0[0] + cov0[0, 1] / cov0[1, 1] * (y0 - m0[1])
    true_cond_var = cov0[0, 0] - cov0[0, 1] ** 2 / cov0[1, 1]

    A = -0.5 * jnp.eye(2)
    B = jnp.array([[3., 0.],
                   [0., 0.1]])
    gamma = B @ B.T

    def forward_m_cov(t):
        F, Q = discretise_lti_sde(A, gamma, t)
        return F @ m0, F @ cov0 @ F.T + Q

    def score(z, t):
        mt, covt = forward_m_cov(t)
        return jax.grad(jax.scipy.stats.multivariate_normal.logpdf, argnums=0)(z, mt, covt)

    def simulate_forward(xy0, key_):
        def scan_body(carry, elem):
            xy = carry
            dw = elem

            xy = F_ @ xy + chol @ dw
            return xy, xy

        F_, Q_ = discretise_lti_sde(A, gamma, dt)
        chol = jnp.linalg.cholesky(Q_)
        dws = jnp.sqrt(dt) * jax.random.normal(key_, (nsteps, 2))
        return jnp.concatenate([xy0[None, :], jax.lax.scan(scan_body, xy0, dws)[1]], axis=0)

    # Reference distribution \pi_ref
    m_ref, cov_ref = forward_m_cov(T)

    # The reverse process
    def reverse_drift(uv, t):
        return -A @ uv + gamma @ score(uv, T - t)

    def reverse_drift_u(u, v, t):
        uv = jnp.asarray([u, v])
        return (-A @ uv + gamma @ score(uv, T - t))[0]

    def reverse_drift_v(v, u, t):
        uv = jnp.asarray([u, v])
        return (-A @ uv + gamma @ score(uv, T - t))[1]

    key, subkey = jax.random.split(key)
    uv0s = m_ref + jax.random.normal(subkey, (10_000, 2)) @ jnp.linalg.cholesky(cov_ref)

    def backward_euler(uv0, key_):
        def scan_body(carry, elem):
            uv = carry
            dw, t = elem

            uv += reverse_drift(uv, t) * dt + B @ dw
            return uv, None

        _, subkey_ = jax.random.split(key_)
        dws = jnp.sqrt(dt) * jax.random.normal(subkey_, (nsteps, 2))
        return jax.lax.scan(scan_body, uv0, (dws, ts[:-1]))[0]

    key, subkey = jax.random.split(key)
    # keys = jax.random.split(subkey, num=10_000)
    # approx_init_samples = jax.vmap(backward_euler, in_axes=[0, 0])(uv0s, keys)
    #
    # # First test if the backward is implemented correctly
    # npt.assert_allclose(jnp.mean(approx_init_samples, axis=0), m0, rtol=1e-1)
    # npt.assert_allclose(jnp.cov(approx_init_samples, rowvar=False), cov0, rtol=1e-1)

    # Now pMCMC for the conditional sampling
    def transition_sampler(us, v, t, key_):
        return (us + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us, v, t) * dt
                + math.sqrt(dt) * B[0, 0] * jax.random.normal(key_, us.shape))

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def likelihood_logpdf(v, u_prev, v_prev, t_prev):
        cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
        return jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * B[1, 1])

    def init_sampler(key_, nsamples_):
        return (m_ref[0] + jnp.sqrt(cov_ref[0, 0]) * jax.random.normal(key_)) * jnp.ones((nsamples_,))

    def ref_logpdf(x):
        return jax.scipy.stats.norm.logpdf(x, m_ref[0], jnp.sqrt(cov_ref[0, 0]))

    def fwd_ys_sampler(key_, y0_):
        xy0 = jnp.array([0., y0_])
        return simulate_forward(xy0, key_)[:, 1]

    @jax.jit
    def mcmc_kernel(subkey_, uT_, log_ell_, ys_, xT_):
        return pmcmc_kernel(subkey_, uT_, log_ell_, ys_, xT_,
                            y0, ts,
                            fwd_ys_sampler,
                            None,
                            init_sampler, ref_logpdf,
                            transition_sampler, likelihood_logpdf,
                            stratified, nparticles, delta=None)

    # Test the invariance of the MCMC kernel
    key, subkey = jax.random.split(key)
    true_samples = true_cond_m + jnp.sqrt(true_cond_var) * jax.random.normal(subkey, (nsamples,))

    key, subkey = jax.random.split(key)
    ys = fwd_ys_sampler(subkey, y0)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nsamples)
    prop_samples = jax.vmap(mcmc_kernel, in_axes=[0, 0, None, None, None])(keys,
                                                                           true_samples,
                                                                           0.,
                                                                           ys, 0.)[0]

    import matplotlib.pyplot as plt

    plt.hist(true_samples, density=True, bins=50, alpha=0.5)
    plt.hist(prop_samples, density=True, bins=50, alpha=0.5)
    plt.show()

    # # Test the MCMC loop
    # key, subkey = jax.random.split(key)
    # approx_cond_samples = np.zeros((nsamples,))
    # uT, log_ell = 0., 0.
    # ys, xT = fwd_ys_sampler(subkey, y0), m_ref[0]
    # for i in range(nsamples):
    #     key, subkey = jax.random.split(key)
    #     uT, log_ell, ys, xT, mcmc_state = mcmc_kernel(subkey, uT, log_ell, ys, xT)
    #     approx_cond_samples[i] = uT
    #
    # approx_cond_samples = approx_cond_samples[burn_in:]
    # npt.assert_allclose(jnp.mean(approx_cond_samples), true_cond_m, rtol=1e-2)
    # npt.assert_allclose(jnp.var(approx_cond_samples), true_cond_var, rtol=3e-2)
