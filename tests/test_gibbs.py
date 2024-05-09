"""
This tests the Gibbs kernel.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from fbs.samplers import gibbs_kernel as _gibbs_kernel
from fbs.sdes import make_linear_sde, StationaryConstLinearSDE
from functools import partial

jax.config.update("jax_enable_x64", False)


def test_gibbs_kernel():
    """Test if the kernel targets at p(x0, xT, y_(0:T] | y0).
    To do so, we test the marginal p(x0 | y0).
    Gaussian model
    """
    key = jax.random.PRNGKey(666)

    # Define p(x0, y0)
    m0 = jnp.array([-1., 1.])
    cov0 = jnp.array([[2., 0.4],
                      [0.4, 0.5]])
    y0 = jnp.array([0., ])

    # Compute p(x0 | y0) for testing
    true_posterior_mean = m0[0] + cov0[0, 1] / cov0[1, 1] * (y0 - m0[1])
    true_posterior_cov = cov0[0, 0] - cov0[0, 1] / cov0[1, 1] * cov0[1, 0]

    # Define the noising SDE and its backward
    T = 1.
    nsteps = 100
    dt = T / nsteps
    ts = jnp.linspace(0, T, nsteps + 1)

    sde = StationaryConstLinearSDE(a=-0.5, b=1.)
    discretise_linear_sde, cond_score_t_0, simulate_cond_forward = make_linear_sde(sde)

    def forward_m_cov(t):
        F_, Q_ = discretise_linear_sde(t, ts[0])
        return F_ * m0, F_ ** 2 * cov0 + Q_ * jnp.eye(2)

    def score(z, t):
        mt, covt = forward_m_cov(t)
        chol = jax.scipy.linalg.cho_factor(covt)
        return -jax.scipy.linalg.cho_solve(chol, z - mt)

    # Compute the terminal reference distribution
    mT, covT = forward_m_cov(T)

    # Compute the cross-covariance over the path
    def cov_fn(t, s):
        semigroup = jnp.exp(-0.5 * jnp.abs(t - s))
        return jax.lax.cond(t < s,
                            lambda _: forward_m_cov(t)[1] * semigroup,
                            lambda _: forward_m_cov(s)[1] * semigroup,
                            None)

    def unpack(xy):
        return xy[..., :1], xy[..., 1:]

    # The reverse process
    def reverse_drift(uv, t):
        return -sde.drift(uv, T - t) + sde.dispersion(T - t) ** 2 * score(uv, T - t)

    def reverse_drift_u(u, v, t):
        uv = jnp.concatenate([u, v])
        return unpack(reverse_drift(uv, t))[0]

    def reverse_drift_v(v, u, t):
        uv = jnp.concatenate([u, v])
        return unpack(reverse_drift(uv, t))[1]

    def reverse_dispersion(t):
        return sde.dispersion(T - t)

    def transition_sampler(us_prev, v_prev, t_prev, key_):
        return (us_prev + jax.vmap(reverse_drift_u, in_axes=[0, None, None])(us_prev, v_prev, t_prev) * dt
                + math.sqrt(dt) * reverse_dispersion(t_prev) * jax.random.normal(key_, us_prev.shape))

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def transition_logpdf(u, u_prev, v_prev, t_prev):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u,
                                                   u_prev + reverse_drift_u(u_prev, v_prev, t_prev) * dt,
                                                   math.sqrt(dt) * reverse_dispersion(t_prev)))

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def likelihood_logpdf(v, u_prev, v_prev, t_prev):
        cond_m = v_prev + reverse_drift_v(v_prev, u_prev, t_prev) * dt
        return jnp.sum(jax.scipy.stats.norm.logpdf(v, cond_m, math.sqrt(dt) * reverse_dispersion(t_prev)))

    def fwd_sampler(key_, x0_, y0_):
        return simulate_cond_forward(key_, jnp.concatenate([x0_, y0_]), ts)

    # Gibbs kernel
    nparticles = 10
    nsamples = 10000
    burnin = 10

    @jax.jit
    def gibbs_kernel(key_, x0_, y0_, us_star_, bs_star_):
        return _gibbs_kernel(key_, x0_, y0_, us_star_, bs_star_,
                             ts, fwd_sampler, sde, unpack, nparticles,
                             transition_sampler, transition_logpdf, likelihood_logpdf,
                             marg_y=False, explicit_backward=True, explicit_final=False)

    # Run the kernel
    x0 = jnp.array([0., ])
    us_star = jnp.zeros((nsteps + 1, 1))
    bs_stars = jnp.zeros((nsteps + 1), dtype=int)
    x0s = np.zeros((nsamples,))
    for i in range(nsamples):
        key, subkey = jax.random.split(key)
        x0, us_star, bs_stars, acc = gibbs_kernel(subkey, x0, y0, us_star, bs_stars)
        x0s[i] = x0

    x0s = x0s[burnin:]

    npt.assert_allclose(jnp.mean(x0s), true_posterior_mean, rtol=1e-2)
    npt.assert_allclose(jnp.var(x0s), true_posterior_cov, rtol=2e-2)
