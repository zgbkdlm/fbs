"""Script generating the illustration of the forward-backward algorithm.
Adrien
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve
import jax
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

T = 25
N = 250 - 1
dt = T / N
KEY = jax.random.PRNGKey(1234)
FWD_KEY, BWD_KEY = jax.random.split(KEY)
M0 = np.array([-1., 1.])
P0 = np.array([[1.5, 0.4],
               [0.4, 1.]])

Y0 = 0.5
M0_cond_Y0 = M0[0] + P0[0, 1] / P0[1, 1] * (Y0 - M0[1])
P0_cond_Y0 = P0[0, 0] - P0[0, 1] ** 2 / P0[1, 1]

A = np.exp(-dt) * np.eye(2)
B = np.eye(2) - A ** 2
C = np.exp(dt) * np.eye(2)


def logp(xy, t):
    A_t = jnp.exp(-t)
    m = jnp.exp(-t) * M0
    P = P0 * A_t ** 2 + jnp.eye(2) * (1 - A_t ** 2)
    return -0.5 * ((xy - m).T @ solve(P, xy - m, assume_a="pos"))


def score(xy, t):
    return jax.grad(logp, argnums=0)(xy, t)


def cond_fwd(key):
    fwd_keys = jax.random.split(key, N + 1)

    def body(carry, key_t):
        t, y = carry
        y_next = A[1, 1] * y + B[1, 1] ** 0.5 * jax.random.normal(key_t, shape=())
        return (t + dt, y_next), y

    (t_fin, _), ys = jax.lax.scan(body, (0., Y0), fwd_keys)
    return ys


def cond_bwd(key, ys, use_known_dyn=False):
    vs = ys[::-1]

    bwd_keys = jax.random.split(key, N + 2)
    u0 = jax.random.normal(bwd_keys[0], shape=())

    def body(carry, inp):
        key_t, v_t = inp
        u, t = carry
        uv = jnp.array([u, v_t])

        cond_score = score(uv, T - t)

        if use_known_dyn:
            eps = B[0, 0] ** 0.5 * jax.random.normal(key_t, shape=())
            eps += 2 * cond_score[0] * dt
            u_next = C[0, 0] * u + eps
        else:
            eps = np.sqrt(2) * dt ** 0.5 * jax.random.normal(key_t, shape=())
            eps += 2 * cond_score[0] * dt
            # Semi-explicit Euler given the score and Gaussian noise
            u_next = (u + 0.5 * dt * u + eps) / (1 - 0.5 * dt)
        return (u_next, t + dt), u

    _, us = jax.lax.scan(body, (u0, 0.), (bwd_keys[1:], vs))
    return us


def bwd(key):
    bwd_keys = jax.random.split(key, N + 2)
    uv0 = jax.random.normal(bwd_keys[0], shape=(2,))

    def body(carry, key_t):
        t, uv_t = carry
        score_val = score(uv_t, t)
        eps = np.sqrt(2) * dt ** 0.5 * jax.random.normal(key_t, shape=(2,))
        uv_next = uv_t + dt * uv_t + eps + 2 * score_val * dt
        return (t - dt, uv_next), uv_t

    (t_fin, _), uvs = jax.lax.scan(body, (T, uv0), bwd_keys[1:])
    return uvs[-1]


TEST_KEYS = jax.random.split(KEY, 50_000)

XS_YS_test = jax.vmap(bwd)(TEST_KEYS)
print(np.mean(XS_YS_test, 0))
print(np.cov(XS_YS_test, rowvar=False))

YS = cond_fwd(FWD_KEY)
US = cond_bwd(BWD_KEY, YS)


def fwd_bwd(key, use_known_dyn=False):
    fwd_key, bwd_key = jax.random.split(key)
    ys = cond_fwd(fwd_key)
    us = cond_bwd(bwd_key, ys, use_known_dyn)
    return us[-1]


XS_cond_known = jax.vmap(fwd_bwd, in_axes=[0, None])(TEST_KEYS, True)
XS_cond_not_known = jax.vmap(fwd_bwd, in_axes=[0, None])(TEST_KEYS, False)

print(XS_cond_known.mean(), XS_cond_known.std())
print(XS_cond_not_known.mean(), XS_cond_not_known.std())
print()
print(M0_cond_Y0, P0_cond_Y0 ** 0.5)

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Recovered conditional distribution vs true")
XS_cond = np.sort(XS_cond_known)
ax.hist(XS_cond, alpha=0.5, bins=100, density=True, label="X|Y=y")
ax.plot(XS_cond, norm.pdf(XS_cond, loc=M0_cond_Y0, scale=P0_cond_Y0 ** 0.5), color="red")
plt.show()

fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
axes[0].plot(np.linspace(0, T, N + 1), YS, label="Y")
axes[1].plot(np.linspace(0, T, N + 1), US, label="U")
axes[0].legend()
axes[1].legend()
plt.show()
