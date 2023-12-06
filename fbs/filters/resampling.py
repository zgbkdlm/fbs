"""
The codes in this file are selected and adapted from
https://github.com/AdrienCorenflos/parallel-ps/blob/e5a24fe0ba4afbf3275bcbfbd06908ace2ea7257/parallel_ps/core/resampling.py
which is now shipped as
a module in https://github.com/blackjax-devs/blackjax/blob/2bbdefc28cc7c2048431405fe5d47a1b76a69e64/blackjax/smc/resampling.py
under the Apache-2.0 license.

Here are the changes:

1. Simplified the signatures of systematic, stratified.
2. Removed the comments of multinomial.
3. Renamed rng_key to key.
4. Changed typing and the code formatting style.

The copyright notice:

Copyright 2020 The Blackjax developers, and Adrien Corenflos

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey


def _sorted_uniforms(n, key: JKey) -> JArray:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]


def _systematic_or_stratified(weights: JArray, key: JKey, is_systematic: bool) -> JArray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(key, ())
    else:
        u = jax.random.uniform(key, (n,))
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           (jnp.arange(n, dtype=weights.dtype) + u) / n)
    return jnp.clip(idx, 0, n - 1)


def systematic(weights: JArray, key: JKey) -> JArray:
    return _systematic_or_stratified(weights, key, True)


def stratified(weights: JArray, key: JKey) -> JArray:
    return _systematic_or_stratified(weights, key, False)


def multinomial(weights: JArray, key: JKey) -> JArray:
    """Not tested.
    """
    n = weights.shape[0]
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           _sorted_uniforms(n, key))
    return jnp.clip(idx, 0, n - 1)


def _avg_n_nplusone(x):
    hx = 0.5 * x
    y = jnp.pad(hx, [[0, 1]], constant_values=0.0, mode="constant")
    y = y.at[..., 1:].add(hx)
    return y
