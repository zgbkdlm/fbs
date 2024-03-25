"""
Generate independent random seeds for reproducibility.

Run this script under the folder `./experiments` before any experiment.
"""
import jax
import numpy as np

key = jax.random.PRNGKey(666)
keys = jax.random.split(key, num=1000)

np.save('./keys', np.asarray(keys))
