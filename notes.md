# NN training

The trained NN seems to be invariant in the time dimension which is awful. Essentially, simulating the backward using f(x, t) and f(x, T - t) gives almost identical results. 

In an isolating experiment, see `scratch_122.py`, the problem seems to be the length of times. The accuracy is much better when `nsteps` is small.