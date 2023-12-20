# TODOs

1. Conditional sampling without the need of using a reference measure. By doing so, we are relieve from 1) exact forward 
and backward bridging between two measures, and/or 2) approximate the terminal by the reference measure. 

   1. Implement the Gibbs sampler with approximate filters, but this may endanger the Gibbs convergence, because the bias 
   accumulates. Probably bother not to implement this.
   2. Implement the MCMC within Gibbs sampler by using conditional SMC.

2. Conditional sampling with a reference measure. To do so, we need to assume that X_T | x_0 is approximately 
distributed according to pi_ref. 

   1. Implement PMCMC targeting at p(y_T, xT, x_0 | y_0).
   