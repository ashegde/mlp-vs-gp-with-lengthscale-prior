# mlp-vs-gp-with-lengthscale-prior

In this experiment, we fit a simple $d$-dimensional cubic with an MLP and two different Gaussian process regression models. Specifically, we use the GPytorch default settings and the dimension-dependent prior strategy proposed in:

Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization Performs Great in High Dimension. arXiv preprint arXiv:2402.02229.

I also found the following blog post helpful:

https://www.miguelgondu.com/blogposts/2024-03-16/when-does-vanilla-gpr-fail/

Note that in our example, the training dataset is quite small, but still contains more samples than dimensions. The example illustrates that GPs with dimension-dependent priors on the lengthscales can be advantageous in the $d=400$ setting.



![parity-2000_samples-400_dimensions](https://github.com/user-attachments/assets/9f3fae67-d372-45ea-be5a-5950a3fa618f)
