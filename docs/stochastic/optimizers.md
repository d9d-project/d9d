---
title: Stochastic Optimizers
---

# Stochastic Optimizers

## About

When training large models in reduced precision (like BF16), standard "Round to Nearest" operations can lead to stalling. If a weight update is smaller than the smallest representable difference for a given float value, the update disappears completely.

Stochastic Rounding replaces rigid rounding with a probabilistic approach: for instance, if a value $x$ is $30\%$ of the way between representable numbers $A$ and $B$, it has a $30\%$ chance of rounding to $B$ and $70\%$ chance of rounding to $A$. Over multiple updates, the statistical expectation matches the true high-precision value, allowing training to converge even when individual updates are technically "too small" for the format.

::: d9d.optim.stochastic
    options:
        show_root_heading: true
        show_root_full_path: true
`