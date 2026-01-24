---
title: Stochastic Rounding
---

# Stochastic Rounding

## About

Standard floating-point casting (e.g., `tensor.to(torch.bfloat16)`) typically utilizes **Round-to-Nearest-Even**. 
This method is statistically biased.

**Stochastic Rounding** solves this by rounding probabilistically based on the distance to the nearest representable numbers. This ensures that the expected value of the rounded result equals the original value: $E[Round(x)] = x$.

This module provides highly optimized **Triton** kernels for performing stochastic casting.

## Benchmarks

### copy_fp32_to_bf16_stochastic_

![](./benchmark/copy_fp32_to_bf16_stochastic_.png)

::: d9d.kernel.stochastic
    options:
        show_root_heading: true
        show_root_full_path: true
