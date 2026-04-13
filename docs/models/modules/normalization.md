# Normalization Layers

## About

The `d9d.module.block.normalization` module implements memory-efficient normalization layers.

## Features

### RMSNorm

`RMSNorm` implements [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).

Uses an efficient custom Triton kernel for forward and backward passes.

It includes native support for zero-centered scaling weights.

#### Kernel Benchmarks (BF16, H100)

**Forward, Hidden Size = 128**

![](./benchmark/rms_norm/rms_norm_forward_N128.png)

**Forward, Hidden Size = 256**

![](./benchmark/rms_norm/rms_norm_forward_N256.png)

**Forward, Hidden Size = 1024**

![](./benchmark/rms_norm/rms_norm_forward_N1024.png)

**Forward, Hidden Size = 4096**

![](./benchmark/rms_norm/rms_norm_forward_N4096.png)

**Forward, Hidden Size = 7168**

![](./benchmark/rms_norm/rms_norm_forward_N7168.png)

**Backward, Hidden Size = 128**

![](./benchmark/rms_norm/rms_norm_backward_N128.png)

**Backward, Hidden Size = 256**

![](./benchmark/rms_norm/rms_norm_backward_N256.png)

**Backward, Hidden Size = 1024**

![](./benchmark/rms_norm/rms_norm_backward_N1024.png)

**Backward, Hidden Size = 4096**

![](./benchmark/rms_norm/rms_norm_backward_N4096.png)

**Backward, Hidden Size = 7168**

![](./benchmark/rms_norm/rms_norm_backward_N7168.png)

::: d9d.module.block.normalization
