# Installation

## About

This page shows how to install d9d from PyPI or from a checkout of the repository. Some features need extras: optional dependencies that you install with `d9d[<extra>]`.

## From PyPI

=== "pip"

    ```bash
    pip install d9d
    ```

=== "poetry"

    ```bash
    poetry add d9d
    ```

=== "uv"

    ```bash
    uv add d9d
    ```

## From a Checkout

The `main` branch can be ahead of the latest release. To get these changes, install d9d from a checkout:

```bash
git clone https://github.com/d9d-project/d9d.git
cd d9d
pip install -e .
```

To work on d9d itself, follow the development setup in [CONTRIBUTING.md](https://github.com/d9d-project/d9d/blob/main/CONTRIBUTING.md).

## Extras

Install an extra with its name in brackets. The quotes keep the shell from expanding the brackets:

```bash
pip install "d9d[aim]"
pip install -e ".[aim]"  # From a checkout.
```

The extras are:

*   `d9d[aim]`: The [Aim](https://aimstack.io/) [experiment tracker](../tracker/index.md).
*   `d9d[visualization]`: [Plotly](https://plotly.com/python/), for [plotting learning rate schedules](../lr_scheduler/visualization.md).
*   `d9d[linear-attention]`: [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) kernels for [linear attention](../models/modules/attention.md#linear-attention).
*   `d9d[backend-sdpa-flash-attention-2]`: [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) kernels as an [SDPA backend](../models/modules/attention.md#scaled-dot-product-attention-backends). You may need to build FlashAttention 2 from source if no prebuilt wheel matches your setup.
*   `d9d[backend-sdpa-flash-attention-4]`: [FlashAttention 4](https://github.com/Dao-AILab/flash-attention) kernels as an [SDPA backend](../models/modules/attention.md#scaled-dot-product-attention-backends).
*   `d9d[moe]`: GPU kernels for [Mixture-of-Experts layers](../models/modules/moe.md). You must build and install [DeepEP](https://github.com/deepseek-ai/DeepEP) and [grouped-gemm](https://github.com/fanshiqing/grouped_gemm/) manually first.
*   `d9d[cce]`: Fused cross-entropy kernels for the [causal language modelling head](../models/modules/head.md#causal-language-modelling). You must install [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy) from its repository first.
