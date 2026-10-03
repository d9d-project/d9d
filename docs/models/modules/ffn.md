# Feed Forward Networks (FFN)

## About

The `d9d.module.block.ffn` package implements the dense feed-forward networks of Transformer blocks.

## SwiGLU

`SwiGLU` implements the [SwiGLU layer](https://arxiv.org/abs/2002.05202): `down(SiLU(gate(x)) * up(x))`.

It computes `SiLU(gate(x)) * up(x)` with a fused Triton kernel, so it requires a GPU supported by Triton.

### Kernel Benchmarks (bf16, H100)

![](./benchmark/silu_mul_bf16.png)

## Usage

```python
import torch

from d9d.module.block.ffn import SwiGLU

ffn = SwiGLU(hidden_size=2048, intermediate_size=6144).cuda()

hidden_states = torch.randn(2, 16, 2048, device="cuda")
output = ffn(hidden_states)  # (2, 16, 2048)
```

## API Reference

::: d9d.module.block.ffn
