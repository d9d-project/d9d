from collections.abc import Sequence
from typing import Any

import torch
from d9d.kernel.normalization import rms_norm

from d9d_test.kernel.helper.benchmark import (
    BenchmarkMode,
    CommonKernelProvider,
    KernelProvider,
    TritonBenchmarkRunner,
    TritonBenchmarkSpec,
)
from d9d_test.kernel.rms_norm.reference_impl import rms_norm_liger_kernel, rms_norm_torch, rms_norm_torch_functional

_PROVIDERS = [
    KernelProvider(name=CommonKernelProvider.D9D, kernel=rms_norm),
    KernelProvider(name=CommonKernelProvider.TORCH_EAGER, kernel=rms_norm_torch),
    KernelProvider(name=CommonKernelProvider.TORCH_COMPILE, kernel=rms_norm_torch, run_torch_compile=True),
    KernelProvider(name=CommonKernelProvider.TORCH_FUNCTIONAL, kernel=rms_norm_torch_functional),
    KernelProvider(name=CommonKernelProvider.LIGER_KERNEL, kernel=rms_norm_liger_kernel),
]


class RmsNormBenchmarkSpec(TritonBenchmarkSpec):
    @property
    def x_name(self) -> str:
        return "m"

    @property
    def x_vals(self) -> Sequence[int]:
        return [512, 1024, 2048, 4096, 8192, 16384, 32768]

    @property
    def configurations(self) -> list[dict[str, Any]]:
        return [
            {"n": n}
            for n in (
                128,  # head dim qwen3 235B
                256,  # head dim qwen3.5 397B
                1024,  # bert large and some other models
                4096,  # qwen3.5 397B / qwen3 235B
                7168,  # deepseek v3.2
            )
        ]

    def build_inputs(self, m: int, n: int, **params: Any) -> dict[str, Any]:
        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.ones((n,), device="cuda", dtype=torch.bfloat16, requires_grad=True)

        return {"x": x, "weight": weight, "eps": 1e-6, "zero_centered": False}

    def total_bytes(self, mode: BenchmarkMode, m: int, n: int, **params: Any) -> int:
        base = m * n
        base *= {BenchmarkMode.forward: 4, BenchmarkMode.backward: 6}[mode]
        return base


if __name__ == "__main__":
    TritonBenchmarkRunner(
        name="rms_norm",
        providers=_PROVIDERS,
        has_backward=True,
        spec=RmsNormBenchmarkSpec(),
    ).run()
