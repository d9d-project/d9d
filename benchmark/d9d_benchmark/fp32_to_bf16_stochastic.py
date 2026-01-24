import functools

import torch
import triton
from d9d.kernel.stochastic import copy_fp32_to_bf16_stochastic_


def copy_fp32_to_bf16_stochastic_torch_(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    val_i32 = source.view(torch.int32)
    noise = torch.randint(
        0, 65536, source.shape,
        device=source.device,
        dtype=torch.int32
    )
    val_noisy = val_i32 + noise
    res = (val_noisy >> 16).to(torch.int16).view(torch.bfloat16)
    target.copy_(res)
    return target


copy_fp32_to_bf16_stochastic_torch_compile_ = torch.compile(copy_fp32_to_bf16_stochastic_torch_)


_PROVIDER_TO_FN = {
    "d9d": copy_fp32_to_bf16_stochastic_,
    "torch-eager": copy_fp32_to_bf16_stochastic_torch_,
    "torch-compile": copy_fp32_to_bf16_stochastic_torch_compile_
}


def run_benchmark():
    configs = [
        triton.testing.Benchmark(
            x_names=["n_elements"],
            x_vals=[2 ** i for i in range(20, 27)],
            line_arg="provider",
            line_vals=list(_PROVIDER_TO_FN.keys()),
            line_names=list(_PROVIDER_TO_FN.keys()),
            styles=[("blue", "-"), ("red", "--"), ("green", "-")],
            ylabel="GB/s",
            plot_name="stochastic-rounding-bench",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(n_elements: int, provider: str) -> tuple[float, ...]:
        # Setup Data
        src = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        tgt = torch.empty(n_elements, device="cuda", dtype=torch.bfloat16)

        # We measure bandwidth:
        # Read FP32 (4 bytes) + Write BF16 (2 bytes) = 6 bytes per element

        quantiles = [0.5, 0.2, 0.8]
        fn = functools.partial(_PROVIDER_TO_FN[provider], tgt, src)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

        def gbps(ms: float) -> float:
            # Read FP32 (4 bytes) + Write BF16 (2 bytes) = 6 bytes per element
            return (6 * n_elements * 1e-9) / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path="..", show_plots=True, print_data=True)


if __name__ == "__main__":
    run_benchmark()
