import functools

import torch
import triton
from d9d.kernel.swiglu import silu_mul

from d9d_test.kernel.silu_mul.reference_impl import silu_mul_torch

_PROVIDER_TO_FN = {
    "d9d": silu_mul,
    "torch-eager": silu_mul_torch,
    "torch-compile": torch.compile(silu_mul_torch),
}


def run_benchmark():
    configs = [
        triton.testing.Benchmark(
            x_names=["n_elements"],
            x_vals=[2**i for i in range(20, 27)],  # 1M to ~130M elements
            line_arg="provider",
            line_vals=list(_PROVIDER_TO_FN.keys()),
            line_names=list(_PROVIDER_TO_FN.keys()),
            styles=[("blue", "-"), ("red", "--"), ("green", "-")],
            ylabel="GB/s",
            plot_name="silu_mul_forward_bf16",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(n_elements: int, provider: str) -> tuple[float, ...]:
        dtype = torch.bfloat16
        device = "cuda"

        x = torch.randn(n_elements, device=device, dtype=dtype)
        y = torch.randn(n_elements, device=device, dtype=dtype)

        # Bind arguments to the function
        fn = functools.partial(_PROVIDER_TO_FN[provider], x, y)

        # Measure execution time
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        def gbps(ms: float) -> float:
            # Bandwidth Calculation:
            # Operation is element-wise: out = silu(x) * y
            # Memory Traffic:
            #   - Read  x (2 bytes)
            #   - Read  y (2 bytes)
            #   - Write out (2 bytes)
            # Total IO per element = 6 bytes
            total_bytes = 6 * n_elements
            return (total_bytes * 1e-9) / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=None, show_plots=True, print_data=True)


if __name__ == "__main__":
    run_benchmark()
