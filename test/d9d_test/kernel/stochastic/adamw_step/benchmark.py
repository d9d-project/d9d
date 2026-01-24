import functools

import torch
import triton
from d9d.kernel.stochastic import adamw_stochastic_bf16_

from d9d_test.kernel.stochastic.adamw_step.reference_impl import adamw_step_torch
from d9d_test.kernel.stochastic.copy.reference_impl import copy_fp32_to_bf16_stochastic_torch_


def adamw_stochastic_bf16_torch(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step):
    p_new, m_new, v_new = adamw_step_torch(
        params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step
    )
    copy_fp32_to_bf16_stochastic_torch_(params, p_new)
    copy_fp32_to_bf16_stochastic_torch_(exp_avg, m_new)
    copy_fp32_to_bf16_stochastic_torch_(exp_avg_sq, v_new)


_PROVIDER_TO_FN = {
    "d9d": adamw_stochastic_bf16_,
    "torch-eager": adamw_stochastic_bf16_torch,
    "torch-compile": torch.compile(adamw_stochastic_bf16_torch),
}


def run_benchmark():
    configs = [
        triton.testing.Benchmark(
            x_names=["n_elements"],
            x_vals=[2 ** i for i in range(20, 27)],  # 1M to ~130M elements
            line_arg="provider",
            line_vals=list(_PROVIDER_TO_FN.keys()),
            line_names=list(_PROVIDER_TO_FN.keys()),
            styles=[("blue", "-"), ("red", "--"), ("green", "-")],
            ylabel="GB/s",
            plot_name="adamw_stochastic_bf16_",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(n_elements: int, provider: str) -> tuple[float, ...]:
        dtype = torch.bfloat16
        device = "cuda"

        p = torch.randn(n_elements, device=device, dtype=dtype)
        g = torch.randn(n_elements, device=device, dtype=dtype)
        m = torch.randn(n_elements, device=device, dtype=dtype)
        v = torch.randn(n_elements, device=device, dtype=dtype)

        # Hyperparams
        lr = 1e-3
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        wd = 1e-2
        step = 10

        fn = functools.partial(
            _PROVIDER_TO_FN[provider],
            p, g, m, v,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
            step=step
        )

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        def gbps(ms: float) -> float:
            # Bandwidth Calculation:
            # Assumes all inputs/outputs are BF16 (2 bytes)
            # P: Read(2) + Write(2) = 4
            # G: Read(2)            = 2
            # M: Read(2) + Write(2) = 4
            # V: Read(2) + Write(2) = 4
            # Total IO per element  = 14 bytes
            total_bytes = 14 * n_elements
            return (total_bytes * 1e-9) / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=None, show_plots=True, print_data=True)


if __name__ == "__main__":
    run_benchmark()
