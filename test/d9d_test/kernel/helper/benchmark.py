import abc
import dataclasses
import functools
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import triton

TKernelSpec = TypeVar("TKernelSpec", bound=Callable)


class CommonKernelProvider(StrEnum):
    D9D = "d9d"
    TORCH_EAGER = "torch-eager"
    TORCH_COMPILE = "torch-compile"
    TORCH_FUNCTIONAL = "torch-functional"
    LIGER_KERNEL = "liger-kernel"


_COMMON_LINE_COLORS = {
    CommonKernelProvider.D9D: "purple",
    CommonKernelProvider.TORCH_EAGER: "red",
    CommonKernelProvider.TORCH_COMPILE: "green",
    CommonKernelProvider.TORCH_FUNCTIONAL: "yellow",
    CommonKernelProvider.LIGER_KERNEL: "lightblue",
}


_COMMON_LINE_STYLES = {
    CommonKernelProvider.D9D: "-",
    CommonKernelProvider.TORCH_EAGER: "--",
    CommonKernelProvider.TORCH_COMPILE: "-",
    CommonKernelProvider.TORCH_FUNCTIONAL: "-",
    CommonKernelProvider.LIGER_KERNEL: "-",
}


@dataclasses.dataclass
class KernelProvider(Generic[TKernelSpec]):
    name: str
    kernel: TKernelSpec
    run_torch_compile: bool = False
    line_color: str | None = None
    line_style: str | None = None

    @property
    def line_color_default(self) -> str:
        if self.line_color:
            return self.line_color
        return _COMMON_LINE_COLORS.get(self.name, "black")

    @property
    def line_style_default(self) -> str:
        if self.line_style:
            return self.line_style
        return _COMMON_LINE_STYLES.get(self.name, "-")


class BenchmarkMode(StrEnum):
    forward = "forward"
    backward = "backward"


def _configuration_to_name(configuration: dict[str, Any]) -> str:
    return "_".join(f"{str(k).upper()}{v}" for k, v in configuration.items())


class TritonBenchmarkSpec(abc.ABC):
    @property
    @abc.abstractmethod
    def x_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def x_vals(self) -> Sequence[int]: ...

    @property
    @abc.abstractmethod
    def configurations(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def build_inputs(self, **params: Any) -> dict[str, Any]: ...

    @abc.abstractmethod
    def total_bytes(self, mode: BenchmarkMode, **params: Any) -> int: ...


class TritonBenchmarkRunner:
    def __init__(
        self,
        name: str,
        providers: list[KernelProvider[TKernelSpec]],
        has_backward: bool,
        spec: TritonBenchmarkSpec,
    ):
        self._name = name
        self._providers = {x.name: x for x in providers}
        self._modes = list(BenchmarkMode) if has_backward else [BenchmarkMode.forward]
        self._spec = spec

    def run(self):
        configs = [
            triton.testing.Benchmark(
                x_names=[self._spec.x_name],
                x_vals=list(self._spec.x_vals),
                line_arg="provider",
                line_vals=[x.name for x in self._providers.values()],
                line_names=[x.name for x in self._providers.values()],
                styles=[(x.line_color_default, x.line_style_default) for x in self._providers.values()],
                ylabel="GB/s",
                plot_name=f"{self._name}_{mode}_{_configuration_to_name(configuration)}",
                args={"mode": mode, **configuration},
            )
            for mode in self._modes
            for configuration in self._spec.configurations
        ]

        @triton.testing.perf_report(configs)
        def benchmark(mode: BenchmarkMode, provider: str, **parameters) -> tuple[float, ...]:
            # resetting torch.compile() cache is crucial since without it,
            # we will benchmark all the torch-compiled functions wrong - torch compile
            # will find the best configuration during the first calls (like small x axis values)
            # and then reuse it for others leading to suboptimal performance
            torch.compiler.reset()

            inputs = self._spec.build_inputs(**parameters)

            # Setup operation wrapper
            provider_info = self._providers[provider]
            fn = provider_info.kernel
            if provider_info.run_torch_compile:
                fn = torch.compile(fn)
            fn = functools.partial(fn, **inputs)

            def gbps(ms: float) -> float:
                total_bytes = self._spec.total_bytes(mode, **parameters)
                return (total_bytes * 1e-9) / (ms * 1e-3)

            grad_to_none = [x for x in inputs.values() if isinstance(x, torch.Tensor)]

            match mode:
                case BenchmarkMode.forward:
                    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

                case BenchmarkMode.backward:
                    out = fn()
                    grad_out = torch.randn_like(out)

                    def _backward() -> None:
                        out.backward(grad_out, retain_graph=True)

                    ms, min_ms, max_ms = triton.testing.do_bench(
                        _backward, quantiles=[0.5, 0.2, 0.8], grad_to_none=grad_to_none
                    )
                case _:
                    raise ValueError("Unknown benchmark mode")

            return gbps(ms), gbps(max_ms), gbps(min_ms)

        save_dir = Path("logs") / "benchmark" / self._name
        save_dir.mkdir(exist_ok=True, parents=True)
        benchmark.run(save_path=save_dir, show_plots=False, print_data=True)
