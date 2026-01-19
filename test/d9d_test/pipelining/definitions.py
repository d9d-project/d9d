import torch
from torch import nn

from d9d.pipelining.api import ModuleSupportsPipelining


class CustomMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        with torch.no_grad():
            ctx.save_for_backward(a, b)
            return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            a, b = ctx.saved_tensors

            grad_a = grad_b = None

            if ctx.needs_input_grad[0]:
                grad_a = torch.matmul(grad_output, b.transpose(-1, -2))
            if ctx.needs_input_grad[1]:
                grad_b = torch.matmul(a.transpose(-1, -2), grad_output)

            return grad_a, grad_b


class PipelineModel(nn.Module, ModuleSupportsPipelining):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(8, 8) / 4)
        self.w2 = nn.Parameter(torch.randn(8, 8) / 4)
        self.w3 = nn.Parameter(torch.randn(8, 8) / 4)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, y: torch.Tensor):  # x is args (passed by), y is kwargs (inputs)
        r = CustomMatmul.apply(x, self.w1)
        r = r * 1.05
        r = r @ self.w2
        r = self.act(r) + y
        r = CustomMatmul.apply(r, self.w3)
        return {
            'x': r
        }

    def infer_stage_inputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        return {
            'x': torch.empty((inputs['x'].shape[0] // n_microbatches, 8))
        }

    def infer_stage_outputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        return {
            'x': torch.empty((inputs['x'].shape[0] // n_microbatches, 8))
        }


def register_pp_hooks(model: PipelineModel) -> dict[str, int]:
    counts = {'w1': 0, 'w2': 0, 'w3': 0}

    def _make_hook(name: str):
        def _update_counts(param):
            counts[name] += 1

        return _update_counts

    model.w1.register_post_accumulate_grad_hook(_make_hook('w1'))
    model.w2.register_post_accumulate_grad_hook(_make_hook('w2'))
    model.w3.register_post_accumulate_grad_hook(_make_hook('w3'))
    return counts


def check_pp_hooks_ran(state: dict[str, int], count: int):
    assert state['w1'] == count
    assert state['w2'] == count
    assert state['w3'] == count


def extract_grad(x: torch.Tensor) -> torch.Tensor | None:
    if x.grad is None:
        return None

    return x.grad.detach().clone()


def _snapshot_grad(model: PipelineModel, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        'x': extract_grad(x),
        'y': extract_grad(y),
        'w1': extract_grad(model.w1),
        'w2': extract_grad(model.w2),
        'w3': extract_grad(model.w3)
    }


def do_standard_backward(model: PipelineModel, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(x, y)['x']
    loss = out.mean()
    loss.backward()
    snapshot = _snapshot_grad(model, x, y)
    model.zero_grad()
    x.grad = None
    y.grad = None
    return snapshot


def build_pp_model() -> PipelineModel:
    torch.manual_seed(42)
    model = PipelineModel().cuda()
    return model


def build_pp_inputs(x_with_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(4242)
    x = torch.randn(128, 8, requires_grad=x_with_grad, device='cuda')
    y = torch.randn(128, 8, requires_grad=False, device='cuda')
    return x, y
