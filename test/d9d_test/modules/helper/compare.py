import torch
import torch.nn.functional as F

from .tolerances import GradTolerance


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        return x.reshape(1, -1)
    return x.reshape(-1, x.shape[-1] * x.shape[-2])


def assert_angle_and_norm_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    tol: GradTolerance,
    name: str = "grad",
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}")

    actual_fp32 = _as_2d(actual.detach().float())
    expected_fp32 = _as_2d(expected.detach().float())

    both_zero = (actual_fp32.abs().sum(dim=-1) == 0) & (expected_fp32.abs().sum(dim=-1) == 0)
    actual_fp32 = actual_fp32[~both_zero]
    expected_fp32 = expected_fp32[~both_zero]

    if actual_fp32.numel() == 0 and expected_fp32.numel() == 0:
        return

    angle_error = (1 - torch.cosine_similarity(actual_fp32, expected_fp32, dim=-1)).max().item()
    actual_norm = actual_fp32.norm(dim=-1, p=2)
    expected_norm = expected_fp32.norm(dim=-1, p=2)

    if angle_error > tol.tol_angle:
        raise AssertionError(f"{name}: angle error too large: {angle_error:.6g} > {tol.tol_angle:.6g}")

    torch.testing.assert_close(
        actual_norm,
        expected_norm,
        atol=tol.tol_norm_abs,
        rtol=tol.tol_norm_rel,
        msg=lambda error: f"{name}.norm: {error}",
    )


def assert_kl_div_close_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    threshold: float = 1e-3,
) -> None:
    log_probs_actual = F.log_softmax(actual.detach().float(), dim=-1)
    probs_expected = F.softmax(expected.detach().float(), dim=-1)

    kl_div = F.kl_div(log_probs_actual, probs_expected, reduction="batchmean")
    kl_div_item = kl_div.item()

    if kl_div_item > threshold:
        raise AssertionError(f"KL divergence too large: {kl_div_item:.6g} > {threshold:.6g}")
