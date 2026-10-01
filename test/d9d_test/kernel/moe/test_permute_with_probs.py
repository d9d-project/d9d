import pytest
import torch
from d9d.kernel.moe import moe_permute_with_probs, moe_unpermute_mask
from torch.testing import assert_close

from d9d_test.kernel.moe.reference_impl import permute_torch, unpermute_torch

_CASES = [
    pytest.param(1000, 333, 16, 4, id="small"),
    # 70000 * 8 = 560000 permuted rows, 560000 * 4096 > 2^31: row offsets overflow int32
    pytest.param(70000, 4096, 16, 8, id="int32_overflow"),
]


def _make_routing(num_tokens: int, num_experts: int, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.rand((num_tokens, num_experts), device="cuda").argsort(dim=1)[:, :topk]
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, topk_ids, True)
    probs = torch.rand((num_tokens, num_experts), device="cuda") * routing_map
    return routing_map, probs


@pytest.mark.local
@pytest.mark.parametrize(("num_tokens", "hidden_size", "num_experts", "topk"), _CASES)
def test_permute(num_tokens, hidden_size, num_experts, topk):
    torch.manual_seed(42)
    routing_map, probs = _make_routing(num_tokens, num_experts, topk)
    num_out_tokens = num_tokens * topk

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    probs = probs.requires_grad_()
    x_ref = x.detach().float().requires_grad_()
    probs_ref = probs.detach().clone().requires_grad_()

    permuted, permuted_probs, _ = moe_permute_with_probs(x, probs, routing_map, num_out_tokens=num_out_tokens)
    permuted_ref, permuted_probs_ref = permute_torch(x_ref, probs_ref, routing_map)

    assert_close(permuted, permuted_ref.to(torch.bfloat16), rtol=0, atol=0)
    assert_close(permuted_probs, permuted_probs_ref, rtol=0, atol=0)

    grad_permuted = torch.randn_like(permuted)
    grad_permuted_probs = torch.randn_like(permuted_probs)
    torch.autograd.backward((permuted, permuted_probs), (grad_permuted, grad_permuted_probs))
    torch.autograd.backward((permuted_ref, permuted_probs_ref), (grad_permuted.float(), grad_permuted_probs))

    assert_close(x.grad, x_ref.grad.to(torch.bfloat16), rtol=1.6e-2, atol=1e-5)
    assert_close(probs.grad, probs_ref.grad, rtol=0, atol=0)


@pytest.mark.local
@pytest.mark.parametrize("with_merging_probs", [False, True])
@pytest.mark.parametrize(("num_tokens", "hidden_size", "num_experts", "topk"), _CASES)
def test_unpermute(num_tokens, hidden_size, num_experts, topk, with_merging_probs):
    torch.manual_seed(42)
    routing_map, probs = _make_routing(num_tokens, num_experts, topk)
    num_out_tokens = num_tokens * topk
    restore_shape = torch.Size((num_tokens, hidden_size))

    with torch.no_grad():
        dummy = torch.empty(restore_shape, dtype=torch.bfloat16, device="cuda")
        _, _, row_id_map = moe_permute_with_probs(dummy, probs, routing_map, num_out_tokens=num_out_tokens)
        del dummy

    y = torch.randn((num_out_tokens, hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y_ref = y.detach().float().requires_grad_()
    merging_probs = probs.requires_grad_() if with_merging_probs else None
    merging_probs_ref = probs.detach().clone().requires_grad_() if with_merging_probs else None

    out = moe_unpermute_mask(y, row_id_map, merging_probs=merging_probs, restore_shape=restore_shape)
    out_ref = unpermute_torch(y_ref, routing_map, merging_probs_ref)

    assert_close(out, out_ref.to(torch.bfloat16), rtol=1.6e-2, atol=1e-5)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out.float())

    assert_close(y.grad, y_ref.grad.to(torch.bfloat16), rtol=1.6e-2, atol=1e-5)
    if with_merging_probs:
        assert_close(merging_probs.grad, merging_probs_ref.grad, rtol=1e-4, atol=1e-4)
