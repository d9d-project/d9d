import pytest
import torch
from d9d.model_state.mapper import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity
from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.positional import RotaryEmbeddingStyle
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention

from d9d_test.modules.block.attention.grouped_query.batch import build_inputs, materialize_attention_inputs
from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed


class ModelStateMapperSplitQGate(ModelStateMapper):
    def __init__(
        self,
        merged_name: str,
        q_name: str,
        gate_name: str,
        head_dim: int,
    ) -> None:
        self._merged_name = merged_name
        self._q_name = q_name
        self._gate_name = gate_name
        self._head_dim = head_dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset(
            [
                StateGroup(
                    inputs=frozenset([self._merged_name]),
                    outputs=frozenset([self._q_name, self._gate_name]),
                )
            ]
        )

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        merged_tensor = group[self._merged_name].T

        reshaped_tensor = merged_tensor.view(merged_tensor.shape[0], -1, self._head_dim * 2)

        q_chunk, gate_chunk = reshaped_tensor.chunk(2, dim=-1)

        q_chunk = q_chunk.reshape(q_chunk.shape[0], -1).T.contiguous()
        gate_chunk = gate_chunk.reshape(gate_chunk.shape[0], -1).T.contiguous()

        return {
            self._q_name: q_chunk,
            self._gate_name: gate_chunk,
        }


HEAD_DIM = 32
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)


def build_hf(dtype: torch.dtype):
    with torch_seed(42):
        return (
            Qwen3_5MoeAttention(
                Qwen3_5MoeTextConfig(
                    hidden_size=512,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    attention_bias=False,
                    rms_norm_eps=1e-6,
                    head_dim=HEAD_DIM,
                    partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
                    _attn_implementation="eager",
                ),
                layer_idx=0,
            )
            .cuda()
            .to(dtype)
        )


def build_d9d(dtype: torch.dtype):
    with torch_seed(43):
        module = (
            GroupedQueryAttention(
                hidden_size=512,
                num_attention_heads=16,
                num_key_value_heads=4,
                qk_norm_eps=1e-6,
                head_dim=HEAD_DIM,
                is_causal=True,
                rope_style=RotaryEmbeddingStyle.HALF,
                rope_dim=ROPE_DIM,
                enable_output_gate=True,
                qk_norm_zero_centered=True,
            )
            .cuda()
            .to(dtype)
        )
        module.reset_parameters()
    return module


def _build_state_mapper():
    return ModelStateMapperParallel(
        [
            ModelStateMapperSplitQGate(
                "q_proj.weight", q_name="q_proj.weight", gate_name="gate_proj.weight", head_dim=32
            ),
            *(
                ModelStateMapperIdentity(f"{param_name}.weight")
                for param_name in (
                    "k_norm",
                    "k_proj",
                    "q_norm",
                    "v_proj",
                    "o_proj",
                )
            ),
        ]
    )


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype):
    init = build_inputs(dtype, rope_dim=ROPE_DIM)
    mapper = _build_state_mapper()

    # HF
    inputs_hf = materialize_attention_inputs(init)
    module_hf = build_hf(dtype)

    hidden_states_hf, _ = module_hf(
        inputs_hf.hidden_states + inputs_hf.pre,
        attention_mask=inputs_hf.attention_mask,
        position_embeddings=inputs_hf.rope,
    )
    hidden_states_hf.mean().backward()

    # d9d
    inputs_d9d = materialize_attention_inputs(init)
    module_d9d = build_d9d(dtype)
    clone_module_weights(module_hf, module_d9d, map_with=mapper)

    hidden_states_d9d = module_d9d(
        inputs_d9d.hidden_states + inputs_d9d.pre,
        attention_mask=None,
        position_embeddings=inputs_d9d.rope,
    )
    hidden_states_d9d.mean().backward()

    # Check
    torch.testing.assert_close(
        hidden_states_d9d,
        hidden_states_hf,
        atol=1e-2,
        rtol=1e-2,
    )

    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-6, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)


# TODO(max): add context parallel test with new context parallelism API
