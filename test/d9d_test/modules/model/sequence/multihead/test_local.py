import pytest
import torch
import torch.nn.functional as F
from d9d.module.block.head import SequenceCausalLMHeadShared, SequencePoolingHeadShared
from d9d.module.model.io import (
    SequenceHeadsShared,
    SequenceInput,
    SequenceShared,
)
from d9d.pipelining.api import PipelineStageInfo

from d9d_test.modules.model.sequence.catalogue import ModelCatalogue
from d9d_test.modules.model.sequence.multihead.batch import build_multihead_batch
from d9d_test.modules.model.sequence.multihead.catalogue import (
    D9D_MODEL_FACTORIES_MULTIHEAD,
    HEAD_NAME_CLS,
    HEAD_NAME_LM,
    NUM_LABELS_MULTIHEAD,
)


def _has_grad(module) -> bool:
    return any(param.grad is not None for param in module.parameters())


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(model_type, model_factory)
        for model_type, factories in D9D_MODEL_FACTORIES_MULTIHEAD.items()
        for model_factory in factories
    ],
)
def test_two_heads_share_one_backbone(model_type: ModelCatalogue, model_factory_d9d):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    batch = build_multihead_batch(num_labels=NUM_LABELS_MULTIHEAD)
    batch_size = batch.sequence.input_ids.shape[0]

    model = model_factory_d9d(stage)

    outputs = model(
        SequenceInput(input_ids=batch.sequence.input_ids),
        SequenceHeadsShared(
            sequence=SequenceShared(position_ids=batch.sequence.position_ids),
            heads={
                HEAD_NAME_LM: SequenceCausalLMHeadShared(labels=batch.lm_labels),
                HEAD_NAME_CLS: SequencePoolingHeadShared(pooling_mask=batch.pooling_mask),
            },
        ),
    )

    # Each head's output is keyed by the name it was composed under.
    assert set(outputs) == {HEAD_NAME_LM, HEAD_NAME_CLS}

    logps = outputs[HEAD_NAME_LM].logps
    scores = outputs[HEAD_NAME_CLS].scores

    assert logps.shape == batch.sequence.input_ids.shape
    assert scores.shape == (batch_size, NUM_LABELS_MULTIHEAD)
    assert scores.dtype == torch.float32

    # The task combines per-head tensors; the model holds no loss policy.
    lm_loss = logps[batch.lm_labels != -100].mean()
    cls_loss = F.cross_entropy(scores, batch.cls_labels)
    (lm_loss + cls_loss).backward()

    # Gradients flow through the shared backbone and into both heads.
    assert _has_grad(model.model)
    assert _has_grad(model.heads[HEAD_NAME_LM])
    assert _has_grad(model.heads[HEAD_NAME_CLS])
