import pytest
import torch
import torch.nn.functional as F
from d9d.pipelining.api import PipelineStageInfo
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing import assert_close

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights
from d9d_test.modules.helper.compare import assert_kl_div_close_logits
from d9d_test.modules.model.sequence.catalogue import ModelCatalogue
from d9d_test.modules.model.sequence.classification.batch import build_classification_batch
from d9d_test.modules.model.sequence.classification.catalogue import (
    D9D_MODEL_FACTORIES_CLS,
    D9D_TO_HF_MAPPER_CLS,
    HF_MODEL_FACTORY_CLS,
    HF_TO_D9D_MAPPER_CLS,
)


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_CLS.items()
        for model_factory in factories
    ],
)
def test_consistent_to_hf(model_type: ModelCatalogue, model_factory_d9d):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    batch = build_classification_batch()

    model_hf = HF_MODEL_FACTORY_CLS[model_type]()

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        outputs_hf = model_hf(
            input_ids=batch.sequence.input_ids,
            position_ids=batch.sequence.position_ids,
            labels=batch.labels,
        )
        outputs_hf.loss.backward()

    model_d9d = model_factory_d9d(stage)
    clone_module_weights(from_module=model_hf, to_module=model_d9d, map_with=HF_TO_D9D_MAPPER_CLS[model_type])

    outputs_d9d = model_d9d(
        input_ids=batch.sequence.input_ids, position_ids=batch.sequence.position_ids, pooling_mask=batch.pooling_mask
    )
    scores = outputs_d9d["scores"]
    assert scores.dtype == torch.float32

    assert_kl_div_close_logits(scores.bfloat16(), outputs_hf.logits, threshold=1e-3)

    loss_my = F.cross_entropy(scores.bfloat16(), batch.labels)
    assert_close(loss_my, outputs_hf.loss, atol=1e-2, rtol=0.001)
    loss_my.backward()

    assert_mapped_gradients_close(from_module=model_d9d, to_module=model_hf, map_with=D9D_TO_HF_MAPPER_CLS[model_type])
