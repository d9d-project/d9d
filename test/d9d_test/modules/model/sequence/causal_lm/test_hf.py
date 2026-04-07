import pytest
from d9d.pipelining.api import PipelineStageInfo
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing import assert_close

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights
from d9d_test.modules.model.sequence.catalogue import ModelCatalogue
from d9d_test.modules.model.sequence.causal_lm.batch import build_causal_lm_batch
from d9d_test.modules.model.sequence.causal_lm.catalogue import (
    D9D_MODEL_FACTORIES_CAUSAL_LM,
    D9D_TO_HF_MAPPER_CAUSAL_LM,
    HF_MODEL_FACTORY_CAUSAL_LM,
    HF_TO_D9D_MAPPER_CAUSAL_LM,
)


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_CAUSAL_LM.items()
        for model_factory in factories
    ],
)
def test_consistent_to_hf(model_type: ModelCatalogue, model_factory_d9d):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    batch = build_causal_lm_batch()

    labels_shift = batch.labels[:, 1:]

    model_hf = HF_MODEL_FACTORY_CAUSAL_LM[model_type]()

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        outputs_hf = model_hf(
            input_ids=batch.sequence.input_ids,
            position_ids=batch.sequence.position_ids,
            labels=batch.labels,
        )
        outputs_hf.loss.backward()

    model_d9d = model_factory_d9d(stage)
    clone_module_weights(from_module=model_hf, to_module=model_d9d, map_with=HF_TO_D9D_MAPPER_CAUSAL_LM[model_type])

    outputs_d9d = model_d9d(
        input_ids=batch.sequence.input_ids[:, :-1],
        position_ids=batch.sequence.position_ids[:, :-1],
        labels=labels_shift,
    )
    loss_d9d = outputs_d9d["logps"][labels_shift != -100].mean()
    loss_d9d.backward()

    assert_close(loss_d9d, outputs_hf.loss, atol=1e-4, rtol=0.001)
    assert_mapped_gradients_close(
        from_module=model_d9d, to_module=model_hf, map_with=D9D_TO_HF_MAPPER_CAUSAL_LM[model_type]
    )
