import pytest
import torch
from d9d.module.block.head import SequencePoolingHeadShared
from d9d.module.model.io import SequenceHeadsShared, SequenceInput, SequenceShared
from d9d.pipelining.api import PipelineStageInfo
from torch.nn.attention import SDPBackend, sdpa_kernel

from d9d_test.modules.helper import GradTolerance, assert_mapped_gradients_close, clone_module_weights
from d9d_test.modules.helper.compare import assert_angle_and_norm_close
from d9d_test.modules.model.sequence.catalogue import ModelCatalogue
from d9d_test.modules.model.sequence.embedding.batch import build_embedding_batch
from d9d_test.modules.model.sequence.embedding.catalogue import (
    D9D_MODEL_FACTORIES_EMBEDDING,
    D9D_TO_HF_MAPPER_EMBEDDING,
    HEAD_NAME_EMBEDDING,
    HF_MODEL_FACTORY_EMBEDDING,
    HF_TO_D9D_MAPPER_EMBEDDING,
)


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_EMBEDDING.items()
        for model_factory in factories
    ],
)
def test_consistent_to_hf(model_type: ModelCatalogue, model_factory_d9d) -> None:
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    batch = build_embedding_batch()

    model_hf = HF_MODEL_FACTORY_EMBEDDING[model_type]()

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        outputs_hf = model_hf(
            input_ids=batch.sequence.input_ids,
            position_ids=batch.sequence.position_ids,
            output_hidden_states=False,
        )
        hf_hidden = outputs_hf.last_hidden_state

        hf_embeddings = hf_hidden[batch.pooling_mask == 1].float()

        loss_hf = hf_embeddings.mean()
        loss_hf.backward()

    model_d9d = model_factory_d9d(stage)
    clone_module_weights(from_module=model_hf, to_module=model_d9d, map_with=HF_TO_D9D_MAPPER_EMBEDDING[model_type])

    outputs_d9d = model_d9d(
        SequenceInput(input_ids=batch.sequence.input_ids),
        SequenceHeadsShared(
            sequence=SequenceShared(position_ids=batch.sequence.position_ids),
            heads={HEAD_NAME_EMBEDDING: SequencePoolingHeadShared(pooling_mask=batch.pooling_mask)},
        ),
    )
    embeddings = outputs_d9d[HEAD_NAME_EMBEDDING].embeddings
    assert embeddings.dtype == torch.float32

    assert_angle_and_norm_close(
        embeddings,
        hf_embeddings,
        tol=GradTolerance(tol_angle=0.01, tol_norm_abs=3e-4, tol_norm_rel=3e-4),
        name="embedding",
    )

    loss_d9d = embeddings.mean()
    loss_d9d.backward()

    assert_mapped_gradients_close(
        from_module=model_d9d, to_module=model_hf, map_with=D9D_TO_HF_MAPPER_EMBEDDING[model_type]
    )
