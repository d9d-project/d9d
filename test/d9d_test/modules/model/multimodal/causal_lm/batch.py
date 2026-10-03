from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext
from d9d.dataset import compute_multimodal_position_ids
from d9d.module.base import MediaSegments

from d9d_test.modules.helper import torch_seed
from d9d_test.modules.helper.distributed import shard_batch_dim

_IGNORE_INDEX = -100

_BATCH_SIZE = 4
_SEQ_LEN = 41
_VOCAB_SIZE = 100
_MEDIA_TOKEN_ID = 98

# feature grids: one image and one 2-frame video for sample 0; one image for sample 1;
# samples 2 and 3 are text-only.
_MEDIA_PER_SAMPLE: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((1, 4, 8), (2, 4, 4)),
    ((1, 8, 4),),
    (),
    (),
)

# uniform-media batch for distributed tests: every sample carries exactly one image with this
# grid, so sharding the packed features along dim 0 follows the batch dim sharding exactly.
_UNIFORM_BATCH_SIZE = 16
_UNIFORM_SEQ_LEN = 33
_UNIFORM_GRID: tuple[int, int, int] = (1, 4, 8)


@dataclass
class MultimodalCausalLMBatch:
    input_ids: torch.Tensor
    media: MediaSegments
    media_token_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor


def build_multimodal_causal_lm_batch(
    feature_dim: int,
    spatial_merge_size: int,
    device: torch.device | str = "cuda",
) -> MultimodalCausalLMBatch:
    """Builds a deterministic multimodal batch with mixed media/text-only samples.

    Args:
        feature_dim: The packed media feature dimension.
        spatial_merge_size: The spatial merge factor of the vision encoder.
        device: Target device.

    Returns:
        The multimodal batch (media placeholders embedded into ``input_ids``, 3D position ids).
    """
    with torch_seed(1543):
        input_ids = torch.randint(0, _VOCAB_SIZE - 2, (_BATCH_SIZE, _SEQ_LEN), dtype=torch.long)
        media_token_mask = torch.zeros(_BATCH_SIZE, _SEQ_LEN, dtype=torch.bool)

        all_grids: list[tuple[int, int, int]] = []
        total_patches = 0
        for sample_idx, sample_grids in enumerate(_MEDIA_PER_SAMPLE):
            offset = 2  # leave a short text prefix
            for grid in sample_grids:
                t, h, w = grid
                num_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
                media_token_mask[sample_idx, offset : offset + num_tokens] = True
                input_ids[sample_idx, offset : offset + num_tokens] = _MEDIA_TOKEN_ID
                offset += num_tokens + 1  # one text token between media segments
                all_grids.append(grid)
                total_patches += t * h * w

        grid_thw = torch.tensor(all_grids, dtype=torch.long)
        features = torch.randn(total_patches, feature_dim)

        position_ids = torch.stack(
            [
                compute_multimodal_position_ids(
                    input_ids[sample_idx],
                    media_token_mask[sample_idx],
                    torch.tensor(_MEDIA_PER_SAMPLE[sample_idx], dtype=torch.long).reshape(-1, 3),
                    spatial_merge_size,
                )
                for sample_idx in range(_BATCH_SIZE)
            ],
            dim=1,
        )  # (3, batch, seq)

        labels = input_ids.clone()
        labels[media_token_mask] = _IGNORE_INDEX

    return MultimodalCausalLMBatch(
        input_ids=input_ids.to(device),
        media=MediaSegments(features=features.to(device), grid_thw=grid_thw.to(device)),
        media_token_mask=media_token_mask.to(device),
        position_ids=position_ids.to(device),
        labels=labels.to(device),
    )


def uniform_media_tokens_per_sample(spatial_merge_size: int) -> int:
    """Returns the number of media tokens each sample of the uniform batch carries.

    Args:
        spatial_merge_size: The spatial merge factor of the vision encoder.

    Returns:
        The per-sample media token count.
    """
    t, h, w = _UNIFORM_GRID
    return t * (h // spatial_merge_size) * (w // spatial_merge_size)


def build_uniform_multimodal_causal_lm_batch(
    feature_dim: int,
    spatial_merge_size: int,
    device: torch.device | str = "cuda",
) -> MultimodalCausalLMBatch:
    """Builds a deterministic multimodal batch where every sample carries one identical-grid image.

    The uniform layout makes the packed media stream sharding follow the batch-dim sharding
    exactly, which distributed self-consistency tests rely on.

    Args:
        feature_dim: The packed media feature dimension.
        spatial_merge_size: The spatial merge factor of the vision encoder.
        device: Target device.

    Returns:
        The multimodal batch.
    """
    with torch_seed(9271):
        t, h, w = _UNIFORM_GRID
        num_tokens = uniform_media_tokens_per_sample(spatial_merge_size)
        patches_per_sample = t * h * w

        input_ids = torch.randint(0, _VOCAB_SIZE - 2, (_UNIFORM_BATCH_SIZE, _UNIFORM_SEQ_LEN), dtype=torch.long)
        media_token_mask = torch.zeros(_UNIFORM_BATCH_SIZE, _UNIFORM_SEQ_LEN, dtype=torch.bool)
        media_token_mask[:, 2 : 2 + num_tokens] = True
        input_ids[media_token_mask] = _MEDIA_TOKEN_ID

        grid = torch.tensor([_UNIFORM_GRID], dtype=torch.long)
        grid_thw = grid.repeat(_UNIFORM_BATCH_SIZE, 1)
        features = torch.randn(_UNIFORM_BATCH_SIZE * patches_per_sample, feature_dim)

        position_ids = torch.stack(
            [
                compute_multimodal_position_ids(
                    input_ids[sample_idx], media_token_mask[sample_idx], grid, spatial_merge_size
                )
                for sample_idx in range(_UNIFORM_BATCH_SIZE)
            ],
            dim=1,
        )  # (3, batch, seq)

        labels = input_ids.clone()
        labels[media_token_mask] = _IGNORE_INDEX

    return MultimodalCausalLMBatch(
        input_ids=input_ids.to(device),
        media=MediaSegments(features=features.to(device), grid_thw=grid_thw.to(device)),
        media_token_mask=media_token_mask.to(device),
        position_ids=position_ids.to(device),
        labels=labels.to(device),
    )


def shard_uniform_multimodal_batch(
    batch: MultimodalCausalLMBatch, dist_ctx: DistributedContext
) -> MultimodalCausalLMBatch:
    """Shards a uniform-media multimodal batch across data-parallel ranks.

    Args:
        batch: The uniform-media batch (one identical-grid image per sample).
        dist_ctx: The distributed context.

    Returns:
        The local shard of the batch.
    """
    batch_size = batch.input_ids.shape[0]
    patches_per_sample = batch.media.features.shape[0] // batch_size

    # (batch, patches, dim) view so the packed features shard along the batch dim
    features = batch.media.features.view(batch_size, patches_per_sample, -1)

    return MultimodalCausalLMBatch(
        input_ids=shard_batch_dim(batch.input_ids, dist_ctx),
        media=MediaSegments(
            features=shard_batch_dim(features, dist_ctx).flatten(0, 1),
            grid_thw=shard_batch_dim(batch.media.grid_thw, dist_ctx),
        ),
        media_token_mask=shard_batch_dim(batch.media_token_mask, dist_ctx),
        position_ids=shard_batch_dim(batch.position_ids.transpose(0, 1), dist_ctx).transpose(0, 1),
        labels=shard_batch_dim(batch.labels, dist_ctx),
    )
