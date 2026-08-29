import dataclasses

import torch

from d9d_test.modules.helper import torch_seed

VISION_HIDDEN_SIZE = 256
VISION_INTERMEDIATE_SIZE = 512
VISION_NUM_HEADS = 4
VISION_PATCH_SIZE = 16
VISION_TEMPORAL_PATCH_SIZE = 2
VISION_IN_CHANNELS = 3
VISION_SPATIAL_MERGE_SIZE = 2
VISION_NORM_EPS = 1e-6

# one 8x8 image, one 4x12 image and one 2-frame 4x8 video (feature-grid units)
VISION_GRID_THW = ((1, 8, 8), (1, 4, 12), (2, 4, 8))

FEATURE_DIM = VISION_IN_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE


@dataclasses.dataclass(frozen=True)
class VisionInputsInit:
    features: torch.Tensor
    grid_thw: torch.Tensor
    pre_init: torch.Tensor


@dataclasses.dataclass(frozen=True)
class VisionInputs:
    features: torch.Tensor
    grid_thw: torch.Tensor
    pre: torch.nn.Parameter


def total_patches() -> int:
    return sum(t * h * w for t, h, w in VISION_GRID_THW)


def build_vision_inputs(dtype: torch.dtype) -> VisionInputsInit:
    with torch_seed(2024):
        return VisionInputsInit(
            features=torch.randn((total_patches(), FEATURE_DIM), device="cuda", dtype=dtype),
            grid_thw=torch.tensor(VISION_GRID_THW, dtype=torch.long, device="cuda"),
            pre_init=torch.zeros((1, FEATURE_DIM), device="cuda", dtype=dtype),
        )


def materialize_vision_inputs(init: VisionInputsInit) -> VisionInputs:
    return VisionInputs(
        features=init.features.clone(),
        grid_thw=init.grid_thw.clone(),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )
