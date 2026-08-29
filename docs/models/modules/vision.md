# Vision Blocks

## About

The `d9d.module.block.vision` package provides building blocks for vision encoders operating on
**packed variable-length media segments**: all images and videos of a microbatch are concatenated
into one `(total_patches, feature_dim)` stream, with per-segment feature grids described by a
`(num_segments, 3)` tensor of `(temporal, height, width)` values. No padding is ever applied;
attention uses variable-length kernels and never crosses segment boundaries.

Model packages assemble their vision towers from these blocks explicitly, exactly as decoder
layers are assembled from attention/FFN blocks (see [Multimodality](../multimodality.md)).

## Features

### Patch Embedding

`PatchEmbedding` projects packed raw patches into the vision hidden size using a `Conv3d` with
kernel size equal to stride — a linear projection per spatio-temporal patch.

### Interpolated Position Embedding

`InterpolatedPositionEmbedding` holds a learned square-grid position table and bilinearly
resamples it to each segment's `(height, width)` feature grid, repeating across temporal frames.

### 2D Rotary Embedding

`VisionRotaryEmbedding2D` produces per-patch `(cos, sin)` rotary embeddings where half of the
frequencies encode the patch row and the other half the patch column. The output is compatible
with `RotaryEmbeddingApplicator` in the `HALF` style.

### Packed Vision Attention

`PackedVisionAttention` computes bidirectional multi-head attention over the packed stream using
a pluggable [variable-length SDPA backend](./attention.md#scaled-dot-product-attention-backends).
Segment boundaries are provided as cumulative sequence lengths (see `segment_cu_seqlens`).

### Spatial Patch Merger

`SpatialPatchMerger` merges `spatial_merge_size**2` neighboring patch embeddings into one media
token and projects it to the language model hidden size.

::: d9d.module.block.vision
