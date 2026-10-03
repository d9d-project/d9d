# Model State I/O

## About

The `d9d.model_state.io` package reads and writes model checkpoints. It is built on the [`d9d.model_state.mapper`](mapper.md) framework, so it can transform model states on the fly while it streams them. It does not load the whole checkpoint into memory at once.

## Checkpoint Format

d9d uses the Hugging Face checkpoint format. Parameter tensors live in sharded `model-00001-of-XXXXX.safetensors` files. The `model.safetensors.index.json` file maps every tensor to its file. The reader also accepts a single unindexed `model.safetensors` file.

## Why I/O Supports Transformations

In d9d, reading and writing model states always goes through a mapper. This decouples the **checkpoint architecture** (how weights are stored on disk) from the **model architecture** (how weights are used in PyTorch code).

This matters for:

*   **Hugging Face compatibility**: Your model can use a single packed `qkv_proj` tensor and still read community checkpoints that store `q_proj`, `k_proj` and `v_proj` separately. The mapper stacks them while it reads. You do not need conversion scripts or converted copies of large models.
*   **Runtime structure changes, e.g. LoRA**: Adapters such as LoRA wrap the original layers. A `some_linear.weight` key on disk then needs to load into `some_linear.base.weight`. The mapper reroutes these keys without loading the full state dict first.

## How Loading Works

Standard loading reads the whole state dict into CPU RAM, processes it and moves the results to the GPU. This uses a lot of memory and CPU-GPU transfers, and every pipeline-parallel rank repeats the same work.

d9d loads differently:

*   **Streaming**: A tensor stays in memory only until its mapper group runs. For example, once the "stack q, k, v" group runs, d9d frees the source tensors.
*   **Selective reading**: The reader inspects the `ModelStateMapper` and reads only the files that contain its inputs.

## How Saving Works

Standard saving often gathers all parameters on a single rank, which can run out of memory. It can also require manual handling of file names and indices across many GPUs.

d9d saves differently:

*   **Streaming**: A source tensor stays in memory only until its mapper group runs. Output tensors stay in memory only until they are flushed to a `.safetensors` file.
*   **Distributed writing**: Besides local saving, the writer accepts a `ProcessGroup` or a `DeviceMesh`. With pipeline parallelism, only one rank per pipeline stage writes, so every parameter is written exactly once.

## Usage

These examples load and save model states without transforming them. For complex mappings, see the [Model State Mapper](mapper.md) page.

### Raw I/O: Streamed Loading

This example reads a checkpoint tensor by tensor.

```python
from pathlib import Path
from d9d.model_state.io import read_model_state
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# Load every state of the model unchanged.
mapper = identity_mapper_from_module(model)

# src_dir must contain either sharded .safetensors files with model.safetensors.index.json,
# or a single unindexed model.safetensors.
loader_stream = read_model_state(
    src_dir=Path("./checkpoint"),
    mapper=mapper,
    device="cpu"  # Or "cuda:0"
)

state_dict = {}
for name, tensor in loader_stream:
    print(f"Loaded and transformed: {name} -> {tensor.shape}")
    state_dict[name] = tensor
```

### Raw I/O: Streamed Saving

This example saves a model locally in shards of at most 1 GiB.

```python
from pathlib import Path
from d9d.model_state.io import write_model_state_local
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# The writer creates the shard files and the index file.
write_model_state_local(
    dest_dir=Path("./output_checkpoint"),
    mapper=identity_mapper_from_module(model),
    state_generator=model.state_dict().items(),
    shard_size_gb=1.0
)
```

### Raw I/O: Distributed Checkpoint Conversion

You can convert a checkpoint offline on a cluster. For example, you can convert a Hugging Face checkpoint into a format with packed `qkv` tensors. You do not need a single machine with enough RAM for the whole model. Each of N GPUs processes 1/N of the keys and writes its part of a new sharded checkpoint.

```python
import torch.distributed as dist
from pathlib import Path
from d9d.model_state.io import read_model_state, write_model_state_distributed
from d9d.model_state.mapper.compose import ModelStateMapperShard
from d9d.model_state.mapper.adapters import identity_mapper_from_mapper_outputs

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Describe how the ENTIRE model is converted,
# e.g. "stack q, k, v", "rename MLP", "load everything else as is".
mapper = build_my_custom_mapper()

# Each rank processes only its share of the dependency groups,
# so no two ranks load, process or save the same tensors.
local_work_mapper = ModelStateMapperShard(
    sub_mapper=mapper,
    total_shards=world_size,
    current_shard=rank
)

# read_model_state yields tensors that already have their target names and shapes,
# so the writer only passes them through.
writer_mapper = identity_mapper_from_mapper_outputs(local_work_mapper)

# The reader loads and transforms the source tensors, and the writer saves them.
# Rank 0 then writes model.safetensors.index.json for all ranks and gives the files their final names.
write_model_state_distributed(
    dest_dir=Path("./converted_checkpoint"),
    mapper=writer_mapper,
    state_generator=read_model_state(
        src_dir=Path("./original_checkpoint"),
        mapper=local_work_mapper,
        device="cuda",
        show_progress=False  # Hide reader progress bars to keep the output readable
    ),
    process_group=dist.group.WORLD,
    shard_size_gb=4.0,
    show_progress=True
)
```

### PyTorch Module I/O: Streamed Loading

This example loads a checkpoint whose keys match the model keys. `identity_mapper_from_module` makes sure that only the model's own states are loaded.

```python
from pathlib import Path
from d9d.model_state.io import load_model_state
from d9d.model_state.mapper.adapters import identity_mapper_from_module

model = ...

# Load every key of the model as is.
mapper = identity_mapper_from_module(model)

load_model_state(
    src_dir=Path("./checkpoints/v1"),
    mapper=mapper,
    device="cuda",
    model=model
)
```

### PyTorch Module I/O: Streamed Saving with a DeviceMesh

`save_model_state_pipeline_parallel` saves a model that is distributed over a PyTorch `DeviceMesh` with any parallelism dimensions:

*   **DTensor gathering**: It gathers `DTensor` shards into full tensors before writing.
*   **One writer per pipeline stage**: With data or tensor parallelism, several GPUs hold copies or shards of the same parameters. Only the rank at coordinate 0 of every non-pipeline dimension writes to disk.
*   **Index merging**: Each pipeline rank writes its own files. Pipeline rank 0 then merges their indices into one global index file.

Every rank must call the function, because the gathering is collective.

```python
from pathlib import Path

from torch.distributed.device_mesh import init_device_mesh
from d9d.model_state.io import save_model_state_pipeline_parallel
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# pp=2 (pipeline), dp=2 (data), tp=2 (tensor).
mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "dp", "tp"))

# Each pipeline rank holds two stages of the model.
my_stages = [TransformerStage(...), TransformerStage(...)]

# Combine the states of all local stages into one mapper.
mapper = ModelStateMapperParallel([
    identity_mapper_from_module(stage) for stage in my_stages
])

save_model_state_pipeline_parallel(
    dest_dir=Path("./checkpoint"),
    mapper=mapper,
    device_mesh=mesh,
    pipeline_dim_name="pp",
    models=my_stages,
    shard_size_gb=4.0
)
```

## API Reference

::: d9d.model_state.io
