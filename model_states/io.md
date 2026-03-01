---
title: Model State I/O
---

# Model State IO

## About

The `d9d.model_state.io` package handles the reading and writing of model checkpoints. 

We use checkpoint format that is compatible with HuggingFace format. This format is characterized by using sharded `model-00001-of-XXXXX.safetensors` .safetensors files for storing parameter tensors along with `model.safetensors.index.json` file containing the metadata.

It is tightly integrated with the [`d9d.model_state.mapper`](mapper.md) framework to allow for **Streamed Transformation** - converting model architectures on-the-fly during IO without loading the entire model into memory.

## Core Concepts

### Why Support Transformations

In d9d all the model state input/output logic is natively integrated with mapping and transforming model states. Such a combined system acts as a powerful abstraction layer that decouples the **checkpoint architecture** (how weights are stored on disk) from the **model architecture** (how weights are used in PyTorch code).

This integration is critical for:

*   **Native HuggingFace Compatibility**: You can use highly optimized, custom model implementations (e.g., using a single packed `qkv_proj` tensor) while reading directly from standard community checkpoints (which typically store `q_proj`, `k_proj`, and `v_proj` separately). The mapper handles the reshaping and stacking on-the-fly during the read stream. This eliminates the need for maintaining separate "conversion scripts" or storing duplicate, converted copies of large models.
*   **Runtime Structure Adapting (e.g., LoRA)**: When injecting adapters like LoRA, the runtime model structure changes - often wrapping original layers. For example, a standard `some_linear.weight` on disk might need to be loaded into `some_linear.orig.weight` in memory. Instead of loading the full state dict and manually patching keys (which spikes memory), the mapper reroutes these keys **without the need of materializing the model weights fully**.

### How Loading Works

Standard model loading involves loading a huge dictionary into CPU RAM, filtering and processing it, and moving the results to GPU. 
This approach is ineffective since it requires a lot of CPU-GPU transfers, consumes high amount of memory and involves duplicate work across different pipeline parallel workers.

d9d proposes a different approach:

*   **Streaming & Eviction**: Tensors are loaded in streamed manner and therefore kept in memory only when needed. Once a mapper group (e.g., "stack Q, K, V") is executed, the source tensors are immediately evicted from memory.
*   **Topology-Aware Loading**: Instead of blindly loading all the files, the reader inspects the `ModelStateMapper`. It calculates exactly which files contain the required inputs.

### How Saving Works 

Standard model saving often requires gathering all parameters to a single rank (causing OOM) or manual orchestration of file names and indices across hundreds of GPUs. 

d9d's approach automates the checkpoint exporting lifecycle for large-scale distributed setups:

*   **Streaming & Eviction**: Tensors are saved in streamed manner and therefore kept in memory only when needed. Once a mapper group (e.g., "stack Q, K, V") is executed, the source tensors are immediately evicted from memory. Target tensors are kept in memory only before they are flushed to respective `.safetensors` files.
*   **Distributed Awareness**: In addition to providing local model exporting, we provide distributed-aware export functions. The writer natively understands distributed topologies (via `ProcessGroup` or `DeviceMesh`). In Pipeline Parallel scenarios, it identifies which rank holds the specific stage master copy, ensuring that parameters are written exactly once without race conditions or duplication.

## Usage Examples

These examples provide information primarily how to load and write model states in a pass-through way. 
If you want to see examples of complex model state mapping, please refer to [ModelStateMapper](mapper.md) documentation.

### Raw I/O - Streamed Loading
This example shows how to load a model without spiking memory usage.

```python
from pathlib import Path
import torch
from d9d.model_state.io.reader import read_model_state
from d9d.model_state.mapper.leaf import ModelStateMapperStackTensors
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# Define the mapper (Topology)
# in this example we will load all the parameters the model contains
mapper = identity_mapper_from_module(model)

# Start the stream
# 'src_dir' must contain safetensors files and model.safetensors.index.json
loader_stream = read_model_state(
    src_dir=Path("./checkpoint"),
    mapper=mapper,
    device="cpu"  # or "cuda:0"
)

# Iterate through the transformed results
state_dict = {}
for name, tensor in loader_stream:
    print(f"Loaded and transformed: {name} -> {tensor.shape}")
    state_dict[name] = tensor
```

### Raw I/O - Streamed Saving
Saving a model locally, automatically splitting into 1 GB shards.

```python
from pathlib import Path
from d9d.model_state.io.writer import write_model_state_local
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# 1. Create a generator for your model states
state_generator = model.named_parameters()

# 2. Define a mapper (Identity if no transformation is needed during save)
mapper = identity_mapper_from_module(model)

# 3. Write
# This handles sharding and metadata file creation automatically
write_model_state_local(
    dest_dir=Path("./output_checkpoint"),
    mapper=mapper,
    state_generator=state_generator,
    shard_size_gb=1.0  # Split files if they exceed 1GB
)
```

### Raw I/O - Distributed Load-Transform-Save

One of the most powerful features of `d9d` is the ability to perform **Offline Checkpoint Conversion** using a distributed cluster.

If you have a massive checkpoint in Format A (e.g., HuggingFace) and need to convert it to Format B (e.g., a custom Training format with packed QKV), you don't need a single machine with 1TB RAM. instead, you can spin up 8 GPUs, have each GPU process 1/8th of the keys in parallel, and write a new sharded checkpoint.

```python
import torch.distributed as dist
from pathlib import Path
from d9d.model_state.io import read_model_state, write_model_state_distributed
from d9d.model_state.mapper.compose import ModelStateMapperShard
from d9d.model_state.mapper.adapters import identity_mapper_from_mapper_outputs

# 1. Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 2. Define the global transformation logic
# This describes how the ENTIRE model should be converted.
# e.g., "Stack Q,K,V", "Rename MLP", "Load everying else as-is"
mapper = build_my_fancy_custom_mapper()

# 3. Shard the workload
# We wrap the mapper to restrict execution.
# Rank 0 will only process the first 1/N dependency groups, Rank 1 the next, etc.
# This ensures that no two ranks load/process/save the same tensors.
local_work_mapper = ModelStateMapperShard(
    sub_mapper=mapper,
    total_shards=world_size,
    current_shard=rank
)

# 4. Define saving topology
# The 'read_model_state' generator below will yield tensors that have 
# ALREADY been transformed to their target names/shapes.
# The writer just needs to accept these new keys and save them.
# We generate this identity mapper automatically from the output signature.
writer_mapper = identity_mapper_from_mapper_outputs(local_work_mapper)

# 5. Execute the pipeline
# - Reader: Loads specific source files, transforms, yields new tensors.
# - Writer: Receives new tensors, saves to '*.safetensors' files with temporary names
# - Finalizer: Rank 0 creates 'model.safetensors.index.json' covering all ranks and renames .safetensors files to their final names.
write_model_state_distributed(
    dest_dir=Path("./converted_checkpoint"),
    mapper=writer_mapper,
    state_generator=read_model_state(
        src_dir=Path("./original_checkpoint"),
        mapper=local_work_mapper, # Defines what to load and transformations
        device="cuda",
        show_progress=False # Disable read bars to avoid stderr spam
    ),
    process_group=dist.distributed_c10d._get_default_group(),
    shard_size_gb=4.0,
    show_progress=True # Master rank will show global save progress
)
```

### PyTorch Module I/O - Streamed Loading

Loading a checkpoint where disk keys exactly match model keys. `identity_mapper_from_module` ensures only existing model parameters are loaded.

```python
from pathlib import Path
from d9d.model_state.io import load_model_state
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# 1. Setup Model (e.g., empty or on meta device)
model = ...

# 2. Create Identity Topology
# This tells d9d: "Load every key that exists in 'model' as is."
mapper = identity_mapper_from_module(model)

# 3. Stream & Inject
load_model_state(
    src_dir=Path("./checkpoints/v1"),
    mapper=mapper,
    device="cuda",
    model=model
)
```

### PyTorch Module I/O - Streaming Saving (DeviceMesh)

Saves a model in a complex ND Parallel environment using PyTorch `DeviceMesh`.

This features:

*   **DTensor Gathering**: Automatically gathers `DTensor` shards from the mesh into full tensors before writing.
*   **Concurrency Within PP Rank**: In a Data/Tensor/... parallel setup, multiple GPUs hold replicated or sharded copies of the same parameters. This function uses the `DeviceMesh` to ensure that only the "canonical" PP replica (DP Rank 0, TP Rank 0, ...) writes to disk, preventing write conflicts.
*   **Concurrency Across PP Ranks**: Each PP rank writes the data into its own files. After all the PP ranks finish writing, PP Rank 0 merges the metadata from different PP ranks into a single global checkpoint index file.


```python
from pathlib import Path

from torch.distributed.device_mesh import init_device_mesh
from d9d.model_state.io import save_model_state_pipeline_parallel
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# 1. Setup 3D Mesh
# pp=2 (Pipeline), dp=2 (Data), tp=2 (Tensor)
mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "dp", "tp"))

# 2. Define Model Stages
# In this example, each PP rank manages two distinct parts of the model.
my_stages = [TransformerStage(...), TransformerStage(...)]

# 3. Create Topology
# Since this rank manages multiple modules, we create a Parallel mapper
# to combine the requirements of all stages.
mapper = ModelStateMapperParallel([
    identity_mapper_from_module(stage) for stage in my_stages
])

# 4. Save
# The system inspects the mesh. It identifies if the current rank is 
# the "Master" for the provided stages (i.e., dp_rank=0, tp_rank=0).
# If so, it gathers DTensors and writes. If not, it skips writing 
# but participates in the collective gather.
save_model_state_pipeline_parallel(
    dest_dir=Path("./checkpoint"),
    mapper=mapper,
    device_mesh=mesh,
    pipeline_dim_name="pp",
    models=my_stages,
    shard_size_gb=4.0
)
```

::: d9d.model_state.io
    options:
        show_root_heading: true
        show_root_full_path: true
