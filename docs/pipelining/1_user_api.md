---
title: User API
---

# User API

## The d9d Approach

d9d implements a modern, highly modular pipelining engine designed for performance, stability and customization.

### Dynamic Shapes & Algorithmic Shape Inference

To run P2P (Point-to-Point) communication, the receiver must know the shape of the incoming tensor to pre-allocate buffers. d9d asks your model to implement a lightweight protocol (`ModuleSupportsPipelining`) to calculate stage input and output shapes from batch input shapes mathematically, without performing a heavy forward pass or doing a distributed graph tracing.

This allows supporting **Dynamic Shapes** (e.g., varying sequence lengths) efficiently across runs.

### Construction Consistency (No Patching)
A common anti-pattern in distributed training is "Instantiate-then-Delete": creating a huge model on CPU/Meta device and then hacking it apart `del model.layers[N:]`. 

We reject this pattern because of:

1.  **Fragility**: Changes to model architecture require changes to the external slicing script.
2.  **Leaky Abstractions**: Forward methods become full of `if self.layer is not None`.
3.  **Invalid States**: The model object exists in a "zombie" state until sliced.

In d9d, models are **Pipeline-Aware**. Each pipeline rank constructs **only** the sub-graph it owns. The object returned is compliant, complete, and valid immediately.

## Making Models Compatible

### The Protocol

**Implementing the Protocol**

To use Pipeline Parallelism in d9d, your model must implement the `d9d.pipelining.api.ModuleSupportsPipelining` protocol to allow the framework to manage memory and buffer allocations.

**Forward Compatibility**

* Pipelined models currently only support **outputting a dictionary** (`dict[str, torch.Tensor]`). However, we plan to support arbitrary PyTrees in further releases. The keys in the dictionary returned by your `forward` method must strictly match the keys in the dictionary calculated by `infer_stage_outputs_from_pipeline_inputs`. 
* The named arguments accepted by your `forward` method must strictly match the `infer_stage_inputs_from_pipeline_inputs`. 

This allows the communication handler to map tensor names to P2P buffers deterministically.

### Example

Below is a skeleton of a Transformer-like model implemented for d9d pipelining.

```python
import torch
from torch import nn
from d9d.pipelining.api import PipelineStageInfo, distribute_layers_for_pipeline_stage

class MyModelChunk(nn.Module):
    def __init__(self, stage: PipelineStageInfo, config):
        super().__init__()
        self.stage = stage
        self.config = config
        
        # 1. Determine what layers live here
        self.start_layer, self.end_layer = distribute_layers_for_pipeline_stage(
            config.n_layers, num_virtual_layers_pre=1, num_virtual_layers_post=1, stage=stage
        )
        
        # 2. Build sub-modules (using ModuleDict - for compatibility)
        self.layers = nn.ModuleDict({
            str(layer): TransformerBlock(...) 
            for layer in range(self.start_layer, self.end_layer)
        })
        
        # Only build embeddings on first stage
        if stage.is_current_stage_first:
            self.embed = nn.Embedding(...)
            
        # Only build head on last stage
        if stage.is_current_stage_last:
            self.head = nn.Linear(...)

    def forward(self, input_ids=None, hidden_states=None):        
        # Run embeddings only on first stage
        if self.stage.is_current_stage_first:
            x = self.embed(input_ids)
        else:
            x = hidden_states
            
        # Run local layers
        for layer_idx in range(self.start_layer, self.end_layer):
            x = self.layers[str(layer_idx)](x)
        
        outputs = {
            "hidden_states": x
        }
        
        # Last stage logic
        if self.stage.is_current_stage_last:
            logits = self.head(x)
            outputs['logits'] = logits
        
        return outputs

    # --- Protocol Implementation ---

    def infer_stage_inputs_from_pipeline_inputs(self, inputs: dict[str, torch.Tensor], n_microbatches: int):
        batch_size = inputs['input_ids'].shape[0]
        micro_batch_size = batch_size // n_microbatches
        seq_len = inputs['input_ids'].shape[1]
        
        if self.stage.is_current_stage_first:
            # First stage receives raw input IDs
            return {"input_ids": torch.empty((micro_batch_size, seq_len), dtype=torch.long)}
        else:
            # Intermediate stages receive hidden states from previous stage
            return {"hidden_states": torch.empty((micro_batch_size, seq_len, self.hidden_dim))}

    def infer_stage_outputs_from_pipeline_inputs(self, inputs: dict[str, torch.Tensor], n_microbatches: int):
        batch_size = inputs['input_ids'].shape[0]
        micro_batch_size = batch_size // n_microbatches
        seq_len = inputs['input_ids'].shape[1]
        
        outputs = {"hidden_states": torch.empty((micro_batch_size, seq_len, self.config.hidden_dim))}
        
        if self.stage.is_current_stage_last:
            # Last stage outputs logits too
            outputs["logits"] = torch.empty((micro_batch_size, seq_len, self.config.vocab_size))
        
        return outputs
```

## Using the Pipeline

### Supported Schedules

| Example JSON                                                          | Description                                                                                                                                                             |
|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `{"schedule": "inference"}`                                           | Configuration for inference-only pipeline execution. Runs all forward passes sequentially without any backward passes.                                                  |
| `{"schedule": "gpipe"}`                                               | Standard GPipe execution. Assumes a single stage per rank and processes all microbatches for the forward pass before switching to the backward pass.                    |
| `{"schedule": "looped_bfs", "num_stages_per_rank": 2}`                | Looped Breadth-First Search execution. Supports multiple stages per rank (virtualization) and executes all work for a specific stage before moving to the next.         |
| `{"schedule": "1f1b", "num_stages_per_rank": 1, "zero_bubble": true}` | Interleaved 1F1B and Interleaved Zero Bubble execution. Supports multiple stages per rank. Handles sharding backward passes to dI and dW when `zero_bubble` is enabled. |
| `{"schedule": "zero_bubble_v"}`                                       | Zero Bubble V (ZBV) execution. A specialized V-shape topology schedule that splits backward passes into Input and Weight gradients. Requires exactly 2 stages per rank. |
| `{"schedule": "dual_pipe_v"}`                                         | DualPipeV execution. A bidirectional pipeline schedule for high-throughput training using V-shape topology and reciprocal forward/backward scheduling.                  |

### Batch Sharding

Pipelining works by splitting the input batch into `N` microbatches. 
By default, d9d assumes all input and output tensors should be split along dimension 0.

However, if your inputs require different sharding strategy, you can customize this via `PipelineShardingSpec`.

Please see the [sharding utils docs](../core/sharding.md).

```python
from d9d.pipelining.api import PipelineShardingSpec
from d9d.core.sharding import ShardingSpec
from torch.distributed.tensor import Shard

# Example: Split 'images' on dim 1, but replicate 'camera_angles' across all microbatches
my_spec = PipelineShardingSpec(
    input_data={
        "images": Shard(1),
        "camera_angles": None
    }
    # input_kwargs can be defined similarly
)
```

### Usage within the Trainer

Pipelining is available in the [Trainer](TODO) framework. When configuring the Trainer, simply provide an `AnyPipelineScheduleConfig` in your training arguments. The Trainer handles the construction of the schedule and the distribution of layers automatically.

### Advanced - Manual Usage

If you want to use pipelining outside the Trainer (e.g., custom loops), you use the `build_schedule` factory.

The `build_schedule` function requires a **Model Provider** logic. Instead of passing an instantiated model, you pass a function that accepts `PipelineStageInfo` and returns the `nn.Module` for that stage. This ensures construction consistency.

```python
from torch import Tensor
from torch.distributed.tensor import Shard
import torch.nn.functional as F

from d9d.core.dist_context import DistributedContext
from d9d.core.sharding import shard_tree
from d9d.pipelining.factory import build_schedule, PipelineSchedule1F1BConfig
from d9d.pipelining.api import PipelineShardingSpec


# 0. Define an object that manages loss calculation across steps
class PipelineLossHandler:
    def __init__(self, num_microbatches: int):
        self._shard_spec = {
            'target': Shard(0)
        }
        self._num_microbatches = num_microbatches
        self._targets = None

    def set_targets(self, targets: Tensor):
        self._targets = shard_tree(
            {'target': targets},
            sharding_spec=self._shard_spec,
            num_shards=self._num_microbatches,
            enforce_even_split=True
        )

    def compute_loss(self, outputs: dict[str, Tensor], microbatch_idx: int):
        # Implement any custom logic here
        current_target = self._targets[microbatch_idx]
        return F.cross_entropy(outputs['logits'].view(-1, outputs['logits'].shape[-1]), current_target.view(-1))

    
# 1. Define configuration
dist_context: DistributedContext = ...
model_config = ...
n_microbatches = 32
schedule_config = PipelineSchedule1F1BConfig(
    num_stages_per_rank=4,  # 4 Virtual stages per rank
    zero_bubble=True  # Enable ZB1P optimization
)

# 2. Build the schedule, model shards and loss compute function
loss_handler = PipelineLossHandler(num_microbatches=n_microbatches)
schedule_info, modules = build_schedule(
    dist_context=dist_context,
    n_microbatches=32,
    schedule_config=schedule_config,
    model_provider=lambda stage: MyModelChunk(stage, model_config),  # Factory function
    loss_fn=loss_handler.compute_loss,
    sharding_spec=PipelineShardingSpec()  # Default sharding across dim 0
)

# 3. Execution
# The schedule object exposes a simple step API
inputs = {"input_ids": ...}  # Full batch
loss_handler.set_targets(...)  # Set targets for full batch
schedule_info.schedule.configure_buffers(inputs, kwargs={})  # Pre-allocate buffers
schedule_info.schedule.step(inputs, kwargs={})
```

::: d9d.pipelining.api
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.pipelining.factory
    options:
        show_root_heading: true
        show_root_full_path: true
