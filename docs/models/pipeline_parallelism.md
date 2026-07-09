# Pipeline Parallelism

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
from d9d.pipelining.api import PipelineStageInfo, TensorSpec, distribute_layers_for_pipeline_stage

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
    # These receive a single representative microbatch (not a global batch), so shapes are used as-is,
    # and return TensorSpec descriptors (shape/dtype/layout) — no tensors are allocated.

    def infer_stage_inputs_from_pipeline_inputs(self, microbatch_inputs: dict[str, torch.Tensor]):
        micro_batch_size = microbatch_inputs['input_ids'].shape[0]
        seq_len = microbatch_inputs['input_ids'].shape[1]
        
        if self.stage.is_current_stage_first:
            # First stage receives raw input IDs
            return {"input_ids": TensorSpec(shape=(micro_batch_size, seq_len), dtype=torch.long)}
        else:
            # Intermediate stages receive hidden states from previous stage
            return {"hidden_states": TensorSpec(shape=(micro_batch_size, seq_len, self.hidden_dim), dtype=torch.bfloat16)}

    def infer_stage_outputs_from_pipeline_inputs(self, microbatch_inputs: dict[str, torch.Tensor]):
        micro_batch_size = microbatch_inputs['input_ids'].shape[0]
        seq_len = microbatch_inputs['input_ids'].shape[1]
        
        outputs = {"hidden_states": TensorSpec(shape=(micro_batch_size, seq_len, self.config.hidden_dim), dtype=torch.bfloat16)}
        
        if self.stage.is_current_stage_last:
            # Last stage outputs logits too
            outputs["logits"] = TensorSpec(shape=(micro_batch_size, seq_len, self.config.vocab_size), dtype=torch.bfloat16)
        
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

### Microbatches and packs

Pipelining consumes a **pack**: a sequence of ready microbatches for one step. The pack length (and therefore the batch size) may vary from step to step. Buffers are sized
**per microbatch** — each stage infers shapes for every microbatch in the pack independently — so the
microbatches within a single pack may also differ in shape.
The schedule recompiles its program when the microbatch count changes and reallocates buffers when any
microbatch's shape changes.

### Usage within the Trainer

Pipelining is available in the [Trainer](../loop/train.md) framework. When configuring the Trainer, simply provide an `AnyPipelineScheduleConfig` in your training arguments. The Trainer handles the construction of the schedule and the distribution of layers automatically.

### Advanced - Manual Usage

If you want to use pipelining outside the Trainer (e.g., custom loops), you use the `build_schedule` factory.

The `build_schedule` function requires a **Model Provider** logic. Instead of passing an instantiated model, you pass a function that accepts `PipelineStageInfo` and returns the `nn.Module` for that stage. This ensures construction consistency.

```python
from torch import Tensor
import torch.nn.functional as F

from d9d.core.dist_context import DistributedContext
from d9d.pipelining.factory import build_schedule, PipelineSchedule1F1BConfig


# 0. Define an object that manages loss calculation per microbatch
class PipelineLossHandler:
    def __init__(self, targets_microbatches: list[Tensor]):
        self._targets = targets_microbatches

    def compute_loss(self, outputs: dict[str, Tensor], microbatch_idx: int):
        # Implement any custom logic here
        current_target = self._targets[microbatch_idx]
        return F.cross_entropy(outputs['logits'].view(-1, outputs['logits'].shape[-1]), current_target.view(-1))


# 1. Define configuration
dist_context: DistributedContext = ...
model_config = ...
schedule_config = PipelineSchedule1F1BConfig(
    num_stages_per_rank=4,  # 4 Virtual stages per rank
    zero_bubble=True  # Enable ZB1P optimization
)

# 2. Build the per-microbatch inputs (the "pack"). The number of microbatches is decided here, per step.
inputs_microbatches = tuple({"input_ids": mb} for mb in my_input_microbatches)
kwargs_microbatches = tuple({} for _ in inputs_microbatches)
targets_microbatches = [...]  # one target tensor per microbatch

# 3. Build the schedule and model shards (the callback is supplied per step, not here)
loss_handler = PipelineLossHandler(targets_microbatches)
schedule_info, modules = build_schedule(
    dist_context=dist_context,
    schedule_config=schedule_config,
    model_provider=lambda stage: MyModelChunk(stage, model_config),  # Factory function
)

# 4. Execution
# The callback is passed to each step, so it can close over per-step data (here, the targets).
schedule_info.schedule.step(
    inputs_microbatches=inputs_microbatches,
    kwargs_microbatches=kwargs_microbatches,
    callback=loss_handler.compute_loss,
)
```

::: d9d.pipelining.api

::: d9d.pipelining.factory
