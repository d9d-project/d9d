# LoRA

## About

The `d9d.peft.lora` package implements [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685). It wraps linear layers, both `nn.Linear` and d9d's [`GroupedLinear`](../models/modules/moe.md). The wrapper holds the original frozen layer (`base`) and two trainable low-rank layers (`lora_A` and `lora_B`). LoRA does not support `nn.Linear` layers with a bias.

## State Mapping

Because the original layer moves to the `base` submodule, its state keys change. LoRA returns a `ModelStateMapperRename` for each wrapped layer. Chain these mappers after your checkpoint mapper, so standard checkpoints still load (see the example below).

## Usage

```python
import re

from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.model_state.mapper.compose import ModelStateMapperSequential
from d9d.peft import inject_peft_and_freeze, merge_peft
from d9d.peft.lora import LoRA, LoRAConfig, LoRAParameters

model = ...  # Your model, before its weights are loaded.

# Maps the checkpoint keys to the original model structure.
checkpoint_mapper = identity_mapper_from_module(model)

# 1. Configure LoRA for the attention query projections.
config = LoRAConfig(
    module_name_pattern=re.compile(r".*self_attn\.q_proj"),
    params=LoRAParameters(
        r=16,
        alpha=32,
        dropout=0.1
    )
)

# 2. Create the method.
method = LoRA(config)

# 3. Inject.
# Replaces the matching nn.Linear layers with LoRALinear layers in place.
# The PEFT mapper only routes "q_proj.weight" to "q_proj.base.weight".
peft_mapper = inject_peft_and_freeze(method, model)

# Keys that the PEFT mapper does not change pass through unchanged.
mapper = ModelStateMapperSequential([checkpoint_mapper, peft_mapper])

# ... return the mapper from your ModelProvider, or load a checkpoint with it ...

# ... train the model ...

# 4. Merge the adapters into the base weights before exporting the model.
merge_peft(method, model)
```

## API Reference

::: d9d.peft.lora
