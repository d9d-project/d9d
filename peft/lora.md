---
title: LoRA
---

# Low-Rank Adaptation (LoRA)

## About

The `d9d.peft.lora` package implements Low-Rank Adaptation. It works by wrapping existing Linear layers (both standard `nn.Linear` and d9d's [`GroupedLinear`](../modules/moe.md)) with a container that holds the original frozen layer (`base`) and two low-rank trainable matrices (`lora_A` and `lora_B`).

Because the original layer is moved to a submodule (`base`), the state keys change. The LoRA method automatically generates a `ModelStateMapperRename` to handle this transparently during checkpoint loading.

## Usage Example

```python
import torch
import re
from d9d.peft import inject_peft_and_freeze, merge_peft
from d9d.peft.lora import LoRA, LoRAConfig, LoRAParameters

# 1. Configuration
config = LoRAConfig(
    module_name_pattern=re.compile(r".*attention\.q_proj.*"),  # Target Attention Q projections
    params=LoRAParameters(
        r=16,
        alpha=32,
        dropout=0.1
    )
)

# 2. Instantiate Method
method = LoRA(config)

# 3. Inject
# This replaces nn.Linear with LoRALinear layers in-place.
# 'mapper' knows how to route 'q_proj.weight' -> 'q_proj.base.weight'
mapper = inject_peft_and_freeze(method, model)

# ... pass 'mapper' object to d9d's Trainer or manually load a model checkpoint ...

# ... train a model ...

# 4. Merge - for exporting a model
merge_peft(method, model)
```


::: d9d.peft.lora
    options:
        show_root_heading: true
        show_root_full_path: true
