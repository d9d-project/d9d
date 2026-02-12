---
title: Method Stacking
---

# Method Stacking

## About

Complex fine-tuning often requires hybrid approaches.

The `d9d.peft.all` package facilitates this by grouping multiple PEFT configurations into a single `PeftStack`.

## Usage Example

Applying LoRA to attention layers while fully fine-tuning normalization layers.

```python
import re
from d9d.peft.all import PeftStackConfig, peft_method_from_config
from d9d.peft.lora import LoRAConfig, LoRAParameters
from d9d.peft.full_tune import FullTuneConfig
from d9d.peft import inject_peft_and_freeze, merge_peft

# 1. Define your Strategy
config = PeftStackConfig(
    methods=[
        # Method A: LoRA on attention projections
        LoRAConfig(
            module_name_pattern=re.compile(r".*attention\..*_proj.*"),
            params=LoRAParameters(r=8, alpha=16, dropout=0.05)
        ),
        # Method B: Full Tune on LayerNorms
        FullTuneConfig(
            module_name_pattern=re.compile(r".*norm.*")
        )
    ]
)

# 2. Create Factory
# This automatically creates a PeftStack containing the sub-methods
method = peft_method_from_config(config)

# 3. Inject
mapper = inject_peft_and_freeze(method, model)

# ... pass 'mapper' object to d9d's Trainer or manually load a model checkpoint ...

# ... train a model ...

# 4. Merge - for exporting a model
merge_peft(method, model)
```

::: d9d.peft.all
    options:
        show_root_heading: true
        show_root_full_path: true
