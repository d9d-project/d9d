# Method Stacking

## About

Fine-tuning can combine several methods. The `d9d.peft.all` package groups multiple PEFT configurations into a single `PeftStack`. The stack injects its methods in order and merges them in reverse order.

## Usage

This example applies LoRA to attention layers and fully fine-tunes normalization layers.

```python
import re
from d9d.peft.all import PeftStackConfig, peft_method_from_config
from d9d.peft.lora import LoRAConfig, LoRAParameters
from d9d.peft.full_tune import FullTuneConfig
from d9d.peft import inject_peft_and_freeze, merge_peft

# 1. Define the methods.
config = PeftStackConfig(
    methods=[
        # LoRA on attention projections.
        LoRAConfig(
            module_name_pattern=re.compile(r".*self_attn\..*_proj"),
            params=LoRAParameters(r=8, alpha=16, dropout=0.05)
        ),
        # Full fine-tuning of normalization layers.
        FullTuneConfig(
            module_name_pattern=re.compile(r".*norm.*")
        )
    ]
)

# 2. Build a PeftStack that contains both methods.
method = peft_method_from_config(config)

# 3. Inject.
mapper = inject_peft_and_freeze(method, model)

# ... pass the mapper to d9d's Trainer or load a checkpoint with it ...

# ... train the model ...

# 4. Merge the adapters into the base weights before exporting the model.
merge_peft(method, model)
```

## API Reference

::: d9d.peft.all
