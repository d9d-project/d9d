# Full Fine-Tuning

## About

The `d9d.peft.full_tune` package brings standard fine-tuning into the PEFT workflow. It does not change the model architecture. It unfreezes all parameters of the modules whose names fully match a regular expression, e.g. normalization layers or a specific head.

Full fine-tuning is most useful together with other PEFT methods through [Method Stacking](./stack.md). For example, you can apply LoRA to attention layers and fully fine-tune the normalization layers.

## Usage

```python
import re
from d9d.peft import inject_peft_and_freeze
from d9d.peft.full_tune import FullTune, FullTuneConfig

method = FullTune(FullTuneConfig(module_name_pattern=re.compile(r".*norm.*")))

# Freezes every parameter except those of modules named like "...norm...".
mapper = inject_peft_and_freeze(method, model)
```

## API Reference

::: d9d.peft.full_tune
