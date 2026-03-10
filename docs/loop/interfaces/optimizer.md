# Optimizer

## Auto Optimizer

For standard PyTorch usage, `d9d` includes the `d9d.loop.auto` package. These providers ingest a Pydantic configuration object and manage the creation of standard optimizers.

Supports `AdamW`, `Adam`, `SGD`, and `StochasticAdamW`.

```python
from d9d.loop.auto import AutoOptimizerProvider, AutoOptimizerConfig

provider = AutoOptimizerProvider(
    AutoOptimizerConfig.model_validate_json('{"name": "adamw", "lr": 1e-4}')
)
```

::: d9d.loop.auto.auto_optimizer

## Interface

If you need a custom optimizer, you implement the `OptimizerProvider` protocol.

::: d9d.loop.control.optimizer_provider
