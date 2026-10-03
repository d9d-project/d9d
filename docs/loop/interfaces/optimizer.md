# Optimizer

## About

The loop creates one optimizer per local model stage through an `OptimizerProvider`. You can configure a standard optimizer with `AutoOptimizerProvider`, or implement the `OptimizerProvider` protocol yourself.

## Auto Optimizer

The `d9d.loop.auto` package builds standard optimizers from a Pydantic configuration. It supports `AdamW`, `Adam`, `SGD` and [`StochasticAdamW`](../../optimizer/stochastic.md). The `name` field selects the optimizer.

## Custom Optimizer

For a custom optimizer, implement the `OptimizerProvider` protocol. It receives the model stage and returns a PyTorch optimizer.

## Usage

`AutoOptimizerConfig` is a discriminated union, so validate it with a Pydantic `TypeAdapter`:

```python
from pydantic import TypeAdapter

from d9d.loop.auto import AutoOptimizerConfig, AutoOptimizerProvider

config = TypeAdapter(AutoOptimizerConfig).validate_json('{"name": "adamw", "lr": 1e-4}')
provider = AutoOptimizerProvider(config)
```

Inside a larger Pydantic config, declare a field of type `AutoOptimizerConfig` instead.

## API Reference

::: d9d.loop.auto.auto_optimizer

::: d9d.loop.control.optimizer_provider
