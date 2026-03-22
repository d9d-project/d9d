# Interfaces & Logic

## About

The `d9d` training loop is agnostic to the specific model or data being trained. You interact with the loop by implementing **Providers** (factories) and **Tasks** (step logic).

For standard use cases (like standard Optimizers), `d9d` provides **Auto** implementations that can be configured purely via Pydantic models, avoiding the need to write custom provider classes.

## Navigation


*   **[User Tasks](./task.md)**: See how to implement `TrainTask` and `InferenceTask` to define your custom step logic, manipulate batch inputs, pass data across pipeline states, and compute losses.
*   **[Model Definition](./model.md)**: Learn how to implement `ModelProvider` to initialize models, handle state mapping, and configure horizontal parallelism.
*   **[Data Loading](./data.md)**: Understand the `DatasetProvider` for instantiating datasets, managing data collation, and configuring distributed-aware sharding.
*   **[Events & Hooks](./events.md)**: Discover how to hook into specific moments of the train or inference lifecycle using the declarative Event Bus.
*   **[Optimizers](./optimizer.md)**: Learn about the `AutoOptimizerProvider` for easy Pydantic-based configuration of standard optimizers, or how to write your own `OptimizerProvider`.
*   **[Learning Rate Scheduler](./lr_scheduler.md)**: Explore the `AutoLRSchedulerProvider` for piecewise scheduling (warmup, hold, decay) or the custom `LRSchedulerProvider` interface.
