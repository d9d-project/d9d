# Interfaces

## About

The d9d loop does not depend on a specific model or dataset. You plug your logic into the loop by implementing **providers** (factories) and **tasks** (step logic).

For common cases, such as standard optimizers, d9d provides **Auto** implementations. You configure them with Pydantic models, so you do not need to write a provider class.

## Navigation

*   **[User Tasks](./task.md)**: Implement `TrainTask` and `InferenceTask` to build model inputs, carry data to the loss and compute the loss.
*   **[Model Definition](./model.md)**: Implement `ModelProvider` to initialize models, map their state and set up horizontal parallelism.
*   **[Data Loading](./data.md)**: Use a `DataProvider` to build the `MicrobatchPackStream`, collate data and shard it across ranks.
*   **[Event Bus and Hooks](./events.md)**: Hook into specific moments of the train or inference lifecycle through the event bus.
*   **[Optimizer](./optimizer.md)**: Configure standard optimizers with `AutoOptimizerProvider`, or write your own `OptimizerProvider`.
*   **[Learning Rate Scheduler](./lr_scheduler.md)**: Use `AutoLRSchedulerProvider` for piecewise schedules (warmup, hold, decay), or write your own `LRSchedulerProvider`.
