# Checkpointing

## About

Training and inference jobs save their state to checkpoints, so an interrupted job resumes where it stopped. A checkpoint is a [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html): a directory in which every rank writes its own shards. The `checkpointing` section of the job configuration sets where and how often the job saves.

## What a Checkpoint Holds

Every checkpoint holds:

*   The current step.
*   The model stages.
*   The position in the data stream, per data-parallel rank.
*   The `state_dict()` of the task.

A training checkpoint also holds:

*   The optimizer and the LR scheduler.
*   The metrics and the experiment tracker.

With `model_stage_factory.checkpoint_only_trainable_parameters`, a checkpoint holds only the parameters that require gradients, e.g. the LoRA adapters. On resume, the job loads the frozen parameters again from the model state that `model_stage_factory.source_checkpoint` points to.

A checkpoint is not a model state. To share the weights with other tools, export the model as `.safetensors` files (see [Export](../guides/running.md#export) and [Model State Mapper](../model_states/mapper.md)).

## Where and When

A training job saves to `<save_dir>/<run.name>/save-<step>`. An inference job has no run name and saves to `<save_dir>/save-<step>`.

`period_steps` sets the steps at which the job saves:

*   **A number**: The job saves every that many steps and at the last step.
*   **`"last_step"`**: The job saves only at the last step.
*   **`"disable"`**: The job never saves.

All ranks wait for each other before and after a save. After a save, the main process deletes the oldest checkpoints, so that `num_to_keep` of them remain. With `num_to_keep: null`, the job keeps all of them.

This config saves every 100 steps and keeps the 2 latest checkpoints:

```yaml
trainer:
  checkpointing:
    save_dir: runs/checkpoints
    period_steps: 100
    num_to_keep: 2
```

## Resuming

At the start, every rank looks for the latest `save-<step>` in the directory of the job and loads it. If there is none, the job starts from the first step. After a load, the job continues as follows:

*   **The job was interrupted**: It continues from the step after its latest checkpoint. An inference job skips the data that it already processed.
*   **You raised `total_steps`**: It continues up to the new last step. The LR scheduler follows the new `total_steps`, so phases that are set in percent of the steps move.
*   **The job is finished**: It logs `Training is already complete, nothing to do` or `Inference is already complete, nothing to do` and stops.

[Running Jobs](../guides/running.md#checkpoints-and-resuming) shows how to resume a job and how to start over.

## Limits

*   **A shared file system**: Every rank reads and writes the same checkpoint directory. With several nodes, `save_dir` must be on a file system that all nodes see.

## API Reference

::: d9d.loop.config.CheckpointingConfig
