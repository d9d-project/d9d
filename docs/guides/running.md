# Running Jobs

## About

This page shows how to launch a d9d job on one GPU, on several GPUs and on several nodes. It also describes what the job writes to disk and how it resumes. The examples use the [Quickstart](../getting_started/quickstart.md) script.

## One GPU

If all parallelism degrees in `mesh` are 1, the job does not build a device mesh or a process group. It runs as a plain Python process:

```bash
python train.py config.yaml
```

## Several GPUs

Launch one process per GPU with [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html). The number of processes must equal the product of the parallelism degrees in `mesh`. Degrees that only regroup ranks that other degrees already count, such as `expert_parallel`, do not add to the product (see [Distributed Context](../core/dist_context.md)).

For example, this `mesh` replicates the model on 2 GPUs:

```yaml
mesh:
  data_parallel_replicate: 2
```

```bash
torchrun --nproc-per-node 2 train.py config.yaml
```

This `mesh` shards the parameters, gradients and optimizer states over 2 GPUs (FSDP) instead:

```yaml
mesh:
  data_parallel_shard: 2
```

With `AutoDataProvider`, as in the Quickstart, `global_batch_size` stays the same when you add data-parallel ranks. `AutoDataProvider` shards the dataset across the ranks and changes the number of microbatches per step, so that each step still processes `global_batch_size` samples. `global_batch_size` must be divisible by the number of data-parallel ranks × `microbatch_size`. A [custom data provider](../loop/interfaces/data.md#writing-a-custom-dataprovider) decides this itself.

The model provider decides how each module is distributed. The Quickstart provider calls `parallelize_hsdp`, which works for both meshes above. Pipeline, tensor and expert parallelism need support in the model. See [Horizontal Parallelism](../models/horizontal_parallelism.md) and [Pipeline Parallelism](../models/pipeline_parallelism.md).

If you already ran the Quickstart, delete `runs/` or change `trainer.run.name` first. Otherwise, the job tries to resume the Quickstart run (see [Checkpoints and Resuming](#checkpoints-and-resuming)).

## Several Nodes

Run `torchrun` on every node with the same rendezvous settings. For 2 nodes with 8 GPUs each:

```bash
torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint <host of node 0>:29500 train.py config.yaml
```

See [Checkpointing](../loop/checkpointing.md#limits) for where `trainer.checkpointing.save_dir` must be with several nodes.

## Logs

Every rank logs to standard output. See [Troubleshooting](./troubleshooting.md#reading-the-logs) for the log format and for per-rank log files.

## Checkpoints and Resuming

The job saves checkpoints to `trainer.checkpointing.save_dir`/`trainer.run.name`/`save-<step>`. Start the same script with the same run name to resume from the latest checkpoint, e.g. after a crash or after you raised `total_steps`. To start over, change `trainer.run.name` or delete the checkpoints of the run.

See [Checkpointing](../loop/checkpointing.md) for what a checkpoint holds and for its limits.

## Export

`Trainer.export()` saves the model weights as `.safetensors` files. The `prepare_export_model_stage()` method of the model provider returns the state mapper that converts the in-memory weights to the saved keys. For example, a mapper can save the weights in the Hugging Face format (see [Fine-Tune a Hugging Face Model](./finetune_huggingface.md)).
