# Quickstart

## About

This page trains a small model on one GPU in a few seconds. The job has the same parts as every d9d job: a model, a model provider, a task, a data provider and a configuration. The code is in [`example/quickstart`](https://github.com/d9d-project/d9d/tree/main/example/quickstart).

## Run the Example

Install d9d (see [Installation](./installation.md)) and PyYAML, which reads the config. Then get the example from the repository and run it:

```bash
pip install d9d pyyaml
git clone https://github.com/d9d-project/d9d.git
cd d9d/example/quickstart
python train.py config.yaml
```

A job without parallelism runs as a single process, so it does not need `torchrun`. The job writes the loss and the gradient norm of every step to its log:

```text
...
[d9d] [local] ... - INFO - Starting job from scratch
[d9d] [local] ... - INFO - step 0 (stage=train): loss=13.2302
[d9d] [local] ... - INFO - step 0 (stage=train): l2_grad_norm_total=23.721
[d9d] [local] ... - INFO - step 1 (stage=train): loss=15.0928
[d9d] [local] ... - INFO - step 1 (stage=train): l2_grad_norm_total=23.8028
...
[d9d] [local] ... - INFO - step 255 (stage=train): loss=0.0134067
[d9d] [local] ... - INFO - step 255 (stage=train): l2_grad_norm_total=0.484559
Training: 100%|██████████| 256/256
```

The job writes its results to `runs/`:

*   **`runs/checkpoints/quickstart/`**: The two latest checkpoints, `save-200` and `save-256`.
*   **`runs/export/`**: The trained weights as `.safetensors` files.

If you start the job again, it loads `save-256` and logs `Training is already complete, nothing to do`. See [Running Jobs](../guides/running.md) for checkpoints and resuming.

## The Code

```python
--8<-- "example/quickstart/train.py"
```

The script has these parts:

*   **`ProjectConfig`**: One Pydantic model for the whole YAML file. Pydantic validates the file before the job starts. See [Configuration](../loop/config.md).
*   **`RegressionSample`, `RegressionBatch`, `RegressionInput` and `RegressionState`**: Dataclasses for one sample of the dataset, the microbatch that `collate()` builds, what the model receives and what `compute_loss()` needs besides the model output.
*   **`RegressionDataset`**: A standard PyTorch `Dataset`. `AutoDataProvider` shards it across data-parallel ranks, batches it with the `collator` and groups the microbatches into steps. See [Data Loading](../loop/interfaces/data.md).
*   **`MLP`**: A standard `nn.Module`. It implements `reset_parameters()`, because d9d builds the model on the `meta` device and initializes it on the GPU. Its `forward()` takes `inputs` and `shared`, like every [pipeline stage](../models/pipeline_parallelism.md). See [Model Design](../models/model_design.md).
*   **`MLPProvider`**: Builds the model, distributes it and returns the [state mappers](../model_states/mapper.md) for loading and export. `parallelize_hsdp()` shards the model along `dp_cp_shard` and replicates it along the other dimensions of the mesh slice. A job on one GPU does not call `parallelize_model_stage()`. See [Horizontal Parallelism](../models/horizontal_parallelism.md) and [Model Definition](../loop/interfaces/model.md).
*   **`RegressionTask`**: Splits each microbatch into the model input and the state, and computes the loss. The five type parameters of `TrainTask` serve only type checking. See [User Tasks](../loop/interfaces/task.md).
*   **`main()`**: `TrainingConfigurator` wires the parts together. `configure()` builds the job, `train()` runs it and `export()` saves the weights. `load_checkpoint=False` exports the weights that are in memory after training, without loading the latest checkpoint first. See [Training Loop](../loop/train.md).

## The Configuration

```yaml
--8<-- "example/quickstart/config.yaml"
```

The job makes one pass over the 65,536 samples: 256 steps of 256 samples.

`trainer` holds only the required fields. The other fields keep their defaults. For a first job of your own, you change these:

*   **`run.name`**: The name of the run and of its checkpoint directory.
*   **`schedule.total_steps`**: The number of steps. Without it, the job makes one pass over the data.
*   **`checkpointing`**: Where and how often the job saves checkpoints.
*   **`logging.tracker`**: Where the logged values go. `log` writes them to the log, `aim` sends them to Aim and needs `d9d[aim]`, and `null` drops them. See [Experiment Tracking](../tracker/index.md).

[Configuration](../loop/config.md) describes every field and its default.

## Next Steps

*   **Your own job**: [Write Your Own Job](../guides/write_your_own_job.md) shows what to replace in this script, and [How d9d Works](../concepts/how_d9d_works.md) explains how the parts of a job fit together.
*   **More GPUs**: [Running Jobs](../guides/running.md) shows how to launch the same script on several GPUs, and [Choosing Parallelism](../concepts/parallelism.md) compares the strategies.
*   **A real model**: [Fine-Tune a Hugging Face Model](../guides/finetune_huggingface.md) loads Qwen3 from a Hugging Face checkpoint, trains it and exports it back.
