# d9d and Other Frameworks

## About

This page compares how distributed training frameworks are built, not which features they list. Features converge: a strategy that one framework lacks today usually arrives next year. Design decisions stay. They decide where a new model, parallel layout or training method lives, and how fast you learn that such a change is wrong. The second question matters more when an AI coding agent writes the change. The page answers each question in one part, based on the code of each framework in October 2026 (see [Method](#method)).

## The Frameworks

The frameworks differ first in what you start from:

*   **Recipes for known models**: You pick a model from the catalogue of the framework and start from its config.
    *   **Megatron-LM**: NVIDIA's reference training application. It is built on **Megatron-Core**, NVIDIA's library of parallel transformer layers. **Megatron Bridge**, a separate library, adds recipes for more models and converts checkpoints to and from Hugging Face.
    *   **torchtitan**: The PyTorch team's training platform, with models such as Llama 3, Qwen3 and DeepSeek V3.
    *   **OLMo-core**: The library and scripts that the Allen Institute for AI trains its OLMo models with.
    *   **nanotron**: A Hugging Face application with configs for Llama, Qwen and StarCoder2 pre-training.
*   **Recipes for training methods**: You pick a training method and apply it to a Transformers model.
    *   **Halo**: White Circle's framework with SFT, preference, GRPO, distillation, classification and embedding training. It is built on the Transformers and TRL trainers.
*   **A wrapper around your loop**: You bring the model and the training loop, and the framework distributes them.
    *   **Accelerate**: A Hugging Face library that runs your PyTorch training loop on several GPUs.
    *   **DeepSpeed**: A library that wraps your model in an engine and shards its training state with ZeRO.
    *   **Megatron-Core**: Also a wrapper, when you call it as a library and write the loop yourself.
*   **Parts that you compose**: You write the model and the job in your own project.
    *   **d9d**: A library of parts, e.g. the training loop, the parallelism strategies and the state mappers, that you compose into a job.

A recipe framework is the fastest start while you stay inside its catalogue. The further a change goes from the catalogue, the more the design of the framework matters. The rest of this page compares the design.

## Where a Change Lives

A change that lives in your project is easy to review, and an update of the framework does not touch it. These sections show where each kind of change lives in each framework.

### Where Parallelism Lives

Where the parallelism code lives is the central design decision. To shard one submodule differently, you change the code in that place.

| Where | Frameworks | How you shard one submodule differently |
|:------|:-----------|:----------------------------------------|
| In the model code | Megatron-Core, OLMo-core, nanotron | Change the model: pick another parallel layer class in its `ModuleSpec` (Megatron-Core) or in its code (nanotron), or override the `apply_tp()` method of its block (OLMo-core). |
| In the model's config tree | torchtitan | Set another `ShardingConfig` on that submodule, in the model's `sharding.py` or from your own package with `@override`. Forward methods with SPMD annotations or parallel linear layers must still match it. |
| Around the whole model | DeepSpeed, Accelerate | Not with ZeRO or FSDP: they apply one strategy to the whole model. For tensor parallelism, match the submodule by parameter-name regex (DeepSpeed) or by the Transformers `tp_plan` (Accelerate). |
| In replacement modules, matched by class name | Halo | Not per submodule: one strategy applies to the whole run. Halo swaps Transformers modules by class name. |
| In your code, as a call per submodule | d9d | Call another `parallelize_*` function on that submodule in your model provider, or write your own. |

In most of these frameworks, the model code knows the parallel layout. In Megatron-Core, a layer reads the tensor-parallel group in its constructor and computes its share of attention heads. In torchtitan, some `forward()` methods carry SPMD annotations for the `dp`, `cp` and `tp` mesh axes. Its parallel linear layers also run their collectives in `forward()`.

In d9d, the model provider is a class in your job that builds the model and calls the `parallelize_*` functions. Some d9d layers have a distributed code path, e.g. the DeepEP token exchange of `MoELayer`, and the `parallelize_*` function switches it on. Every parameter that such a function distributes becomes a `DTensor`, which records in its placements how the tensor is split across GPUs. Gradient synchronization, gradient clipping and checkpointing read these placements, so they need no code specific to your model.

### Who Owns the Model

If the framework owns the model class, some changes to the model need changes to the framework.

*   **The framework's model classes**: Megatron-Core models subclass `MegatronModule` and are built from `ModuleSpec` trees. Every stateful torchtitan module must inherit its `Module` base class. nanotron models subclass `NanotronModel` and are built from its tensor-parallel layers and `PipelineBlock` wrappers.
*   **Hugging Face Transformers**: DeepSpeed, Accelerate and Halo train Transformers models as they are, with no conversion. ZeRO, DDP and FSDP work with any module. Tensor and expert parallelism in these frameworks rely on Transformers module names and conventions. Halo relies on these conventions throughout.
*   **You**: In d9d, the model is a class in your project. It implements two methods, `reset_parameters()` and `forward(inputs, shared)`, and inherits no framework class. The `shared` argument carries data that every pipeline stage receives.

Pipeline parallelism shows the difference. torchtitan builds the whole model and deletes the modules of other stages by name. Its default split expects modules named `tok_embeddings`, `layers.N`, `norm` and `lm_head`. The forward and the initialization of the model must work with deleted layers. DeepSpeed needs the model as a flat list of `LayerSpec` objects and splits it evenly or by parameter count. Megatron-Core and nanotron models know their stage, e.g. through the `pre_process` and `post_process` flags of Megatron-Core.

In d9d, the model gets its stage when it is built, and it builds only the layers of that stage. A model with several heads or an unusual layer order splits the way its own code says.

### Where the Checkpoint Map Lives

To start from a Hugging Face checkpoint, a framework maps its tensors to the parameters of the training model. To export the model, it maps them back.

*   **An offline converter**: DeepSpeed merges its per-rank files into one Hugging Face checkpoint with an offline script. OLMo-core converts a Hugging Face checkpoint with a script that loads the whole model into memory.
*   **A map per model, applied while loading**: Megatron Bridge and torchtitan keep a map for each model. Megatron Bridge streams the weights parameter by parameter. torchtitan reads the Hugging Face checkpoint through PyTorch Distributed Checkpoint (DCP), so each rank reads only its shards.
*   **The Hugging Face format as the training checkpoint**: Halo saves standard Hugging Face SafeTensors, so it needs no map.
*   **A mapper from your model provider**: In d9d, your model provider returns a graph of [state mappers](../model_states/mapper.md) for the model it builds, and another one for export. d9d streams the checkpoint through the graph one file at a time.

### Who Owns the Loop

To add a training method, you add a flag, a subclass or a component. The owner of the loop sets which one, and the way a framework adds features sets how readable its code stays.

*   **An application that you clone**: In Megatron-LM, command-line flags control one `train()` function of about 1,000 lines. Each new method becomes a flag and a branch in the loop. Megatron-Core checks the combinations of features in one `TransformerConfig.__post_init__` method of over 2,000 lines. nanotron expects you to clone the repository and subclass its trainer.
*   **A loop that you write**: With Megatron-Core, Accelerate or DeepSpeed, you write the loop. In Accelerate and DeepSpeed, the backend type decides what `prepare()` or an engine call does. The `DeepSpeedEngine` class has about 5,500 lines. With pipeline parallelism, the DeepSpeed engine runs the whole step in `train_batch()`.
*   **The Transformers trainer**: Halo stacks mixins onto the Transformers and TRL trainers and patches methods of their `Accelerator` at runtime. It picks its strategy in one `if` chain.
*   **A loop that you configure**: torchtitan and OLMo-core provide the loop. torchtitan builds every part from a dataclass config tree, and an external package can replace any node by its fully qualified name. A new step order needs a subclass of its `Trainer`. OLMo-core extends the loop with callbacks and a replaceable train module.
*   **A loop of small components**: d9d provides the loop as a sequence of small components, e.g. the gradient manager, the clipper and the checkpointer. You add a model, task, data pipeline or optimizer as a class in your project, behind a typed interface. An event bus calls your code at fixed points of the loop. [How d9d Works](./how_d9d_works.md#the-training-loop) shows the step loop of `Trainer.train()`.

## How You Know a Change Is Right

The earlier a check turns a mistake into an error, the cheaper the mistake, whether you or an AI coding agent made it. These sections start with what a check can see, then follow the checks in the order they run.

### What a Component Can See

A type checker and a test see only what a component receives. Global state and patches hide the rest.

*   **Global process groups**: Megatron-Core keeps its process groups in module-level globals in `parallel_state`, and layers fall back to them when you pass no groups. Megatron-Core is moving to explicit groups and warns about this fallback in its tensor-parallel layers. DeepSpeed also keeps global process groups, and OLMo-core keeps a global device mesh.
*   **Patches**: While ZeRO-3 builds the model, DeepSpeed patches PyTorch, e.g. `torch.empty`, `Tensor.__new__` and the `__init__` of every `nn.Module` subclass. Halo patches Transformers functions at runtime, e.g. the attention dispatch.
*   **Implicit state**: Accelerate shares state through singletons and passes the launch configuration through environment variables. torchtitan keeps the active device mesh in a thread-local registry that model code reads.
*   **An explicit context**: In d9d, each part gets one `DistributedContext` through its constructor. Layers get their process groups from the `parallelize_*` functions, and d9d does not patch PyTorch functions.

### What the Type Checker Sees

A type checker reports a wrong call while you write the code, before any job runs. The frameworks check their own code as follows:

| | Type checking |
|:--|:--------------|
| Megatron-Core | None |
| DeepSpeed | None |
| torchtitan | pyrefly, default rules, without experiments and RL |
| OLMo-core | mypy, not strict |
| Halo | pyright, basic mode |
| Accelerate | None |
| nanotron | None |
| d9d | ty, default rules |

In d9d, the type checker also sees your code. The interfaces that you implement are generic. A task declares its batch, input and state types, e.g. `TrainTask[RegressionBatch, RegressionInput, None, torch.Tensor, RegressionState]`, and every hook gets a context of these types. A wrong field in `compute_loss()` is then a type error in your code.

### When a Config Mistake Shows Up

A mistake in the configuration either stops the job before it starts, or silently changes what the job does.

*   **Flags and environment variables**: Megatron-LM has about 900 command-line flags, and it generates some of them from dataclasses. A newer dataclass config is replacing them, and both systems exist side by side. Accelerate takes dataclasses in the script. Its launcher passes the launch settings to the library through environment variables.
*   **Dictionaries read with defaults**: DeepSpeed reads the top level of its JSON file with `dict.get`, so a misspelled top-level key is ignored. It validates some sections, e.g. `zero_optimization`, with Pydantic.
*   **Dataclasses**: torchtitan and OLMo-core describe the job as a tree of dataclasses, and each dataclass builds the object it configures. Halo parses YAML and command-line overrides into Transformers argument dataclasses. In torchtitan and Halo, a misspelled key raises an error.
*   **Validated models**: d9d describes the job with Pydantic models and validates the whole configuration when you load it. A misspelled key fails. Each choice, e.g. the optimizer or the pipeline schedule, is a discriminated union, so a key of another choice also fails:

    ```yaml
    optimizer:
      name: adamw
      lr: 1.0e-3
      state_dtype: float32  # A key of stochastic_adamw.
    ```

    ```text
    adamw.state_dtype
      Extra inputs are not permitted
    ```

## What This Adds Up To

In d9d, a change stays small and checkable. The model is a class in your project, and the parallel layout is a function call in your model provider. A new task, data pipeline or optimizer is a component behind a typed interface. Mistakes show up early. A misspelled config key fails at load, and a wrong field in a task is a type error.

This matters more when an AI coding agent writes the change. An agent writes code fast, but it learns only from the errors it sees. A mistake that raises no error gives it nothing to react to. In d9d, the change also stays in a few files of your project, so you review your own code, not a fork.

## Trade-offs

d9d makes choices that cost something:

*   **A Transformers model does not run as it is**: For a model outside the [catalogue](../models/model_catalogue/index.md), you must write a d9d model, a state mapper and a model provider. Accelerate and Halo train Transformers models without this step.
*   **Pipeline parallelism needs model support**: The model builds only the layers of its stage and describes the tensors it sends to the next stage. You write this code for every model.
*   **The step order is fixed**: Providers, tasks and events change what a step does, but not the order of its parts. A training method with a different step order, e.g. several optimizer steps per batch, requires changes in d9d.

## Which Framework to Choose

Start from what you change:

*   **Nothing, and your model is in a catalogue**: Choose a recipe framework by its catalogue and your hardware.
    *   **Megatron-LM or Megatron Bridge**: Transformer, MoE or hybrid Mamba models at the largest scale on NVIDIA GPUs, with tuned layers, e.g. for FP8.
    *   **torchtitan**: Models such as Llama 3, Qwen3 or DeepSeek V3, pre-trained or trained with RL and vLLM generation, on NVIDIA or AMD GPUs.
    *   **OLMo-core**: OLMo-style language models, with the scripts that trained the OLMo models.
    *   **nanotron**: A small pre-training framework to read and to fork.
*   **A training method on a Transformers model**: Choose Halo for ready-made recipes such as SFT, DPO or GRPO.
*   **A model and a loop that you already have**: Choose a wrapper.
    *   **Accelerate**: To run your PyTorch loop on more GPUs with few changes.
    *   **DeepSpeed**: When your model fits in GPU memory only with ZeRO offload to CPU or NVMe, or when you train on accelerators other than NVIDIA GPUs.
*   **The model, the parallelism or the training method itself**: Choose d9d. Each change stays in your own code and fails early, before a long run.

## Method

We read the code of each framework at these commits and checked the claims of its documentation against the code. Dates are commit dates in UTC.

| Framework | Commit | Date |
|:----------|:-------|:-----|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), with Megatron-Core | `ef8cd10` | 2026-10-03 |
| [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | `3255e67` | 2026-10-05 |
| [torchtitan](https://github.com/pytorch/torchtitan) | `838e696` | 2026-10-03 |
| [OLMo-core](https://github.com/allenai/OLMo-core) | `5f6f58a` | 2026-10-03 |
| [nanotron](https://github.com/huggingface/nanotron) | `fb0747b` | 2026-09-23 |
| [Accelerate](https://github.com/huggingface/accelerate) | `01c73fb` | 2026-10-01 |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | `583fc9c` | 2026-10-03 |
| [Halo](https://github.com/whitecircle/halo) | `0bc3a22` | 2026-10-02 |
