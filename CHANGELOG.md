# CHANGELOG

<!-- version list -->

## v0.26.0 (2026-11-11)

### Bug Fixes

- Test enviroment ([#121](https://github.com/d9d-project/d9d/pull/121),
  [`339bccc`](https://github.com/d9d-project/d9d/commit/339bcccfefe748f9f936f3d7dc7b4d9988afbaaf))

- **autograd**: Keep all weight gradients in the split backward pass
  ([#147](https://github.com/d9d-project/d9d/pull/147),
  [`866bcdc`](https://github.com/d9d-project/d9d/commit/866bcdc8325a65958aea2930c852b2c0f09c9d55))

- **data**: Make the BufferSortedDataset order depend only on the seed and the buffer index
  ([#138](https://github.com/d9d-project/d9d/pull/138),
  [`6f9b837`](https://github.com/d9d-project/d9d/commit/6f9b837cf510b62e0fce9cbed8ec7607eb0ea10c))

- **loop**: Keep tied parameters and buffers through materialization
  ([#145](https://github.com/d9d-project/d9d/pull/145),
  [`35aed27`](https://github.com/d9d-project/d9d/commit/35aed27f78aa8dbe64a91435d4f815c95e5f552d))

- **loop**: Save and load checkpoints outside inference mode
  ([#160](https://github.com/d9d-project/d9d/pull/160),
  [`26919ef`](https://github.com/d9d-project/d9d/commit/26919ef796d8d36b3450c85b722ec6da39ce033a))

- **loop**: Validate dtype names in configs ([#135](https://github.com/d9d-project/d9d/pull/135),
  [`509a44d`](https://github.com/d9d-project/d9d/commit/509a44d996cace503dcf3efeee501be92def2f1d))

- **model_state**: Keep inputs and outputs apart when Sequential merges groups
  ([#172](https://github.com/d9d-project/d9d/pull/172),
  [`45d3f3d`](https://github.com/d9d-project/d9d/commit/45d3f3d6aa1d82c47bed8f04a008c9c72585c162))

- **nn**: Keep the dtype of dt_bias when MambaDecayGate resets parameters
  ([#171](https://github.com/d9d-project/d9d/pull/171),
  [`954a985`](https://github.com/d9d-project/d9d/commit/954a98532026e8ad1968e016cbafd483dd2d27af))

- **optim**: Seed StochasticAdamW from an explicit seed
  ([#139](https://github.com/d9d-project/d9d/pull/139),
  [`abbe775`](https://github.com/d9d-project/d9d/commit/abbe77570e3923eb677ee9b9ff92ce48acefbe8e))

- **parallel**: Reject unsupported placements in the gradient norm
  ([#136](https://github.com/d9d-project/d9d/pull/136),
  [`e61248a`](https://github.com/d9d-project/d9d/commit/e61248a5d7d4b134db82f0dc79b1cfecf6c7ed42))

- **pipelining**: Release the forward cache after each step
  ([#146](https://github.com/d9d-project/d9d/pull/146),
  [`691e7a4`](https://github.com/d9d-project/d9d/commit/691e7a4fc01737d670976e73e87297f310900bdc))

### Documentation

- Add DEP writing rules and a dep agent skill ([#140](https://github.com/d9d-project/d9d/pull/140),
  [`580bbd2`](https://github.com/d9d-project/d9d/commit/580bbd2c89fe0d1d743b0d0fac696723deda7159))

- Add sections and rules for PR descriptions ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- Add testing rules, design principles and a review agent skill
  ([#159](https://github.com/d9d-project/d9d/pull/159),
  [`ba8f6da`](https://github.com/d9d-project/d9d/commit/ba8f6dadd5abff7c05c9fff95c8ee0d7cf62f413))

- Add the design principle that the owner of a resource closes it
  ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- Add the implement skill for agents ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- Extend the contributing rules, the DEP template and the agent skills
  ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- **dep**: Add DEP-0011 scoped random number generation
  ([#141](https://github.com/d9d-project/d9d/pull/141),
  [`14c2efe`](https://github.com/d9d-project/d9d/commit/14c2efe18be7019224fe9654aaad986e2ca5c9aa))

- **dep**: Add DEP-0012 rework of DistributedContext
  ([#161](https://github.com/d9d-project/d9d/pull/161),
  [`a4a51ff`](https://github.com/d9d-project/d9d/commit/a4a51ffa4f3b6d236e905e3e9cb9a6cc3545e962))

- **dep**: Align DEP-0012 with its implementation
  ([#161](https://github.com/d9d-project/d9d/pull/161),
  [`a4a51ff`](https://github.com/d9d-project/d9d/commit/a4a51ffa4f3b6d236e905e3e9cb9a6cc3545e962))

- **dep**: Check claims about PyTorch with a prototype in the dep skill
  ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- **dep**: Keep the groups of DEP-0012 on an error and read the coordinates from the mesh
  ([#161](https://github.com/d9d-project/d9d/pull/161),
  [`a4a51ff`](https://github.com/d9d-project/d9d/commit/a4a51ffa4f3b6d236e905e3e9cb9a6cc3545e962))

- **dep**: List only the alternatives a reviewer would raise and say what deferred work gives
  ([#162](https://github.com/d9d-project/d9d/pull/162),
  [`16a728b`](https://github.com/d9d-project/d9d/commit/16a728bc654dc3c1008dfafa5a96988c52611009))

- **dep**: Mark DEP-0012 as implemented in #166
  ([#166](https://github.com/d9d-project/d9d/pull/166),
  [`ba92108`](https://github.com/d9d-project/d9d/commit/ba9210813b47bc940adebe3796e9c2177ce72046))

- **dist_context**: Describe init_dist_context(), the new domains and the mesh configuration
  ([#166](https://github.com/d9d-project/d9d/pull/166),
  [`ba92108`](https://github.com/d9d-project/d9d/commit/ba9210813b47bc940adebe3796e9c2177ce72046))

- **pipelining**: Describe the step handler ([#154](https://github.com/d9d-project/d9d/pull/154),
  [`5ee8e82`](https://github.com/d9d-project/d9d/commit/5ee8e82b45325607a614900400f61ead16aaebcb))

### Features

- **autograd**: Add needs_input_grad() for split backward
  ([#134](https://github.com/d9d-project/d9d/pull/134),
  [`2ccccbc`](https://github.com/d9d-project/d9d/commit/2ccccbcffac129bcc8105f50202123190fbc0afb))

- **dist_context**: Add declarative domains and split the mesh parameters into degrees and sections
  ([#166](https://github.com/d9d-project/d9d/pull/166),
  [`ba92108`](https://github.com/d9d-project/d9d/commit/ba9210813b47bc940adebe3796e9c2177ce72046))

- **dist_context**: Build the context with init_dist_context() and move all users to the new domains
  ([#166](https://github.com/d9d-project/d9d/pull/166),
  [`ba92108`](https://github.com/d9d-project/d9d/commit/ba9210813b47bc940adebe3796e9c2177ce72046))

- **dist_context**: Implement DEP-0012 rework of DistributedContext
  ([#166](https://github.com/d9d-project/d9d/pull/166),
  [`ba92108`](https://github.com/d9d-project/d9d/commit/ba9210813b47bc940adebe3796e9c2177ce72046))

- **pipelining**: Replace the step callbacks with PipelineStepHandler
  ([#154](https://github.com/d9d-project/d9d/pull/154),
  [`5ee8e82`](https://github.com/d9d-project/d9d/commit/5ee8e82b45325607a614900400f61ead16aaebcb))

- **rng**: Seed each call of user code from its own seed scope
  ([#157](https://github.com/d9d-project/d9d/pull/157),
  [`b63324a`](https://github.com/d9d-project/d9d/commit/b63324acae5dab958fe4ee4aab1b5fb9d78a0ed9))

### Performance Improvements

- **pytree**: Find unregistered dataclasses without a Python pass over all leaves
  ([#137](https://github.com/d9d-project/d9d/pull/137),
  [`915833f`](https://github.com/d9d-project/d9d/commit/915833ffbc00bf630581a6f1c9fd3c82d1b91540))


## v0.25.0 (2026-10-29)

### Bug Fixes

- **grad_sync**: Clean up after a failed step and reject unused parameters
  ([`ba139a7`](https://github.com/d9d-project/d9d/commit/ba139a766ce2d79f9ce08afeff07dd6d2f000ada))

- **grad_sync**: Raise clear errors on misuse of the synchronizer
  ([`ba139a7`](https://github.com/d9d-project/d9d/commit/ba139a766ce2d79f9ce08afeff07dd6d2f000ada))

- **grad_sync**: Start bucket reductions in the same order on every rank
  ([`ba139a7`](https://github.com/d9d-project/d9d/commit/ba139a766ce2d79f9ce08afeff07dd6d2f000ada))

- **grad_sync**: Start bucket reductions in the same order on every rank and introduce additional
  safety checks
  ([`ba139a7`](https://github.com/d9d-project/d9d/commit/ba139a766ce2d79f9ce08afeff07dd6d2f000ada))

- **grad_sync**: Wait for gradient writes on all streams before the all-reduce
  ([`ba139a7`](https://github.com/d9d-project/d9d/commit/ba139a766ce2d79f9ce08afeff07dd6d2f000ada))

### Features

- Add the d9d-kernels package and move to a uv workspace
  ([#123](https://github.com/d9d-project/d9d/pull/123),
  [`9bd8d3e`](https://github.com/d9d-project/d9d/commit/9bd8d3e6ab08613394121c43233cc55d414f5ae7))


## v0.24.0 (2026-10-15)

### Documentation

- **moe**: Document the grouped GEMM backends and the new extras
  ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

### Features

- **kernel**: Add torch and quack grouped GEMM kernels
  ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

- **loop**: Log to the tracker without waiting for the GPU
  ([#99](https://github.com/d9d-project/d9d/pull/99),
  [`9cbf486`](https://github.com/d9d-project/d9d/commit/9cbf48613fcfb52d021de3d3f4c7a774962f925b))

- **moe**: Add grouped GEMM backend selection ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

- **moe**: Add grouped GEMM backends for the experts
  ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

- **moe**: Run the experts through the grouped GEMM backends
  ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

- **moe**: Split the moe extra into grouped GEMM and expert parallel backend extras
  ([#100](https://github.com/d9d-project/d9d/pull/100),
  [`f65570e`](https://github.com/d9d-project/d9d/commit/f65570e13b924a106f25bb886936542189de811d))

- **tracker**: Add W&B, TensorBoard and MLflow trackers
  ([#101](https://github.com/d9d-project/d9d/pull/101),
  [`237d140`](https://github.com/d9d-project/d9d/commit/237d140e672cb7ce82e087e026a1382a04ce010f))


## v0.23.2 (2026-10-12)

### Bug Fixes

- **autograd**: Restore the gradient directions when the block raises
  ([#88](https://github.com/d9d-project/d9d/pull/88),
  [`39502b1`](https://github.com/d9d-project/d9d/commit/39502b14320bd8b83500cea4b2cfd77de7d46212))

- **head**: Rename SequenceCausalLMOutput.logps to cross_entropy
  ([#90](https://github.com/d9d-project/d9d/pull/90),
  [`e44f5b7`](https://github.com/d9d-project/d9d/commit/e44f5b7ebeb59cdbabe891c38b7e1651fa9e91f5))

- **head**: Rename SequenceCausalLMOutput.logps to nll
  ([#90](https://github.com/d9d-project/d9d/pull/90),
  [`e44f5b7`](https://github.com/d9d-project/d9d/commit/e44f5b7ebeb59cdbabe891c38b7e1651fa9e91f5))

- **head**: Rename the nll field to cross_entropy
  ([#90](https://github.com/d9d-project/d9d/pull/90),
  [`e44f5b7`](https://github.com/d9d-project/d9d/commit/e44f5b7ebeb59cdbabe891c38b7e1651fa9e91f5))

- **loop**: Make prepare_export_model_stage() optional for jobs that never export
  ([#96](https://github.com/d9d-project/d9d/pull/96),
  [`7199c96`](https://github.com/d9d-project/d9d/commit/7199c969b90c5050f2e14f412b19e5837f48ee7d))

- **metric**: Return long targets from OneHotProcessor for (..., 1) targets
  ([#89](https://github.com/d9d-project/d9d/pull/89),
  [`e686310`](https://github.com/d9d-project/d9d/commit/e686310c4b1a54a2a65330f65e901d4ba44a6f55))

- **module**: Keep the snapshot when the mean aggregator collected nothing
  ([#92](https://github.com/d9d-project/d9d/pull/92),
  [`1aeff12`](https://github.com/d9d-project/d9d/commit/1aeff12f86996ad7c099648a12567b109069e073))

- **packaging**: Add the py.typed marker ([#87](https://github.com/d9d-project/d9d/pull/87),
  [`54e03fc`](https://github.com/d9d-project/d9d/commit/54e03fc2ec181b65e2dc67364c7f022259f87114))

### Performance Improvements

- **metric**: Compute AUROC without a host sync ([#97](https://github.com/d9d-project/d9d/pull/97),
  [`b06427f`](https://github.com/d9d-project/d9d/commit/b06427f87563f3bf056a4f9e5133828702bb6cde))


## v0.23.1 (2026-10-11)

### Bug Fixes

- **grad_sync**: Scale all gradients by the loss weight and size buckets by the local shard
  ([#77](https://github.com/d9d-project/d9d/pull/77),
  [`c304691`](https://github.com/d9d-project/d9d/commit/c30469178d764a820df2d8348326373e0d9b004d))

- **grad_sync**: Size buckets by the local gradient shard in the gradient dtype
  ([#77](https://github.com/d9d-project/d9d/pull/77),
  [`c304691`](https://github.com/d9d-project/d9d/commit/c30469178d764a820df2d8348326373e0d9b004d))

- **loop**: Scale gradients created after install() by the total loss weight
  ([#77](https://github.com/d9d-project/d9d/pull/77),
  [`c304691`](https://github.com/d9d-project/d9d/commit/c30469178d764a820df2d8348326373e0d9b004d))

### Documentation

- **grad_sync**: Say that the bucket size counts the local gradient shard
  ([#77](https://github.com/d9d-project/d9d/pull/77),
  [`c304691`](https://github.com/d9d-project/d9d/commit/c30469178d764a820df2d8348326373e0d9b004d))


## v0.23.0 (2026-10-09)

### Bug Fixes

- **pipelining**: Require the pipelining protocol only for multi-stage pipelines
  ([#76](https://github.com/d9d-project/d9d/pull/76),
  [`873b481`](https://github.com/d9d-project/d9d/commit/873b48122300e3ecc550576222edb9b121db765f))

### Features

- **loop**: Add config defaults and restructure the documentation
  ([`d5b8706`](https://github.com/d9d-project/d9d/commit/d5b8706ccb4b6d9337cb7a8e2e065ac488268c1e))


## v0.22.0 (2026-10-04)

### Documentation

- Add writing rules for the whole project
  ([`5f11a9e`](https://github.com/d9d-project/d9d/commit/5f11a9e435b51215fcc740ec9eb045b39596e857))

- Fix code samples and outdated facts, add troubleshooting
  ([`d19b8ed`](https://github.com/d9d-project/d9d/commit/d19b8ed9ca54fbce38c8fb3cefd60f950a2e312b))

### Features

- **tracker**: Add a tracker that writes scalars to the Python logger
  ([`ef6f13a`](https://github.com/d9d-project/d9d/commit/ef6f13a085ce3d8f43cd9e104d6b2581e05bdc9d))


## v0.21.2 (2026-10-02)

### Bug Fixes

- **pipelining**: Replay weight grads in a single pass pinned to the captured grads
  ([`fe422af`](https://github.com/d9d-project/d9d/commit/fe422af5fea5ee8b3fae99f4e0b61bd0ac32eefe))


## v0.21.1 (2026-10-02)

### Bug Fixes

- **pipelining**: Replay captured weight grads on the outputs they belong to
  ([`a7810c8`](https://github.com/d9d-project/d9d/commit/a7810c81e902ed57e6755ec92dc0344fbff9a98b))


## v0.21.0 (2026-10-02)

### Features

- **profiling**: Annotate step phases and profile the prefetch thread
  ([`1df1809`](https://github.com/d9d-project/d9d/commit/1df180975d1e4994dcc167813140950d28310d16))


## v0.20.1 (2026-10-02)

### Bug Fixes

- **loop**: Run the train and inference loops for JobSchedule.total_steps
  ([`102c1ed`](https://github.com/d9d-project/d9d/commit/102c1edfdf68e34148b44b4fbd4d1ec3a954d4fb))


## v0.20.0 (2026-10-02)

### Features

- **loop**: Prefetch data packs to the device on a side CUDA stream
  ([`f92a173`](https://github.com/d9d-project/d9d/commit/f92a17362ba1a875f38f6384c516f2fda36134ec))


## v0.19.4 (2026-10-01)

### Bug Fixes

- **moe**: Use 64-bit row offsets in permute kernels
  ([`f0b85c6`](https://github.com/d9d-project/d9d/commit/f0b85c6a9fd6367019ce6b46f7428751668da21e))


## v0.19.3 (2026-09-28)

### Bug Fixes

- Repair FA4 backend and typing after torch 2.14 bump
  ([`c4bff1d`](https://github.com/d9d-project/d9d/commit/c4bff1dbce6f83c9649a3ff6134e2e101d3b0246))


## v0.19.2 (2026-09-22)

### Bug Fixes

- **model_state**: Load checkpoints without a safetensors index
  ([`dc2c794`](https://github.com/d9d-project/d9d/commit/dc2c7945c114731df39585f819bb5f1189d015d2))

### Documentation

- Add CLAUDE.md so agent guidance actually gets read
  ([`2d51b73`](https://github.com/d9d-project/d9d/commit/2d51b73b213ba6c6e6af49d7b614ba759ebc8162))


## v0.19.1 (2026-09-21)

### Bug Fixes

- **loop**: Export model state when no parallelism is configured
  ([`b3f7049`](https://github.com/d9d-project/d9d/commit/b3f7049bf35d3d977e3a8ed13370bad1c3f5b0dd))


## v0.19.0 (2026-08-18)

### Features

- Introduce multi-head models
  ([`bd7b09e`](https://github.com/d9d-project/d9d/commit/bd7b09edd869e1f7aab0289075cac4dbcca1f082))


## v0.18.0 (2026-07-20)

### Features

- Allow arbitrary PyTrees to be passed between pipeline stages
  ([#41](https://github.com/d9d-project/d9d/pull/41),
  [`5a475f9`](https://github.com/d9d-project/d9d/commit/5a475f9dcb90171672b9c5ab02e8c3e13e81e111))


## v0.17.0 (2026-07-10)

### Documentation

- DEP-0009 Dataflow Refactoring ([#37](https://github.com/d9d-project/d9d/pull/37),
  [`537bb2e`](https://github.com/d9d-project/d9d/commit/537bb2e37fddd32ecacf535139ecc35f2668dc8b))

### Features

- Advanced pytree support ([#40](https://github.com/d9d-project/d9d/pull/40),
  [`bd218db`](https://github.com/d9d-project/d9d/commit/bd218db4f343843bc3b1af7101d92984355c5fc0))


## v0.16.0 (2026-06-16)

### Documentation

- Add DEP-0008 for backend selection mechanism ([#34](https://github.com/d9d-project/d9d/pull/34),
  [`57055fa`](https://github.com/d9d-project/d9d/commit/57055fa23b34e99fc542b8d2bf540dd924b02939))

### Features

- Implement FA4/FA2/TorchSDPA/Eager attention backends with programmatic auto-selection
  ([#35](https://github.com/d9d-project/d9d/pull/35),
  [`e6374be`](https://github.com/d9d-project/d9d/commit/e6374beb7e7d9ce827745ec467cb5766e3a7b34f))


## v0.15.0 (2026-06-04)

### Features

- Implement wake-sleep API for trainer
  ([`b209861`](https://github.com/d9d-project/d9d/commit/b2098618ea2587d08f34ca783047ea007b15669e))


## v0.14.0 (2026-05-21)

### Bug Fixes

- Make deepep imports lazy
  ([`e2cd994`](https://github.com/d9d-project/d9d/commit/e2cd99467192b607a80337ae557ed06c4897bf85))

### Features

- Add embedding models
  ([`fcd8f4e`](https://github.com/d9d-project/d9d/commit/fcd8f4ec0424b881f45f70239c26c62d014ed239))

- Add RoPE scaling techniques
  ([`92d6e0e`](https://github.com/d9d-project/d9d/commit/92d6e0eed4462ec0ee6cbdcff500965653b68bd2))

- Make ModelStateMapperPrefixScope support different prefixes
  ([`3195d80`](https://github.com/d9d-project/d9d/commit/3195d8095f7fe710ed4d6c1691ec66d1b36885df))


## v0.13.1 (2026-04-23)

### Bug Fixes

- Correct GQA/MQA head expansion logic for Q and K
  ([`62b73bd`](https://github.com/d9d-project/d9d/commit/62b73bd38a9111f18c763b7d0acf065f317edb50))

- Introduce RoutingResult and fix probability renormalization
  ([`a1dabe3`](https://github.com/d9d-project/d9d/commit/a1dabe338bd9726a9f1027050a73c81474cc78c0))

- Remove GPU-CPU sync that was happening during bincount() call in MoE layer
  ([`534facc`](https://github.com/d9d-project/d9d/commit/534facc1bf25b9451ae99164d48d9462492f6edb))

- Remove MoE statistics interface from qwen3 decoder layer and module
  ([`99e0379`](https://github.com/d9d-project/d9d/commit/99e0379e4df07a4762054b5f9654d69bdecfc8ad))


## v0.13.0 (2026-04-20)

### Features

- Migrate to Flash Attention 4
  ([`19f8483`](https://github.com/d9d-project/d9d/commit/19f8483286e0303bddfc77f60bd1a415025babba))


## v0.12.0 (2026-04-14)

### Features

- Add partial RoPE to GQA
  ([`4dcfe7e`](https://github.com/d9d-project/d9d/commit/4dcfe7e54f6eabe20e8f5b7984b3a6280cc00dcb))


## v0.11.0 (2026-04-13)

### Features

- Add Gated DeltaNet layer that use fla-core kernels
  ([`e0e466f`](https://github.com/d9d-project/d9d/commit/e0e466f96e79414f66ef84ee57627300c3dfa3ff))

- Add shared expert to MoE block
  ([`0e93f45`](https://github.com/d9d-project/d9d/commit/0e93f45f8820d19a351432ffcfa7ed27a0c6b484))


## v0.10.0 (2026-04-13)

### Features

- Add custom RMSNorm kernel supporting zero-centered weight
  ([`365ad1f`](https://github.com/d9d-project/d9d/commit/365ad1f7027030fc534da462ef2255f5b92f15e3))

- Add optional output gating for grouped query attention
  ([`af3e411`](https://github.com/d9d-project/d9d/commit/af3e4112b2f4f4bf44542593de20f0e847e63a93))


## v0.9.0 (2026-04-08)

### Bug Fixes

- Allow using MoE parallelization function with [EP]=1
  ([`fdc7d6d`](https://github.com/d9d-project/d9d/commit/fdc7d6d7cc100dd9eb7000c2da7edabe92fc01fe))

- Fix huggingface mappers for classification models
  ([`caa2e63`](https://github.com/d9d-project/d9d/commit/caa2e63d1b78f50c50c8355cba20eafe7ad36748))

### Features

- Add model state mappers that convert HuggingFace model states into d9d ones and vice versa
  ([`47bae88`](https://github.com/d9d-project/d9d/commit/47bae88234329da95f4af90141f3ec7a94803f64))

- Make HF model state mappers for Qwen3 MoE support new fused experts format from transformers v5
  ([`cdecc29`](https://github.com/d9d-project/d9d/commit/cdecc29257d835d15ccbd0af99f5bf591875a7b4))


## v0.8.0 (2026-03-22)

### Features

- Add training hooks system based on event bus along with a set of predefined events
  ([`13a76e2`](https://github.com/d9d-project/d9d/commit/13a76e26fc416c7f9bf6dd9ff3b6fc4e6281e97f))


## v0.7.0 (2026-03-18)

### Features

- Add Multi-Latent Attention implementation along with support for different RoPE layouts
  ([`56684bc`](https://github.com/d9d-project/d9d/commit/56684bcb440e68a94a2d6ee8b6ada7d53b414482))

- Add Qwen3 Dense model
  ([`08047bd`](https://github.com/d9d-project/d9d/commit/08047bd82f65ff0b7910bb000ff00edafbeaad81))


## v0.6.0 (2026-03-11)

### Bug Fixes

- Render multi-process model state loading progress using multiple progress bars (fixes #1)
  ([`defea97`](https://github.com/d9d-project/d9d/commit/defea97dafc4f07f29e722e5feb60b299652ffc9))

### Features

- Implement composition-based classificaiton metrics API
  ([`7d7476c`](https://github.com/d9d-project/d9d/commit/7d7476c29db5ad6959eef6f6c47ca06651f72f1a))


## v0.5.4 (2026-03-10)

### Bug Fixes

- Reorder contents in README
  ([`973c7cb`](https://github.com/d9d-project/d9d/commit/973c7cb3010ec4e863da3729b236f05a80b5911a))

### Documentation

- Migrate from mkdocs-shadcn to zensical
  ([`7c131c4`](https://github.com/d9d-project/d9d/commit/7c131c4fb9ba8085494ca0fdbe4ddbbec446b1ae))


## v0.5.3 (2026-03-01)

### Bug Fixes

- Treat checkpoint saving as a post-step action so the progress bars are not shifted by 1 when
  resume training
  ([`f03715f`](https://github.com/d9d-project/d9d/commit/f03715fb53a903fe23d6f71660ae3a3c3bc36ee0))


## v0.5.2 (2026-02-24)

### Bug Fixes

- Fix and simplify stepper semantics (closes GitHub issue #3)
  ([`b01366c`](https://github.com/d9d-project/d9d/commit/b01366c2572ee6d4188ced879841de72fb9ab20e))


## v0.5.1 (2026-02-12)

### Bug Fixes

- Invalid d9d.core.protocol.training definitions
  ([`7a0ab9a`](https://github.com/d9d-project/d9d/commit/7a0ab9af72dcda5a872c58f28614a8834f2ee6c4))

- Migrate to `ty` type checker and fix newly found minor issues found by it
  ([`37c9642`](https://github.com/d9d-project/d9d/commit/37c96429ec51bd3593329c163eec5211d0144864))


## v0.5.0 (2026-02-12)

### Features

- Add head for classification task
  ([`56491a7`](https://github.com/d9d-project/d9d/commit/56491a712af4c1527b603665fdaca234d93c5639))


## v0.4.0 (2026-02-10)

### Features

- Add BinaryAccuracy and BinaryAUROC (uses approximated but effective implementation!) metrics
  ([`15f281b`](https://github.com/d9d-project/d9d/commit/15f281b1f8f2644d102f69b246cd1ce6ea993c1f))


## v0.3.0 (2026-02-09)

### Features

- SumMetric
  ([`c17e243`](https://github.com/d9d-project/d9d/commit/c17e24321105def019bef3c2b30c72eb2b8e1f43))

- Use separate CUDA stream to simplify metric sync & compute
  ([`2d8d877`](https://github.com/d9d-project/d9d/commit/2d8d877cc815a93a1ba36ca2f6754d59adff501f))


## v0.2.4 (2026-02-09)

### Bug Fixes

- Use init timeout for loop dependency injection
  ([`ca0e322`](https://github.com/d9d-project/d9d/commit/ca0e322347a1b2083ad9ca11a98442b456139037))


## v0.2.3 (2026-02-05)

### Bug Fixes

- Allow running the train loop in local mode (no NCCL init at all)
  ([`b0c0a0e`](https://github.com/d9d-project/d9d/commit/b0c0a0eab97470791551fca169dcb485ccc12681))

- BufferedSortedDataset now uses random value as a sorting tie-breaker and sorts data within a pack
  ([`ed1f439`](https://github.com/d9d-project/d9d/commit/ed1f439eb07b19a9b08d745e620cff02a8670d28))

- Non-tensor objects now process correctly using IteratorBatchGroup
  ([`58ec446`](https://github.com/d9d-project/d9d/commit/58ec446f515d91e4fd7376cdc76399903bc4efc4))


## v0.2.2 (2026-02-05)

### Bug Fixes

- Allow zero weight_decay for Stochastic AdamW
  ([`38587a9`](https://github.com/d9d-project/d9d/commit/38587a9eb1d1b9048d11b0c22ee4011b96cabc3a))


## v0.2.1 (2026-02-05)

### Bug Fixes

- Allow empty gradients in GradientManager
  ([`7a63fb6`](https://github.com/d9d-project/d9d/commit/7a63fb615a539f2ad3135d6a307be35aadd2c222))


## v0.2.0 (2026-02-04)

### Bug Fixes

- Swiglu kernel recompilation
  ([`e4cf4f4`](https://github.com/d9d-project/d9d/commit/e4cf4f4cd8270a4187c61702f1647575bdfa2ecb))

### Documentation

- Contribution guide
  ([`b889b09`](https://github.com/d9d-project/d9d/commit/b889b094d9cfed36402b71b2c070d6a1ddb44947))

- The DEP process
  ([`92e0988`](https://github.com/d9d-project/d9d/commit/92e09888fecb798ae5f82f67d24196bed0cfdd23))

### Features

- Allow running pipeline schedules for inference mode with microbatch-level callback
  ([`923e743`](https://github.com/d9d-project/d9d/commit/923e743b948dd45d6425cb739080e93ba01971ba))

- Allow running pipelining API in local setups by adding a special offline pipeline executor
  ([`268fd05`](https://github.com/d9d-project/d9d/commit/268fd057319674f0b88086253780077391f036ad))

- Inference Loop
  ([`6bd6e72`](https://github.com/d9d-project/d9d/commit/6bd6e72fc456087ad0d2c762095724ee5310cb9d))


## v0.1.1 (2026-02-04)

### Bug Fixes

- Ci
  ([`d837a2d`](https://github.com/d9d-project/d9d/commit/d837a2dbba9e1294f84e222b168d8880e1df5e04))


## v0.1.0 (2026-02-04)

- Initial Release
