# CHANGELOG

<!-- version list -->

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
