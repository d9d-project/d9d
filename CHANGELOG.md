# CHANGELOG

<!-- version list -->

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
