---
title: Model Heads
---

# Model Heads

## About

The `d9d.module.block.head` package handles the model heads.

## Features

### Causal Language Modelling

`SplitLanguageModellingHead` provides a causal language modelling head that computes per-token logprobs. 

It uses efficient fused Linear-Cross-Entropy kernel from the [Cut-Cross-Entropy](https://arxiv.org/abs/2411.09009) project and avoids full logit tensor materialization.

Supports vocab split to multiple independent splits following the [`SplitTokenEmbeddings`](embedding.md) embedding implementation.

::: d9d.module.block.head
    options:
        show_root_heading: true
        show_root_full_path: true
