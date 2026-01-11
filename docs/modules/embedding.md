---
title: Embeddings
---

# Embeddings

## About

The `d9d.module.block.embedding` package provides enhanced embedding layers.

## Features

Currently, this package provides only `SplitTokenEmbeddings` module. You can use this module:

* **Regular Token Embedding Layer**: Specify a single split with global vocab size.
* **For Prompt Tuning**: Add additional tokens to your Tokenizer and specify two splits - first one will be original token embeddings, second one will be newly added learnable prompt tokens. Unfreeze only `nn.Embedding` module that is related to the second split.

::: d9d.module.block.embedding
    options:
        show_root_heading: true
        show_root_full_path: true
