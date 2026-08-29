# Embeddings

## About

The `d9d.module.block.embedding` package provides enhanced embedding layers.

## Features

### Split Token Embeddings

You can use the `SplitTokenEmbeddings` module:

* **Regular Token Embedding Layer**: Specify a single split with global vocab size.
* **For Prompt Tuning**: Add additional tokens to your Tokenizer and specify two splits - first one will be original token embeddings, second one will be newly added learnable prompt tokens. Unfreeze only `nn.Embedding` module that is related to the second split.

### Media Embedding Merge

`merge_media_embeddings` replaces placeholder token embeddings with the embeddings produced by a
modality encoder (see [Multimodality](../multimodality.md)).

::: d9d.module.block.embedding
