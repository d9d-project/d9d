# Model Heads

## About

The `d9d.module.block.head` package provides task heads. A task head turns backbone hidden states into a typed output. Each head takes the hidden states and its own shared input, such as labels or a pooling mask.

## Heads

### Causal Language Modelling

`SplitLanguageModellingHead` computes the per-token cross-entropy loss for causal language modeling. Labels equal to `LM_IGNORE_INDEX` are ignored.

It uses the fused linear cross-entropy kernel from [Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy) ([paper](https://arxiv.org/abs/2411.09009)), so it does not materialize the full logits tensor. It requires the `d9d[cce]` extra.

The vocabulary can consist of several independent splits, like in [`SplitTokenEmbeddings`](./embedding.md).

### Classification

`ClassificationHead` applies dropout and a linear projection, and returns fp32 logits.

### Embedding

`EmbeddingHead` returns fp32 embeddings, with an optional linear projection and L2 normalization.

Both `ClassificationHead` and `EmbeddingHead` take an optional `pooling_mask` that selects the tokens to use. You can build it with `d9d.dataset.token_pooling_mask_from_attention_mask`.

## Usage

```python
import torch

from d9d.module.block.head import LM_IGNORE_INDEX, SequenceCausalLMHeadShared, SplitLanguageModellingHead

head = SplitLanguageModellingHead(
    split_vocab_size={"orig": 151936, "special": 8},
    split_order=["orig", "special"],
    hidden_size=2048,
).to(device="cuda", dtype=torch.bfloat16)

hidden_states = torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
labels = torch.randint(0, 151944, (2, 16), device="cuda")

output = head(hidden_states, SequenceCausalLMHeadShared(labels=labels))
loss = output.logps[labels != LM_IGNORE_INDEX].mean()
```

## API Reference

::: d9d.module.block.head
