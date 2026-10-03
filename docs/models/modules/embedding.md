# Embeddings

## About

The `d9d.module.block.embedding` package provides the `SplitTokenEmbeddings` token embedding layer. It is built from several named embedding tables, called splits. Each split owns a contiguous range of the global vocabulary, in the order given by `split_order`.

## Use Cases

*   **Regular token embedding:** use a single split with the full vocabulary size.
*   **Prompt tuning:** add new tokens to your tokenizer and use two splits. The first split holds the original token embeddings, and the second one holds the new prompt tokens. Train only the `nn.Embedding` of the second split.

The [`SplitLanguageModellingHead`](./head.md) uses the same splits for the output vocabulary.

## Usage

```python
import torch

from d9d.module.block.embedding import SplitTokenEmbeddings

embeddings = SplitTokenEmbeddings(
    split_vocab_size={"orig": 151936, "prompt": 16},
    split_order=["orig", "prompt"],
    hidden_size=2048,
)
embeddings.reset_parameters()

# Train only the new prompt tokens.
embeddings.token_embedding["orig"].requires_grad_(False)

# Indices 151936 and above belong to the "prompt" split.
input_ids = torch.tensor([[1, 2, 151936, 151937]])
hidden_states = embeddings(input_ids)  # (1, 4, 2048)
```

## API Reference

::: d9d.module.block.embedding
