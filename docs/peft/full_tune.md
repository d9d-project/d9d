---
title: Full Fine-Tuning
---

# Full Fine-Tuning

## About

The `d9d.peft.full_tune` package allows you to integrate standard fine-tuning into the PEFT workflow. It does not alter the model architecture. Instead, it uses regex patterns to identify specific modules (e.g., Norm layers or specific Heads) and unfreezes their parameters.

This is particularly useful when combined with other PEFT methods via [Stacking](./stack.md), allowing for hybrid training strategies (e.g., LoRA on Attention + Full Tune on LayerNorm).

::: d9d.peft.full_tune
    options:
        show_root_heading: true
        show_root_full_path: true

```
