# d9d

**d9d** is a distributed training framework that is built on top of PyTorch 2.0. It aims to be hackable and efficient.

## Why another framework?

Distributed training frameworks such as **Megatron-LM** are monolithic the way you could run a script from the command line to train any of *predefined* models, using any of *predefined* regimes and any of *predefined* distribution plans. While powerful, these systems can be difficult to hack and integrate into novel research workflows. Their focus is often on providing a complete, end-to-end solution, which can limit flexibility for those who want to experiment.

Another point is that creating your own distributed training solution from scratch could be tricky since you have to implement so many low-level things such as distributed weight loading that are usually identical between various setups, and you have to manually tackle with common performance issues.

**d9d** was designed to fill a gap between monolithic frameworks and homebrew setups to provide a modular yet effective solution for distributed training.


## What d9d is and isn't?

In terms of core concept d9d:

* is a pluggable framework for implementing distributed training regimes for your deep learning models
* aims to provide building blocks and core infrastructure with clear interfaces that may be composed and implemented in your own way, so it's as hackable as possible
* is **not** an all-in-one platform for setting up pre-training and post-training like **torchtitan**, **Megatron-LM** and **torchforge** that could be ran from command line

In terms of code base d9d:

* attempts to provide a clear codebase without typical problems of a mature deep learning project (such as `if torch.__version__ == ...: ... else ...`)
* will **not** maintain backward compatibility with older PyTorch versions and older hardware to keep everything simple

* aims to use fancy new PyTorch 2.0 APIs like `DTensor` and `DeviceMesh` as much as possible
* is eager to add something not fancy nor PyTorch-native if it will improve performance, for instance, we implement MoE layer using communications from **DeepEp**, reindexing kernels from **Megatron-LM** and some custom grouped-GEMM implementation.

In terms of efficiency d9d:

* attempts to maintain high efficiency and high MFU for mid-sized setups
* but still may be slower than **Megatron-LM** in some cases - feel free to open issues if you encounter any performance issues


## Features

### Supported

### To Do
