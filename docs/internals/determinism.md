---
title: Determinism
---

# Determinism

## About

The `d9d.internals.determinism` package handles the initialization of Random Number Generators (RNG) across distributed processes.

!!! warning "Warning:" 
    If you are utilizing the standard `d9d` training or inference infrastructure, you **do not** need to call these functions manually. The framework automatically handles seed initialization during startup. This package is primarily intended for users extending `d9d`.

::: d9d.internals.determinism
    options:
        show_root_heading: true
        show_root_full_path: true
