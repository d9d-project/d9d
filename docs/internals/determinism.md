# Determinism

## About

The `d9d.internals.determinism` package seeds the random number generators (RNG) of all distributed processes. `set_seeds` seeds Python, NumPy, PyTorch and the `DTensor` RNG. By default, ranks in different pipeline stages get different seeds. Ranks along all other mesh dimensions share a seed.

!!! warning "Internal API"
    If you use the standard d9d training or inference loop, you do not need to call this package. d9d seeds the RNGs at startup. This page is for users who extend d9d.

## API Reference

::: d9d.internals.determinism
