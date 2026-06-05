# Contributing Guide

First off, thank you for considering contributing to **d9d**!

We aim to build a distributed training framework that is **efficient, hackable, and reliable**. To maintain this balance, we adhere to a strong engineering culture: strict type-checking, rigorous linting, and a structured proposal process for major changes.

This document outlines the standards and workflows for contributing to the project.

Before starting work on a major feature, we highly recommend jumping into our [Discord server](https://discord.gg/sNRjDbxVrg) to discuss your approach with the core maintainers!

## Development Setup

**d9d** uses [Poetry](https://python-poetry.org/) for dependency management and packaging. You will need Python 3.11+.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/d9d-project/d9d.git
    cd d9d
    ```

2.  **Install dependencies:**
    ```bash
    # Install all dependencies but optional ones requiring manual builds
    poetry install --without compat-local-overrides
    
    # Install pre-commit hooks
    poetry run pre-commit install
    ```

3. **(Optional) Install dependencies requiring manual builds:**
   ```bash
    # If you want to develop functional requiring optional dependencies
    # that have to be built manually - just build the optional packages
    # (you may find them in pyproject.toml), put into `./packages` directory
    # and run this:
    poetry install --with compat-local-overrides
    ```

## Development Workflow

We provide a `Makefile` to automate common development tasks.

| Command       | Description                                                                                                                   |
|:--------------|:------------------------------------------------------------------------------------------------------------------------------|
| `make test`   | Runs **both** local unit tests and distributed `torchrun` tests. **NOTE:** currently distributed tests require a 8-GPU setup. |
| `make lint`   | Runs `ruff` formatting, `ruff` linter checks and type checking using `ty`.                                                    |
| `make mkdocs` | Starts a local documentation server at `localhost:8081`.                                                                      |

## The DEP Process (Design First)

**d9d** is a foundational tool; architectural mistakes are expensive. Therefore, we follow the **D9D Enhancement Proposal (DEP)** process for significant changes.

**You need a DEP if:**
*   You are making breaking changes to the public API.
*   You are introducing a major new distributed strategy or base module.

**You DO NOT need a DEP if:**
*   You are fixing bugs.
*   You are adding a new model implementation using existing APIs.
*   You are improving documentation or internal performance (without API changes).

👉 **[Read DEP-0001: The DEP Process](./deps/0001-dep-process.md)** for details on how to draft, propose, and implement a DEP.

## Code Quality Standards

We enforce strict quality standards to keep the codebase maintainable.

### Design Principles

They are not enforced by tooling, but PRs that violate them may be asked to change.

* **Composition over inheritance**: components are small single-responsibility classes wired together (see `d9d/loop/component/`). Avoid "God classes" that accumulate unrelated responsibilities, avoid speculative base classes that exist only to hoard "common" code.
* **Define contracts structurally.** Use a `typing.Protocol` for a *trait* - a secondary capability bolted onto a type that already has its own base class (e.g. `ModuleLateInit` on an `nn.Module`), where you want duck-typed conformance without forcing inheritance. Use an `abc.ABC` when the interface *is* the object's primary identity and the hierarchy is the "main" type (e.g. `PipelineSchedule`).
* **No reflection where it can be avoided.** Avoid `getattr` / `hasattr` / `inspect` and string-name dispatch. Prefer an explicit `match`-`case`, a proper interface, or a factory. Reflection is acceptable *only* when introspection is intrinsic to the feature itself - i.e. declarative registration APIs that cannot work without it, such as a `@subscribe`/`@register` decorator wiring handlers by signature.
* **Inject dependencies; don't reach for them.** Components receive their collaborators as constructor arguments and store them as private fields. Don't pull them from globals/singletons or construct them internally - wiring happens at the edges (`d9d/loop/run/`).
* **Reuse before reinventing.** If PyTorch or the stdlib already solves it, use it, rather than hand-rolling an equivalent.
* **No needless indirection.** Don't add a wrapper that only forwards to another function/object without adding meaning. Inline it instead.
* **Validate eagerly, fail fast.** Validate constructor args up front; raise if a method is called outside its required lifecycle scope rather than silently misbehaving.
* **Decide behavior from explicit inputs, not inferred state.** Drive branching with an explicit parameter, not by sniffing the shape/dtype/contents of the data. Inferred checks silently encode invariants the caller and the next reader won't know are there - make them part of the signature instead.
* **Validate at the boundary; trust within it**. Data crossing an untrusted boundary (user config, deserialized state) is validated once at the edge into a model that guarantees its own invariants — that's the validation layer, and we use `pydantic` for it. Pass trusted internal data as plain `dataclasses` and assume it is already valid. Don't re-validate trusted internal data, and don't pass unvalidated raw input deeper than the edge.
* **Separate configuration from behavior.** Config objects describe; classes behave. Don't merge them into one dataclass that needs `__post_init__` magic.
* **Polymorphism for configurable objects via discriminated unions.** When a configurable object has selectable behavior, model the choices as a Pydantic discriminated union and resolve them in a `build_*()` factory with an exhaustive `match (case _: raise)`.

### Linting & Formatting
We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
Configuration is strict (see `pyproject.toml` for the authoritative list of enabled rules).

The rule set is broad. The conventions below summarize what it means in practice so you can write
conforming code without memorizing rule codes.

#### Formatting
*   Double quotes, 4-space indentation, 120-char line length.
*   Imports are auto-sorted (`I`). Run `make lint` to fix ordering.

#### Typing & annotations
*   **Annotate everything public** (`ANN`): function args and return types. `None` returns may be omitted.
*   `typing.Any` is allowed (`ANN401` is off) but should be a last resort.
*   Prefer modern syntax (`UP`, `FA`): `X | Y` over `Optional`/`Union`, builtin generics (`list[int]`),
    and `from __future__ import annotations` where it helps.

#### Naming
*   `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
*   Exceptions: `F` (for `nn.functional`) and `BLOCK_SIZE`/uppercase kernel args are allowed.
*   Private attributes should be prefixed with `_`: `self._something = ...`

#### Boundaries
*   Every package needs `__init__.py` (`INP`); `/example/` is exempt.

#### Tests
*   Use idiomatic `pytest` (`PT`): `pytest.raises`, fixtures, parametrization.
*   Tests relax several rules: `assert` is allowed, no docstrings/annotations required, private
    access and non-top-level imports are fine.

### Type Checking
We use **[ty](https://github.com/astral-sh/ty)** to ensure type safety.

*   **Coverage:**
    *   **Core Code:** strict type checking is enabled.
    *   **Tests:** Excluded from strict type analysis (`test/**`) - at least for now.
    *   **External Kernels:** External low-level kernels and wrappers (e.g., `d9d/kernel/cce`, `deep_ep`) are explicitly ignored.

### Testing
We have two tiers of tests:
1.  **Local (`-m local`):** Standard logic tests that run in a single process.
2.  **Distributed (`-m distributed`):** Tests that strictly require `torchrun`.

**Requirement:** All PRs must pass `make test`. If you add a feature, you must add corresponding tests.

## Documentation

### Docstrings

We follow the [Google Python style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

*   **Style:** Use Google-style docstrings (`Args:`, `Returns:`, `Raises:`, etc.).
*   **No type annotations in docstrings:** Types are already declared in the signature and checked by `ty`. Do not repeat them in the docstring.
*   **Document `__init__`:** Write a docstring even for `__init__`, but keep it short and to the point, e.g. `"""Constructs the ``Trainer`` object."""`.
*   **Public API coverage:** Always write docstrings for everything considered public API.

### Documentation Site

The site is built with **Zensical**.

*   **Location:** Page sources live in `docs/`; the code they document lives in `d9d/`.
*   **Building:** Run `make mkdocs` to preview changes locally.
*   **Registering pages (`zensical.toml`):** The site navigation is **not** auto-generated from the `docs/` directory - it is defined explicitly in the `nav` table of `zensical.toml`. Whenever you add, remove, rename, or move a page under `docs/`, you must update `nav` accordingly. New top-level subsystems should also be added to the appropriate section (and mirrored in `docs/toc.md`).

## Commit Messages & PRs

We use [Semantic Release](https://python-semantic-release.readthedocs.io/en/latest/) to automate versioning and changelogs. **Your commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.**

### Format
```text
<type>(<scope>): <subject>
```

### Types
| Type       | Description                                             | Version Bump |
|:-----------|:--------------------------------------------------------|:-------------|
| `feat`     | New feature                                             | **Minor**    |
| `fix`      | Bug fix                                                 | **Patch**    |
| `perf`     | Performance improvement                                 | **Patch**    |
| `docs`     | Documentation only                                      | None         |
| `style`    | Formatting                                              | None         |
| `refactor` | Code change that neither fixes a bug nor adds a feature | None         |
| `test`     | Adding missing tests, refactoring tests                 | None         |
| `chore`    | Build process, dependency updates                       | None         |
| `ci`       | CI configuration changes                                | None         |

### Examples
*   `feat(moe): add deepep communication support`
*   `fix(checkpoint): fix async dcp`
*   `docs: update contributing guide`

## Pull Request Checklist

Before submitting a PR, ensure you have:

1.  [ ] Created a DEP (if the change is major).
2.  [ ] Added tests for your change.
3.  [ ] Ran `make lint` to fix formatting, imports and check for typing issues.
4.  [ ] Ran `make test` to ensure no regressions.
5.  [ ] Used a Conventional Commit title for your PR.

---

**Happy Hacking!** 🚀
