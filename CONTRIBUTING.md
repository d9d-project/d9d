# Contributing Guide

First off, thank you for considering contributing to **d9d**!

We aim to build a distributed training framework that is **efficient, hackable, and reliable**. To maintain this balance, we adhere to a strong engineering culture: strict type-checking, rigorous linting, and a structured proposal process for major changes.

 This document outlines the standards and workflows for contributing to the project.

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
| `make lint`   | Runs `ruff` linting checks.                                                                                                   |
| `make mypy`   | Runs static type checking via `mypy`.                                                                                         |
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

### Linting & Formatting
We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. 
Configuration is strict (see `pyproject.toml` for enabled rules).

### Type Checking
*   **Strict Typing:** While `disallow_untyped_defs` is currently relaxed, we strongly encourage full type hints for all function signatures.
*   **Pydantic:** We use Pydantic V2. Ensure your code is compatible with pydantic-mypy plugins.

### Testing
We have two tiers of tests:
1.  **Local (`-m local`):** Standard logic tests that run in a single process.
2.  **Distributed (`-m distributed`):** Tests that strictly require `torchrun`.

**Requirement:** All PRs must pass `make test`. If you add a feature, you must add corresponding tests.

## Documentation

Documentation is built with **MkDocs**.
*   **Docstrings:** Public APIs must have clear docstrings.
*   **Building:** Run `make mkdocs` to preview changes.
*   **Location:** Source files are in `docs/` and documented code in `d9d/`.

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
3.  [ ] Ran `make lint` to fix formatting and imports.
4.  [ ] Ran `make mypy` to ensure type safety.
5.  [ ] Ran `make test` to ensure no regressions.
6.  [ ] Used a Conventional Commit title for your PR.

---

**Happy Hacking!** 🚀
