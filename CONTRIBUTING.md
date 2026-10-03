# Contributing Guide

First off, thank you for considering contributing to **d9d**!

We aim to build a distributed training framework that is **efficient, hackable, and reliable**. To maintain this balance, we adhere to a strong engineering culture: strict type-checking, rigorous linting, and a structured proposal process for major changes.

This document outlines the standards and workflows for contributing to the project.

Before starting work on a major feature, we highly recommend jumping into our [Discord server](https://discord.gg/sNRjDbxVrg) to discuss your approach with the core maintainers!

## Development Setup

**d9d** uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging. You will need Python 3.11+.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/d9d-project/d9d.git
    cd d9d
    ```

2.  **Install dependencies:**
    ```bash
    # Create .venv and install the project with all dependency groups (dev, test, docs, examples), without extras
    uv sync

    # Install pre-commit hooks
    uv run pre-commit install
    ```

3.  **(Optional) Install extras:**
    ```bash
    # All extras at once, or pick specific ones with `--extra <name>`
    uv sync --all-extras
    ```
    Some extras (`moe`, `backend-sdpa-flash-attention-2`) contain CUDA extensions without usable prebuilt wheels,
    so uv builds them from source against the locked `torch` (pinned revisions live in `[tool.uv.sources]` of
    `pyproject.toml`). This requires the CUDA 13 toolkit installed at `/usr/local/cuda`, and the first build takes a while.

    `uv sync` removes packages that were not requested, so pass the same `--extra`/`--all-extras` flags on every
    subsequent sync. `uv run` (used by the `Makefile`) does not remove them.

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
*   `__init__.py` files expose the package's public surface via an explicit `__all__` re-export list.

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

## Writing

These rules apply to everything we write: code comments, docstrings, documentation pages, commit messages and
PR descriptions.

### General Rules

*   **Write as little as the reader needs.** When in doubt, cut.
*   **Put each explanation where the reader looks for it.** Explain a line of code next to that line. Explain how to
    use a feature in the documentation. Explain why a change was made in the PR description.
*   **Use simple English.**
    *   Keep sentences short: at most 20 words in procedures and 25 words in descriptions.
    *   Write one idea per sentence. In procedures, write one instruction per numbered step, in the imperative.
    *   Use the active voice and simple tenses.
    *   Use one term for one concept, and use it everywhere.
    *   Do not stack more than three nouns.
    *   Keep a paragraph about one topic, with at most six sentences. Put the main point first.
*   **Write "can" for what is allowed and "must" for what is required.**
*   **Mark stopgaps.** If a design is a stopgap forced by a current limit, say so in one sentence and name the limit.
    In a DEP, list the deferred work explicitly.
*   **Spell common terms one way.** Write "microbatch", "bf16", "fp32", "I/O", "PyTree" and "Hugging Face". Write MiB
    and GiB for sizes in powers of two. Use American spelling.

### Names

*   **Name a new entity after what it is.** Follow the PyTorch or Hugging Face name for the same idea. Do not reuse a
    word that already means something else in d9d.
*   **Make a name exactly as broad as what it covers.** This also applies to DEP and PR titles.

### Comments

*   **Explain *why*, next to the code.** Put a short comment right above the line it explains. Comment a non-obvious
    condition, a workaround or a choice that looks wrong. Do not restate what the code does. Do not move the
    explanation to a class docstring or a module constant just to have a place for it.
*   **Justify code that looks removable.** Some code exists because of an external limitation, such as a kernel
    requirement or a PyTorch quirk. Say so briefly and name the source, so nobody deletes the code as redundant.
*   **Give every suppression a reason.** Name the code and end with ` - <reason>`, e.g.
    `# noqa: BLE001 - re-raised in the consuming thread`. Lazy imports of optional dependencies (`PLC0415`) need no
    reason.
*   **Write a TODO as `# TODO(owner): ...`,** once per item, with a link to its issue.
*   **Write comments as sentences** ([PEP 8](https://peps.python.org/pep-0008/#comments)). Start with a capital
    letter, unless the comment starts with a lowercase identifier, and end with a period.

### Docstrings

We follow the [Google Python style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

*   **Style:** Use Google-style docstrings (`Args:`, `Returns:`, `Raises:`, etc.).
*   **Describe the contract, not the implementation.** Say what the object does, its arguments, return values and
    errors. Implementation choices go into comments. Leave out context the caller does not need and claims the code
    does not guarantee.
*   **No type annotations in docstrings:** Types are already declared in the signature and checked by `ty`. Do not repeat them in the docstring.
*   **Document `__init__`:** Write a docstring even for `__init__`, but keep it short and to the point, e.g. `"""Constructs the ``Trainer`` object."""`.
*   **Public API coverage:** Always write docstrings for everything considered public API.
*   **Write the summary line in the third person.** It fits on one line and ends with a period, e.g. "Computes the
    loss.". An `__init__` summary reads "Constructs the ``X`` object.". A `Raises:` entry starts with "If".
*   **Put identifiers and literals in double backticks,** e.g. ``` ``GroupedLinear`` ``` and ``` ``None`` ```. Do not
    use quotes or Sphinx roles such as `:class:`.
*   **Write tensor shapes one way.** End the description with ```Shape: ``(batch, seq_len, hidden_size)``.```. Use
    parentheses and snake_case dimension names. Use the same name for the same dimension everywhere.
*   **Document fields under `Attributes:`** in data classes, configs and enums. Other classes document their
    arguments in the `__init__` docstring.
*   **Let overrides inherit the contract.** An override may omit its docstring. Document it only if its behavior
    differs from the base contract.
*   **Write a property docstring as a noun phrase,** e.g. "The current step.", without a `Returns:` section.

### Error Messages

*   **Write a full sentence.** Name the offending value as `name ({value})`. If the caller can fix the problem, say
    how. Never raise an exception without a message.
*   **Write calls with parentheses,** e.g. "Call configure_buffers() first."

### Documentation Pages

*   **Say what a feature does and how to use it.** Leave internals (host syncs, alignment, fallback paths) to code
    comments.
*   **Follow the page structure.** Start with `## About`: what the feature is, in one paragraph. Add topic sections.
    Add `## Usage` with a short example for anything users call directly. End with `## API Reference` and the
    `:::` blocks.
*   **Keep each section about one topic.**
*   **Write one paragraph per line.** Do not wrap lines by hand. Use soft wrapping in your editor.
*   **Indent list items by four characters.** Write `*   ` for bullets and `1.  ` for numbered items, so nested
    blocks line up at four spaces.
*   **Use bold only for labels and for a term where you define it.** Do not use bold for emphasis.
*   **Write a labeled list item as `**Label**: Sentence.`** Put the colon outside the bold text. Start the text after
    it with a capital letter and end it with a period.
*   **Write short, real examples.** Use real names from the API. Show configs and environment variables in the form
    users type them.
*   **State facts, not adjectives.** Do not write "efficient", "highly optimized", "powerful" or "seamless". Say what
    makes it fast (a fused kernel, no host sync), or show a benchmark with the hardware and dtype in the heading.
*   **State the limits users must know.** Name the required hardware, dtypes and extras (`d9d[...]`), and what the
    feature does not support.
*   **Do not repeat the API reference.** Arguments and return values come from the docstrings. Use prose for concepts
    and choices.
*   **Link the sources.** Link the paper for a method and the repository for an external library.
*   **Update the documentation in the same PR.** If a change affects what users see or do, update the pages with it.

## Documentation

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

### PR Descriptions

The diff already says *what* changed - the description says *why*, and it should be as short as the
change allows. A one-file fix gets two or three sentences; a new subsystem gets sections. When in
doubt, cut.

Write down, in this order, only the parts that apply:

*   **The problem.** What was broken, missing or slow, and under which configuration it showed up
    (topology, parallelism degrees, hardware). Skip it only when the title is already the whole story.
*   **The approach.** The decision a reviewer cannot read off the diff: why *this* solution, and what
    you rejected. If a DEP covers it, link the DEP instead of re-arguing it here.

Do not restate the diff file by file, do not paste `make lint` output, and do not describe code you
did not write. Do not walk the reviewer through the behaviour you just wrote - on a small change the
*why* is the entire description. Behaviour changes that users must react to (renamed config keys,
new required arguments, dropped defaults) belong in the description even when everything else is
obvious.

### Attribution

Commit messages and PR descriptions carry no tooling attribution. Do not add `Co-Authored-By`
trailers for AI assistants, and do not append "Generated with ..." footers - this applies to
agents working in this repository as well. The human who opens the PR is its author and owns the
change; a tool credit only blurs who is accountable for the code.

## Pull Request Checklist

Before submitting a PR, ensure you have:

1.  [ ] Created a DEP (if the change is major).
2.  [ ] Added tests for your change.
3.  [ ] Ran `make lint` to fix formatting, imports and check for typing issues.
4.  [ ] Ran `make test` to ensure no regressions.
5.  [ ] Used a Conventional Commit title for your PR.
6.  [ ] Written a description that explains the problem and the approach (see above).

---

**Happy Hacking!** 🚀
