---
DEP: <Next free number>
Title: <Short, descriptive title>
Author: <Your Name>
Type: <Feature | Refactor | Process>
---

# Title

<!--
Rules for the whole document. Delete this comment when you copy the template.

*   The Writing rules and the Design Principles of CONTRIBUTING.md apply.
*   State each fact once, in the section where the reader looks for it.
*   A DEP covers only the change that needs a DEP. Fix bugs that need no new abstraction in separate PRs first, and
    link them.
*   While the DEP is a draft, keep it in sync with the code. Drop what other PRs have already fixed.
*   Answer review questions in the review. Change the DEP only when the design changes.
*   Two header fields are added later. The PR that implements the DEP adds `Implemented: #<PR>`. The PR of a DEP that
    replaces this one adds `Superseded-By: DEP-<number>`.
-->

## Abstract

The problem in general terms and the core of the solution, in two short paragraphs. Leave out examples, numbers and details. They belong to the sections below.

## Motivation

Why d9d needs the change. List the concrete defects or limits, one or two sentences each. Check each one in the code or reproduce it. Examples belong here.

## Design Proposal

The specification of the change.

*   **New entities**: Introduce each new public entity explicitly, e.g. "This DEP introduces `X`". Show its signature with a docstring that states the contract. Leave out the bodies.
*   **Rules**: Derive each rule from what it must achieve, and say why it holds.
*   **Tables**: Use a table only for flat reference data. If a row needs an explanation, write a list.
*   **Implementation details**: Leave them to code comments, e.g. the choice of a hash function.
*   **Diagrams**: Add a Mermaid diagram when it shows a flow better than prose.

## Usage

Short code that a user writes, with real names from the API.

```python
import d9d.your.feature

# Example user code
```

## Backward Compatibility

What breaks for users, one item per change, and how they migrate.

## Alternatives Considered

One subsection per alternative: what it is, which frameworks use it and why this design is better. Name the frameworks, and leave out references to files and lines.

## Deferred Work

What the DEP leaves out on purpose, with the reason or the issue that tracks it.
