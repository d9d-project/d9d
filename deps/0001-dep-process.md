---
DEP: 0001
Title: The D9D Enhancement Proposal (DEP) Process
Author: @mrapplexz
Type: Process
---

# DEP-0001: The DEP Process

## Abstract

This document outlines the process for creating, reviewing, and adopting D9D Enhancement Proposals (DEPs). DEPs are the primary mechanism for proposing major new features, architectural changes, or protocol adjustments to the **d9d** distributed training framework.

## Motivation

We see `d9d` as a universal framework, capable of adapting to diverse model architectures and heterogeneous distributed regimes. We also view `d9d` as a long-living foundational tool. 

An architectural mistake is expensive. A poorly designed feature is not merely a cosmetic issue, it results in accumulated technical debt, broken compatibility, forcing users to rewrite their projects when they upgrade, and general unusability. 

To prevent this, we cannot rely on ad-hoc implementation. Every significant change to availability, API, or internal logic requires structured planning. 
The DEP process is designed to force the **think before you do**, i.e. force the comparison of alternatives before code is written, ensuring that the chosen design is not just "working", but is the optimal solution for the long-term health of the project.

## The DEP Lifecycle

The process for a feature to land in `d9d` is strictly sequential.

### 1. Proposal Phase
1.  **Draft:** The author copies `deps/TEMPLATE.md` to `deps/<number>-dep-<name>.md`.
2.  **Pull Request:** A PR is opened with the design document.
3.  **Review:** The proposal is discussed.
4.  **Acceptance:** Once approved by maintainers, the PR acts as a "green light". Merging the document into the main branch accepts it.

### 2. Development Phase
Now you start the main development process. You should open a Pull Request for the feature and mark it a **draft**. A feature is not considered complete until the following is fulfilled:

* **Implementation**
  * Here you write the core logic.
* **Tests**
  * Each feature should be covered with automatic tests (both local and distributed ones).
  * Write unit tests.
  * Write E2E and integration tests if needed.
* **Docstrings**
  * Once logic is frozen is tested, you start documenting the code.
  * The public API should be documented
  * Non-public API may be documented if needed.
* **Documentation**
  * Write the documentation (in the `/docs` directory)
  * Add usage examples, guides, tutorials, everything you find necessary.

### 3. Code Review & Finalization Phase
Once you complete development of your feature, these steps will be passed:

1. **Pull Request**: Author should open (or mark it as **ready**) Pull Request
2. **Code Review**: Discuss the code changes with project maintainers.
3. **Integration**: Once approved by maintainers, PR will be merged. The PR adds `Implemented: #<PR>` to the header of the DEP.

## When is a DEP required?
* New major public API (i.e. new public abstractions, new distributed strategies, new base modules).
* Changes to the existing public API that break backward compatibility, once d9d reaches 1.0.

Note that DEP is **not** required especially for:
* Adding new models that are build on top of existing API.
* Bug fixes.
* Breaking changes that introduce no new abstraction, while d9d is on 0.x.

## DEP Maintenance
DEPs are historical records. Once the implementation of a DEP is merged, the document should not be changed (except for minor typos). 
If the design changes significantly later, a new DEP should be drafted to supersede the old one. The PR of the new DEP adds `Superseded-By: DEP-<number>` to the header of the old one.
