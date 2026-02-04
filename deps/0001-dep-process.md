---
DEP: 0001
Title: The D9D Enhancement Proposal (DEP) Process
Author: @mrapplexz
Status: Active
Type: Process
Created: 2026-02-04
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
1.  **Draft:** The author copies `deps/TEMPLATE.md` to `deps/XXXX-my-feature.md`.
2.  **Pull Request:** A PR is opened with the design document. Status is set to `Draft`.
3.  **Review:** The proposal is discussed.
4.  **Acceptance:** Once approved by maintainers, the PR acts as a "green light". The status changes to `Accepted` and the document is merged into the main branch.

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
3. **Integration**: Once approved by maintainers, PR will be merged.

## When is a DEP required?
* Any changes to the existing public API that break backward compatibility.
* New major public API (i.e. new distributed strategies, new base modules).

Note that DEP is **not** required especially for:
* Adding new models that are build on top of existing API.
* Bug fixes.

## DEP Maintenance
DEPs are historical records. Once a DEP status is `Implemented`, the document should not be changed (except for minor typos). 
If the design changes significantly later, a new DEP should be drafted to supersede the old one.
