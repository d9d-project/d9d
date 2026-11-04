---
name: review
description: Review a change in d9d (your own before you hand it over, a branch or a PR) and walk the user through it group by group. Use when the user asks to review code, docs or a PR, asks in which order to review, or before you present a finished change.
---

# Reviewing Changes

A review checks a change against `CONTRIBUTING.md` and against the rest of d9d. Read its Design Principles, Testing, Writing and Documentation sections before you start. For a DEP, use the `dep` skill. This skill adds only what `CONTRIBUTING.md` does not say.

## 1. Scope the review

1.  Collect every changed file against the base of the branch, including untracked files, tests, docs and examples.
2.  For a large change, start independent subagents in the background now, before you form your own view, and review on your side while they run. Split one PR by concern, and several PRs by part first. The concerns are:
    *   **Correctness**: Scenarios that crash, compute a wrong result or keep memory alive. Reproduce them where possible.
    *   **Design**: `CONTRIBUTING.md`, this skill, and how d9d, PyTorch and other frameworks solve the same problem in their code.
    *   **Integration**: What merged into `main` since the base, conflicts in behavior, and tests after a rebase on the minimum supported Python.
    *   **Tests and text**: The rules of "Testing", and every claim of the docs, the DEP and the PR text against the code.

    Give each subagent the diff, its base, its concern and the limits (read-only, no comments on GitHub, its own temporary worktree, no GPU jobs while the GPUs are busy). Check yourself each finding that you report, and say which ones you did not check.
3.  Read each changed file around the change, and every caller of a changed signature.

## 2. Check the design

Ask for every new object, field, parameter, helper and comment why it is needed. Remove it if nothing needs it. Skip findings and fixes for configurations nobody runs.

*   If a fix adds a hook or a special case for one object, check how other d9d objects and other frameworks solve it. Read their code, not their docs, e.g. PyTorch, torchtitan and Megatron-LM.
*   If a PR breaks a signature anyway, ask for the final form in the same break.
*   If an object starts to hold more, check how long each large tensor lives. Memory must not stay alive longer, e.g. input gradients until the weight pass.

## 3. Check the tests, the docs and the PR text

*   A test that mocks the fixed call passes without the fix, so run a test of a fix on `main`.
*   Check every claim in the docs, the DEP and the PR text against the code. Leave out invented reasons and facts that soon change, e.g. versions or current bugs.
*   Check an implemented DEP against `deps/TEMPLATE.md` and the code.

## 4. Verify

1.  Run `make lint`, the local tests of the touched packages and what the change can break, e.g. the quickstart for loop changes.
2.  Before a commit, apply its files to `HEAD` in a temporary worktree and run `ty` and its tests. If it does not build alone, say so and propose a grouping.

For a PR of someone else, also:

1.  Reproduce each fixed bug on `main`, and name a real configuration where it triggers.
2.  Rebase the PR onto the current `main` and run lint and the tests there. A merged API change can break tests without a conflict.
3.  Run the tests on the minimum supported Python.
4.  Ask to split bundled fixes: one bug per PR, each with its own test.

## 5. Walk the user through it

*   Leave the change uncommitted, so the user reads it in the IDE. Put a PR under review into the tree as an uncommitted branch from `main`.
*   Give the review order: general files first, details last, one group per step. List the files as links with paths from the repository root, not as git commands.
*   Report findings as short grouped lists in the language of the user. Use no wide tables and no long reports.
*   Answer a question with an answer, not an edit. Give your own position, and say which facts you checked and which you assume.
*   For a design choice, show code sketches of the alternatives or a dependency graph.
*   Write PR comments for a human: one comment per finding, one to three sentences with the problem and the ask. Leave out praise, what you checked and what the author already knows.
*   Change only what the review asks. Note findings outside the scope instead of fixing them.
*   Commit a group only after the user approves it, with exactly its files. Push and open a PR only when asked.
