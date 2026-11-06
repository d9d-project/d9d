---
name: implement
description: Implement a change in d9d (a fix, a feature, a refactor or an accepted DEP) and hand it over for review. Use when the user asks to fix, add, change or implement something in the code.
---

# Implementing Changes

An implementation follows `CONTRIBUTING.md`. Read its Design Principles, Testing, Writing and Documentation sections before you start. Use the `dep` skill if the change needs a DEP, and the `review` skill before you hand the change over. This skill adds only what they do not say.

## 1. Understand the task

1.  Answer a question with an answer. "Why X?", "what do you think?" or "list the bugs" asks for an opinion or a list, not an edit.
2.  Find the root problem. Look for the same flaw elsewhere in d9d, and propose one fix for all of it.
3.  Find every consumer of what you change: callers, tests, docs, examples and configs.

## 2. Agree on the shape

1.  For a change beyond a few lines, show the plan before you write code. Show the public API, the file layout or a code sketch, and wait for the answer.
2.  List the options with how PyTorch and other frameworks solve the problem. Read their code, not their docs. Say which option you prefer and why.
3.  Split prerequisite fixes and refactors into their own PRs. Stack the main change on them.
4.  Implement a DEP whose PR is still open in one PR stacked on the DEP PR.

## 3. Write the change

*   Work on the branch the user names. On `main`, create a branch for the task. Stash unrelated changes in the working tree.
*   Touch only what the task needs. Fix the same flaw elsewhere within the task, and fix wrong docs you notice on the way.
*   List other findings, and let the user choose: fix now, file an issue or leave it.
*   Ask the questions of section 2 of the `review` skill while you write, not only at the end.
*   Update the docs, examples and configs in the same step as the code.
*   Keep the DEP in sync. If the code must differ from the DEP, change the DEP first and tell the user.
*   After a rename or a move, grep for the old name in code, tests, docs and examples.

## 4. Verify

1.  Back every claim of the report, the docs and the PR text with code, a run or a source. Mark a guess as a guess.
2.  Run the `review` skill, fix what holds up and review again.

## 5. Hand over

*   End with a short report that stands alone: what changed, and what is left with the next step.
