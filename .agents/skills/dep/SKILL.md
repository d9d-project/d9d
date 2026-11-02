---
name: dep
description: Write, revise or review a D9D Enhancement Proposal (DEP) in deps/. Use when the user asks for a DEP or a design proposal, asks to review or edit one, or when a change needs a DEP by the rules of CONTRIBUTING.md.
---

# D9D Enhancement Proposals

A DEP fixes a design before the code. Read `deps/0001-dep-process.md` for the process and `deps/TEMPLATE.md` for the rules of each section before you start.

## 1. Decide the scope

1.  Check that the change needs a DEP, see "The DEP Process" in `CONTRIBUTING.md`.
2.  Split off every fix that needs no new abstraction. It goes into its own PR first, and the DEP links it.

## 2. Research

1.  Find every consumer and every defect in the code. Reproduce each defect or point to the line that causes it.
2.  Read how PyTorch and other frameworks solve the problem in their code, not in their docs. Compare designs, not features.
3.  Note the constraints that PyTorch forces on the design.

## 3. Write the draft

1.  Copy `deps/TEMPLATE.md` to `deps/<number>-dep-<name>.md`. Take the next free number unless the user gives one.
2.  Follow the comment at the top of the template and the rules in each section. The Writing rules and the Design Principles of `CONTRIBUTING.md` apply.
3.  Derive each rule from what it must achieve. Do not add an option where one choice is correct.

## 4. Review before you show it

1.  Give the draft to an independent reviewer, e.g. a subagent, without your reasoning. Ask it to:
    *   check every claim about code against the code;
    *   criticize the design and find cases it breaks;
    *   check the Writing rules and the template rules.
2.  Fix the findings that hold up.
3.  Check that no sentence is longer than 25 words and that no term means something else in d9d.

## 5. Iterate with the user

*   Answer questions in the chat. Change the DEP only when the design changes, and add no paragraphs that justify an implementation detail.
*   Keep the draft in sync with the code. Drop what merged PRs have already fixed.
*   Commit the DEP and open its PR only when the user asks. Merging the PR accepts the DEP.
