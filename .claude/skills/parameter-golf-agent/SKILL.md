---
name: parameter-golf-agent
description: Full 5-stage agent workflow for implementing Parameter Golf techniques.
  Use PROACTIVELY when the user asks to implement, add, or try a technique, architecture
  change, or optimization. Orchestrates researcher → planner → implementer → verifier
  → reviewer agents in sequence.
allowed-tools: Read, Write, Bash
---

# Parameter Golf Agent Orchestrator

You are orchestrating a 5-stage pipeline. Execute each stage in order, passing the output of each stage as context to the next. Do NOT skip stages.

## Stage 1 — Research
Read `.claude/agents/researcher.md` and follow its instructions completely.
Produce: technique summary, expected BPB gain, code examples, hyperparameter recommendations.

## Stage 2 — Plan
Read `.claude/agents/planner.md` and follow its instructions completely.
Input: Stage 1 research output.
Produce: exact file locations, line numbers, code diffs, and a step checklist.

## Stage 3 — Implement
Read `.claude/agents/implementer.md` and follow its instructions completely.
Input: Stage 2 plan.
Rule: ONE change at a time → verify syntax → next change. Never batch unverified changes.
Produce: implementation summary and list of changes made.

## Stage 4 — Verify
Read `.claude/agents/verifier.md` and follow its instructions completely.
Input: Stage 3 implementation summary.
Run all applicable levels (1–5). Do NOT proceed to Stage 5 if Level 3 fails.
Produce: verification report with pass/fail for each level.

## Stage 5 — Review
Read `.claude/agents/reviewer.md` and follow its instructions completely.
Input: Stages 1–4 outputs.
Produce: APPROVED ✅ or REJECTED ❌ with actionable feedback.

---

## Stopping Rules
- If Stage 1 finds the technique already implemented and working → report and stop.
- If Stage 3 cannot compile after 3 fix attempts → revert changes and stop.
- If Stage 4 Level 3 confirms technique is inactive → return to Stage 2.
- Only APPROVED results from Stage 5 are considered production-ready.
