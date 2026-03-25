---
name: parameter-golf-agent
description: Agent-driven workflow for implementing Parameter Golf techniques. Use this when the user wants to implement a new technique - it will research, plan, implement, verify, and review automatically. Always trigger this skill when user says they want to "implement", "add", or "try" a technique in their Parameter Golf submission.
---

# Parameter Golf Agent Workflow

A comprehensive agent-driven system for implementing Parameter Golf techniques. When a user wants to add a technique to their training script, this skill orchestrates a multi-agent workflow to ensure correct, verified implementation.

## When to Use This Skill

**Always trigger** when user requests:
- "Implement XYZ technique"
- "Add ABC to my model"
- "I want to try technique X"
- "How do I use feature Y"
- "Apply optimization Z"

**Even if they don't say "agent" or "workflow"** - this is the default way to handle technique implementation requests.

## The Agent Workflow

This skill follows a **5-stage agent workflow**:

```
User Request
     ↓
[1] RESEARCH → Understand the technique
     ↓
[2] PLAN → Design implementation strategy
     ↓
[3] IMPLEMENT → Write the code  (you do this, or guide user)
     ↓
[4] VERIFY → Test correctness
     ↓
[5] REVIEW → Final comprehensive check
     ↓
DONE ✅
```

Each stage is handled by a specialized agent with specific expertise.

## Stage 1: Research 🔍

**Agent**: Technique Researcher
**Instructions**: `.claude/commands/agents/researcher.md`
**Goal**: Understand the technique thoroughly before any coding

### What the Researcher Does

1. **Clarifies the request** - What exactly does the user want?
2. **Searches the repository** - Finds existing implementations
3. **Reviews submissions** - Studies how others have used this technique
4. **Checks research** - Reads papers, PRs, documentation
5. **Assesses feasibility** - Is this technique appropriate?
6. **Documents findings** - Provides comprehensive summary

### Research Output

The Researcher provides:
- **What the technique does** (theory and mechanism)
- **Expected impact** (BPB improvement, from submissions)
- **Code examples** (from existing implementations)
- **Hyperparameters** (typical values and tuning guidance)
- **Prerequisites** (what's needed to use it)
- **Feasibility assessment** (should we proceed?)

### When Research Fails

If the Researcher can't find information:
- **Ask the user** for paper/PR/example
- **Suggest alternatives** if available
- **Admit uncertainty** rather than guessing

**Never proceed without understanding the technique.**

### Example Research Output

```markdown
## Technique Research: LeakyReLU²

### Summary
LeakyReLU² applies leaky_relu(x, 0.5).square() activation. Preserves negative
gradients unlike ReLU, reduces dead neurons, improves convergence.

### Expected Impact
- BPB: ~0.003 improvement
- Cost: Negligible
- Complexity: Easy (one line change)

### Example Implementation
File: records/.../2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py:234
[code snippet]

### Recommendation
✅ Proceed - proven technique, simple, no downsides
```

---

## Stage 2: Planning 📋

**Agent**: Implementation Planner
**Instructions**: `.claude/commands/agents/planner.md`
**Goal**: Design exactly how to implement the technique

### What the Planner Does

1. **Reviews research** - Understands what needs to be done
2. **Analyzes current code** - Reads train_gpt.py to find insertion points
3. **Designs changes** - Plans specific modifications with line numbers
4. **Identifies conflicts** - Checks compatibility with existing techniques
5. **Plans verification** - Defines how to test correctness
6. **Creates checklist** - Step-by-step implementation instructions

### Planning Output

The Planner provides:
- **Current state analysis** - What's in the code now
- **Proposed changes** - Exactly what to modify (with line numbers)
- **Hyperparameters** - What environment variables to add
- **Compatibility check** - What works/conflicts with this technique
- **Verification plan** - How to confirm it works
- **Implementation steps** - Ordered checklist for implementation

### Planning Principles

- **Minimal changes** - Only modify what's necessary
- **Preserve existing behavior** - Don't break other features
- **Make it configurable** - Add environment variables for tuning
- **Plan for debugging** - Include verification at each step

### Example Planning Output

```markdown
## Implementation Plan: LeakyReLU²

### Proposed Changes

#### Change 1: Modify MLP activation
Location: train_gpt.py:147-149
Action: Replace SwiGLU with LeakyReLU²

Current:
```python
gate = F.silu(self.gate(x))
return self.down(gate * self.up(x))
```

New:
```python
x = self.up(x)
x = F.leaky_relu(x, negative_slope=0.5).square()
return self.down(x)
```

### Implementation Steps
1. Modify MLP class (lines 145-150)
2. Verify: python train_gpt.py --help
3. Test: 5 iteration smoke test
4. Verify technique is active
```

---

## Stage 3: Implementation 💻

**Agent**: You (Claude Code) or the User
**Goal**: Make the planned changes to the code

### What Happens in Implementation

Based on the Planner's step-by-step instructions:

1. **Make code changes** - Edit train_gpt.py following the plan
2. **Add hyperparameters** - Include environment variables
3. **Test incrementally** - Verify each change before proceeding
4. **Handle errors** - Fix issues as they arise
5. **Document changes** - Comment complex sections

### Implementation Guidelines

**If you're implementing (Claude Code)**:
- Follow the plan precisely
- Use the Edit tool for surgical changes
- Test after each major change
- Keep diffs small and focused
- Don't refactor unrelated code

**If user is implementing**:
- Provide the plan clearly
- Offer code snippets they can copy
- Guide them through each step
- Help debug any issues

### During Implementation

- ✅ **Do**: Follow the plan, test frequently, ask questions if unclear
- ❌ **Don't**: Skip steps, make unplanned changes, ignore errors

### Implementation Checklist

```markdown
- [ ] Make code changes from plan
- [ ] Add/modify hyperparameters
- [ ] Update imports if needed
- [ ] Remove debug statements
- [ ] Test basic syntax (python train_gpt.py --help)
- [ ] Ready for verification
```

---

## Stage 4: Verification ✅

**Agent**: Implementation Verifier
**Instructions**: `.claude/commands/agents/verifier.md`
**Goal**: Confirm the implementation works correctly

### What the Verifier Does

1. **Level 1: Syntax** - Check Python syntax is valid
2. **Level 2: Smoke Test** - Run 5 iterations to catch crashes
3. **Level 3: Feature Verification** - **CRITICAL**: Confirm technique is actually enabled
4. **Level 4: Performance** - Check memory/speed are acceptable
5. **Report findings** - Provide clear pass/fail with evidence

### Verification Levels

#### Level 1: Syntax Validation
```bash
python -m py_compile train_gpt.py
python train_gpt.py --help
```
**Pass**: No errors

#### Level 2: Smoke Test
```bash
RUN_ID=smoke_test ITERATIONS=5 python train_gpt.py
```
**Pass**: Completes 5 iterations without crash

#### Level 3: Feature Verification ⚠️ CRITICAL
**Most important check!** Verify technique is **actually being used**.

Methods:
- Add logging to confirm execution
- Check model architecture
- Use torch hooks
- Compare with/without technique

**Pass**: Clear evidence technique is active

#### Level 4: Performance Check
```bash
RUN_ID=perf_test ITERATIONS=100 python train_gpt.py
```
**Pass**: Memory <20GB/GPU, speed reasonable, loss converging

### Verification Output

The Verifier provides:
- **Status** for each level (✅ PASS / ❌ FAIL)
- **Evidence** showing technique is active
- **Performance metrics** (memory, speed, loss)
- **Issues found** with debugging suggestions
- **Recommendations** for next steps

### Example Verification Output

```markdown
## Verification Report: LeakyReLU²

### Summary
✅ PASS - All levels passed

### Level 3: Feature Verification
✅ PASS - Technique is ACTIVE

Evidence: Added logging to MLP.forward()
Output shows:
```
[VERIFY] Using leaky_relu with slope=0.5
[VERIFY] Applying square after activation
```

✅ Confirmed active on every forward pass.

### Conclusion
Implementation verified. Ready for full training.
```

---

## Stage 5: Review 🔍

**Agent**: Implementation Reviewer
**Instructions**: `.claude/commands/agents/reviewer.md`
**Goal**: Comprehensive final check before production use

### What the Reviewer Does

1. **Code review** - Check implementation matches plan
2. **Activation review** - Verify technique is enabled (critical!)
3. **Integration review** - Check compatibility with other techniques
4. **Configuration review** - Validate hyperparameters
5. **Edge case review** - Test boundary conditions
6. **Performance review** - Assess memory/speed trade-offs
7. **Documentation review** - Ensure changes are documented
8. **Final sign-off** - Approve, approve with conditions, or reject

### Review Checklist

The Reviewer verifies:
- ✅ Code quality (clean, correct, matches plan)
- ✅ Technique is enabled (not accidentally disabled)
- ✅ Integration (works with existing techniques)
- ✅ Configuration (correct defaults, tunable)
- ✅ Edge cases (handles disabled, extreme values, etc.)
- ✅ Performance (memory and speed acceptable)
- ✅ Documentation (comments, parameter docs)

### Review Outcomes

**✅ APPROVED**: Ready for production use
- All checks pass
- Minor issues noted but non-blocking
- Ready for full 8xH100 training

**⚠️ APPROVED WITH CONDITIONS**: Can proceed after fixes
- Most checks pass
- Some issues need addressing
- List specific conditions to meet

**❌ NEEDS FIXES**: Not ready for production
- Critical issues found
- Must fix before proceeding
- Detailed list of required fixes

### Example Review Output

```markdown
## Implementation Review: LeakyReLU²

### Executive Summary
✅ APPROVED - Ready for production

Key Findings:
- Implementation correct and complete
- Technique confirmed active
- Performance within expected range
- One minor suggestion (remove debug print)

### Final Verdict
✅ APPROVED

Recommendation: Remove debug print (line 234), then ready for full training

Expected Result: ~0.003 BPB improvement
```

---

## Workflow Execution

### How to Execute This Workflow

When user requests a technique implementation:

1. **Start with Research**
   ```
   Reading agents/researcher.md and executing research workflow...
   [Researcher searches repository, reads submissions, documents findings]
   ```

2. **Proceed to Planning**
   ```
   Reading agents/planner.md and creating implementation plan...
   [Planner analyzes code, designs changes, creates checklist]
   ```

3. **Implement Changes**
   ```
   Following implementation plan step-by-step...
   [You or user makes code changes]
   ```

4. **Verify Implementation**
   ```
   Reading agents/verifier.md and running verification...
   [Verifier runs tests, confirms technique is active]
   ```

5. **Final Review**
   ```
   Reading agents/reviewer.md and conducting final review...
   [Reviewer checks everything, provides sign-off]
   ```

### Workflow Variations

**Fast path** (for very simple changes):
- Research → Plan → Implement → Quick verify
- Skip full verification if change is trivial (e.g., changing a number)
- Still do final review

**Unknown technique** (user provides new technique):
- Research may require user input
- Ask for papers, examples, or references
- Don't proceed without understanding

**Complex technique** (major changes):
- Research may take longer
- Plan may be more detailed with sub-tasks
- Verification may need custom tests
- Review may be more rigorous

### Handling Issues

**If research fails**:
- Ask user for more information
- Suggest similar known techniques
- Don't proceed without understanding

**If planning seems risky**:
- Warn user about complexity
- Suggest simpler alternatives
- Recommend starting with smaller scope

**If verification fails**:
- Return to implementation
- Fix issues found
- Re-verify after fixes

**If review finds problems**:
- Address critical issues immediately
- Note minor issues for later
- Re-review after fixes

---

## Best Practices

### For the Orchestrator (You)

✅ **Follow the workflow** - Don't skip stages
✅ **Read agent instructions** - Each agent file has detailed guidance
✅ **Be systematic** - Complete each stage before moving to next
✅ **Document everything** - Keep notes of findings at each stage
✅ **Communicate clearly** - Tell user what stage you're in

❌ **Don't skip research** - Never implement without understanding
❌ **Don't skip verification** - Code may run but technique be inactive
❌ **Don't skip review** - Final check catches subtle issues
❌ **Don't make assumptions** - Verify everything with evidence

### For the User

When requesting a technique:
- **Be specific** about what you want
- **Provide context** (paper, PR, example) if you have it
- **Answer questions** during research phase
- **Review outputs** at each stage
- **Test the result** after final approval

### Communication Pattern

Keep user informed at each stage:

```
"I'll implement LeakyReLU² for you using the agent workflow:

Stage 1/5: 🔍 Researching technique...
[Research output]

Stage 2/5: 📋 Creating implementation plan...
[Planning output]

Stage 3/5: 💻 Implementing changes...
[Implementation notes]

Stage 4/5: ✅ Verifying implementation...
[Verification report]

Stage 5/5: 🔍 Final review...
[Review report]

✅ Complete! Your technique is implemented and verified.
Next steps: Run full training on 8xH100."
```

---

## Agent Reference

### Agent Files Location
All agent instructions are in: `.claude/commands/agents/`

- `researcher.md` - Technique research agent
- `planner.md` - Implementation planning agent
- `verifier.md` - Implementation verification agent
- `reviewer.md` - Final review agent

### Reading Agent Instructions

When executing a stage, read the corresponding agent file:

```
Reading agent instructions from .claude/commands/agents/researcher.md...
[Execute research workflow as described in file]
```

The agent files contain:
- Detailed role description
- Step-by-step process
- Output format templates
- Best practices and examples
- Common issues and solutions

### Agent Outputs

Each agent produces structured output that feeds the next stage:

```
Research → Planning → Implementation → Verification → Review
  [findings]  [plan]    [code changes]   [test results]  [approval]
```

---

## Integration with Existing Skills

This agent workflow integrates with other parameter-golf skills:

- **`/parameter-golf`** - Overview and reference
- **`/parameter-golf-techniques`** - Technique catalog (used by Researcher)
- **`/parameter-golf-implement`** - Implementation patterns (used by Planner)
- **`/parameter-golf-verify`** - Verification guide (used by Verifier)
- **`/parameter-golf-submit`** - Submission process (after Review approval)

The agent workflow is the **recommended way** to implement techniques. The other skills provide reference material.

---

## Example: Complete Workflow

**User**: "I want to implement Exclusive Self-Attention"

**Claude Code (You)**:

```
I'll implement Exclusive Self-Attention (XSA) using the agent workflow.

═══════════════════════════════════════
Stage 1/5: 🔍 Research
═══════════════════════════════════════

Reading agents/researcher.md...

Let me research XSA in the repository...

[Searches records, finds implementations, reads about XSA]

## Research Summary: Exclusive Self-Attention (XSA)

**What it does**: XSA subtracts the component of attention output that's aligned
with the token's own value vector. This reduces self-reinforcement and improves
token diversity.

**Expected Impact**: ~0.005 BPB improvement (from 2026-03-21 submission)

**Typical Usage**: Apply to last 4 layers only (too expensive for all layers)

**Example**: records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/

[Detailed code examples and hyperparameters]

**Recommendation**: ✅ Proceed - proven technique, moderate complexity

═══════════════════════════════════════
Stage 2/5: 📋 Planning
═══════════════════════════════════════

Reading agents/planner.md...

Based on research, creating implementation plan...

## Implementation Plan: XSA

[Analyzes current code]
Current: Standard attention in all 11 layers
Plan: Modify last 4 layers to use XSA

**Change 1**: Add XSA_LAST_N hyperparameter
**Change 2**: Modify CausalSelfAttention class to support XSA mode
**Change 3**: Apply XSA to layers[7:11]

[Detailed code changes with line numbers]

**Verification Plan**:
1. Syntax check
2. Smoke test (5 iterations)
3. Verify XSA is active (check attention outputs)
4. Performance check (memory, speed)

═══════════════════════════════════════
Stage 3/5: 💻 Implementation
═══════════════════════════════════════

Following implementation plan...

[Makes code changes step by step]

Change 1: Adding XSA_LAST_N hyperparameter... ✅
Change 2: Modifying CausalSelfAttention class... ✅
Change 3: Applying XSA to last 4 layers... ✅

Basic syntax check: python train_gpt.py --help ✅

═══════════════════════════════════════
Stage 4/5: ✅ Verification
═══════════════════════════════════════

Reading agents/verifier.md...

Running verification tests...

**Level 1: Syntax** ✅ PASS
**Level 2: Smoke Test** ✅ PASS (5 iterations completed)
**Level 3: Feature Verification** ✅ PASS

Evidence: Added hook to attention layers
Output shows:
```
Layer 7: Applying XSA
Layer 8: Applying XSA
Layer 9: Applying XSA
Layer 10: Applying XSA
```

✅ XSA confirmed active on last 4 layers

**Level 4: Performance** ✅ PASS
Memory: 16.2GB/GPU (+1.0GB, acceptable)
Speed: 4.1s/iter (+0.3s, acceptable)

═══════════════════════════════════════
Stage 5/5: 🔍 Final Review
═══════════════════════════════════════

Reading agents/reviewer.md...

Conducting comprehensive review...

## Review Summary: XSA

✅ Code quality: Clean, matches plan
✅ Technique active: Verified on last 4 layers
✅ Integration: Compatible with existing techniques
✅ Configuration: XSA_LAST_N=4 (good default)
✅ Edge cases: Tested disabled/extreme values
✅ Performance: Within acceptable range
✅ Documentation: Commented and referenced

**Final Verdict**: ✅ APPROVED

═══════════════════════════════════════
✅ Implementation Complete!
═══════════════════════════════════════

**Summary**: Exclusive Self-Attention implemented on last 4 layers

**What was changed**:
- Added XSA_LAST_N hyperparameter (default=4)
- Modified CausalSelfAttention to support XSA mode
- Applied XSA to layers 7-10 (last 4 of 11)

**Verification**:
- ✅ All tests passed
- ✅ Technique confirmed active
- ✅ Performance acceptable (+1GB, +0.3s/iter)

**Next Steps**:
1. Remove debug hooks (lines 234, 456)
2. Run full training: torchrun --nproc_per_node=8 train_gpt.py
3. Monitor for ~0.005 BPB improvement
4. Document results for submission

**Expected Result**: ~0.005 BPB improvement based on 2026-03-21 submission

Ready to proceed with full training? Let me know if you want to adjust any
hyperparameters first.
```

---

## Summary

This agent workflow ensures:
- ✅ **Thorough understanding** before implementation
- ✅ **Detailed planning** with clear steps
- ✅ **Correct implementation** following best practices
- ✅ **Verified functionality** with evidence
- ✅ **Comprehensive review** before production

**Always use this workflow for technique implementation.**

It prevents common mistakes:
- ❌ Implementing without understanding
- ❌ Making unplanned changes
- ❌ Code that runs but technique is inactive
- ❌ Missing edge cases or conflicts
- ❌ Proceeding with untested code

**The workflow takes time but saves time** by catching issues early and ensuring correctness.

---

## Related Skills

- `/parameter-golf` - Main overview
- `/parameter-golf-techniques` - Technique catalog
- `/parameter-golf-implement` - Implementation patterns
- `/parameter-golf-verify` - Verification guide
- `/parameter-golf-submit` - Submission process

**This agent workflow is the recommended way to implement techniques.**
