# Parameter Golf Agent System

This directory contains specialized agent instructions for implementing Parameter Golf techniques using a systematic workflow.

## Overview

The agent system provides a structured, 5-stage workflow for implementing new techniques in Parameter Golf submissions:

```
Research → Plan → Implement → Verify → Review
```

Each stage is handled by a specialized agent with specific expertise and responsibilities.

## Agent Files

### 1. `researcher.md` - Technique Researcher 🔍
**Purpose**: Understand the technique thoroughly before any coding begins

**Responsibilities**:
- Clarify what technique the user wants
- Search repository for existing implementations
- **Research external sources** (GitHub PRs, papers, docs)
- Review submissions and PRs
- Document theoretical background
- Assess feasibility
- Provide code examples and hyperparameters

**Key Features**:
- **GitHub MCP Integration**: Can fetch and analyze PRs from openai/parameter-golf
- **Research Papers**: Can extract information from provided papers (arXiv links, PDFs)
- **Documentation**: Can read external docs, blog posts, guides
- **Cross-referencing**: Combines local and external sources for best approach

**Output**: Comprehensive research summary with recommendations

### 2. `planner.md` - Implementation Planner 📋
**Purpose**: Design exactly how to implement the technique

**Responsibilities**:
- Analyze current code state
- Design specific changes with line numbers
- Identify potential conflicts
- Plan verification steps
- Create step-by-step implementation checklist

**Output**: Detailed implementation plan with code snippets

### 3. `implementer.md` - Code Implementer 💻
**Purpose**: Make the actual code changes incrementally and safely

**Responsibilities**:
- Review and understand the implementation plan
- Implement changes one at a time
- Test after each change (syntax, imports, basic functionality)
- Debug and fix errors immediately
- Document changes as they're made
- Prepare code for verification

**Key Features**:
- **Incremental implementation**: Change → Test → Repeat
- **Immediate error handling**: Fix issues before proceeding
- **Style matching**: Follow existing code conventions
- **Clean up**: Remove debug code before finishing

**Output**: Implementation summary with test results

### 4. `verifier.md` - Implementation Verifier ✅
**Purpose**: Confirm the implementation works correctly

**Responsibilities**:
- Level 1: Syntax validation (improved `py_compile` usage)
- Level 2: Smoke test (5 iterations)
- Level 3: Feature verification (CRITICAL - confirm technique is active)
- Level 4: Performance check (memory, speed)
- Report findings with evidence

**Key Features**:
- **Enhanced syntax check**: Properly uses `python -m py_compile` with error checking
- **Multiple verification strategies**: Logging, hooks, architecture inspection
- **Evidence-based**: Provides proof that technique is actually enabled

**Output**: Verification report with pass/fail for each level

### 5. `reviewer.md` - Implementation Reviewer 🔍
**Purpose**: Comprehensive final review before production

**Responsibilities**:
- Code quality review
- Technique activation review (double-check it's enabled)
- Integration compatibility check
- Configuration validation
- Edge case testing
- Performance assessment
- Documentation review
- Final sign-off

**Output**: Detailed review report with approval/rejection

## Usage

### From User Perspective

When a user requests implementing a technique:

```
User: "I want to implement LeakyReLU squared"
```

The agent workflow automatically:
1. Researches LeakyReLU² in the repository
2. Plans the implementation with specific code changes
3. Implements the changes (or guides user)
4. Verifies it works and is actually enabled
5. Reviews everything comprehensively

### From Claude Code Perspective

Use the `/parameter-golf-agent` skill which orchestrates these agents:

```markdown
1. Read agents/researcher.md and execute research
2. Read agents/planner.md and create implementation plan
3. Read agents/implementer.md and implement code changes
4. Read agents/verifier.md and run verification tests
5. Read agents/reviewer.md and conduct final review
```

## Key Features

### 🎯 Systematic Approach
Each agent has a clear role and produces structured output that feeds the next stage.

### 🔬 External Source Integration (NEW)
The researcher can now work with external sources:
- **GitHub PRs**: Fetch and analyze PRs from openai/parameter-golf via GitHub MCP
- **Research Papers**: Extract techniques from papers (arXiv, PDFs)
- **Documentation**: Read external docs, blog posts, implementation guides
- **Combined Research**: Cross-reference external sources with local implementations

### 💻 Dedicated Implementation Agent (NEW)
The new implementer agent provides:
- **Incremental changes**: One modification at a time with immediate testing
- **Error handling**: Fixes issues before proceeding
- **Style matching**: Follows existing code conventions
- **Clean implementation**: Removes debug code, matches patterns

### 🔍 Enhanced Verification (IMPROVED)
The verifier now includes:
- **Better syntax checking**: Proper `py_compile` usage with error codes
- **Multiple verification strategies**: Logging, hooks, architecture inspection
- **Evidence-based confirmation**: Proves technique is actually enabled

### 🔍 Critical Verification
The verifier specifically checks that techniques are **actually enabled**, not just present in code. This catches the common issue where code runs fine but the technique is silently disabled.

### 📋 Comprehensive Review
The reviewer does a final check covering code quality, configuration, edge cases, and performance before approving for production.

### 🛡️ Error Prevention
The workflow prevents common mistakes:
- ❌ Implementing without understanding
- ❌ Code that runs but technique is inactive
- ❌ Missing edge cases or compatibility issues
- ❌ Proceeding with unverified changes

## Workflow Example

**Implementing Exclusive Self-Attention (XSA)**:

```
Stage 1: Research
- Found XSA in 3 submissions
- Expected impact: ~0.005 BPB
- Typically applied to last 4 layers
- Provides code examples

Stage 2: Plan
- Modify CausalSelfAttention class
- Add XSA_LAST_N hyperparameter
- Apply to layers 7-10 (last 4 of 11)
- Detailed steps with line numbers

Stage 3: Implement
- Make code changes following plan
- Test incrementally
- Handle any errors

Stage 4: Verify
- ✅ Syntax valid
- ✅ Smoke test passes
- ✅ XSA confirmed active (with evidence!)
- ✅ Performance acceptable

Stage 5: Review
- ✅ Code quality good
- ✅ Technique enabled
- ✅ Compatible with existing techniques
- ✅ Edge cases handled
- ✅ APPROVED for production
```

## Best Practices

### For Researchers
- Never proceed without understanding
- Cite all sources (submissions, PRs, papers)
- Quantify expected impact with evidence
- Be honest about uncertainty
- **Use GitHub MCP** when user mentions PRs
- **Extract key info** from papers when provided
- **Cross-reference** external and local sources

### For Planners
- Make minimal, focused changes
- Provide exact line numbers
- Plan verification at each step
- Consider compatibility issues

### For Implementers
- **Implement one change at a time**
- **Test after each change** (syntax, imports, basic tests)
- Match existing code style
- Fix errors immediately before proceeding
- Clean up debug code
- Document changes as you go

### For Verifiers
- Run all levels in order
- **Critical**: Verify technique is actually active (Level 3)
- Provide clear evidence
- Don't skip tests
- Use proper `py_compile` syntax checking

### For Reviewers
- Be thorough - this is the final gate
- Check that technique is enabled (common issue!)
- Test edge cases
- Document concerns clearly

## Integration

The agent system integrates with existing parameter-golf skills:

- `/parameter-golf` - Main overview
- `/parameter-golf-techniques` - Technique catalog (used by Researcher)
- `/parameter-golf-implement` - Implementation patterns (used by Planner)
- `/parameter-golf-verify` - Verification guide (used by Verifier)
- `/parameter-golf-submit` - Submission process (after Review)

## Why This System?

Traditional approach:
```
User: "Add XSA"
Claude: [makes changes]
User: [runs code, doesn't see improvement]
Problem: XSA was disabled by default flag!
```

Agent workflow approach:
```
User: "Add XSA"
Research: XSA found in 3 submissions, ~0.005 BPB gain
Plan: Add XSA_LAST_N=4, modify attention in last 4 layers
Implement: [changes made]
Verify: Added logging, confirmed XSA active in layers 7-10 ✅
Review: Double-checked XSA_LAST_N defaults to 4 (enabled) ✅
Result: Technique actually works!
```

## When to Use

**Always use the agent workflow** for:
- Implementing new techniques
- Adding architecture features
- Modifying training logic
- Changing evaluation methods

**Skip the full workflow** only for:
- Trivial changes (adjusting a constant)
- Documentation updates
- Bug fixes in non-critical code

## Future Enhancements

Potential improvements:
- Add `implementer.md` for more detailed implementation guidance
- Create technique-specific agent variants
- Add automated testing scripts
- Build a technique compatibility database
- Generate submission documentation automatically

## Notes

The agent instructions are designed to be read and followed by Claude Code when executing the workflow. They provide detailed, step-by-step guidance for each stage while allowing flexibility for different techniques and situations.

The key insight is that **verification of technique activation** (Stage 4, Level 3) is critical - code might run perfectly but the technique could be disabled by a flag, unreachable in the code path, or silently falling back to default behavior. The verifier specifically addresses this with multiple verification strategies.
