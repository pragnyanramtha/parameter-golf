# Coding Style Rules

## Python Style
- Match the existing code style in `train_gpt.py` — do NOT apply auto-formatters (black, ruff) across the file
- Use 4-space indentation (already standard)
- Keep lines under ~120 characters where possible
- Preserve existing naming conventions (`camelCase` for tensors, `snake_case` for functions/vars)

## Feature Flags
All experimental features MUST be gated behind environment variable flags:

```python
# Pattern: default OFF (0) for new experiments, can enable via env
MY_FEATURE = int(os.getenv("MY_FEATURE", 0))

# Pattern: default ON for established techniques
EMA_ENABLED = int(os.getenv("EMA_ENABLED", 1))
```

This allows clean ablation testing without code changes.

## Submission Hygiene
- Remove ALL debug `print()` statements before committing
- Remove any `breakpoint()` or `pdb` calls
- Do not commit temporary log files to `records/` without corresponding `submission.json`

## Research & Comments
- When implementing a novel technique, add a short comment block citing the source PR or paper:
  ```python
  # XSA (Exclusive Self-Attention): see PR #198 by jfprincz
  # Applied to last XSA_LAST_N layers only
  ```
