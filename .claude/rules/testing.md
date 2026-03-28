# Submission & Testing Rules

## Verification is Mandatory

Before any technique is considered "done", it MUST pass all applicable levels of the verification pyramid:

| Level | Check | Time |
|-------|-------|------|
| 1 | Python syntax valid | instant |
| 2 | Smoke test (10–100 steps, no crash) | 1–5 min |
| 3 | Technique **actually active** (not silently disabled) | varies |
| 4 | Full 1×H100 run (bpb, timing, artifact size) | 10 min |
| 5 | 3-seed 8×H100 statistical test | ~30 min |

**Never skip Level 3.** The most common failure mode is code that runs but whose technique is inactive due to a flag or unreachable code path.

## Statistical Significance
A SOTA record submission requires:
- ≥3 seeds (standard: 1337, 42, 2025)
- Mean improvement ≥0.005 BPB over current SOTA
- p-value <0.01 (paired t-test)

## Artifact Size Budget
The 16 MB limit is **hard**: `code_bytes + zlib-compressed-model_bytes < 16,000,000`

Always reserve a buffer: target ≤15.5 MB to avoid marginal failures across seeds.

## PR Structure
All submissions go in `records/track_10min_16mb/YYYY-MM-DD_<short-name>/` with:
- `README.md` — technique explanation + results
- `submission.json` — name, github_id, val_bpb, bytes_total, seeds
- `train_SEED.log` × 3 — stdout from each seed run
- `train_gpt.py` — self-contained, runnable script
