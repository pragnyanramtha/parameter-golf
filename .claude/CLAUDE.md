# Parameter Golf — Claude Instructions

## Project Overview

**OpenAI Model Craft Challenge: Parameter Golf** — train the best language model that fits in a **16 MB artifact** and trains in **under 10 minutes on 8×H100s**, evaluated by compression on the FineWeb validation set (bits-per-byte, lower is better).

- **Baseline score**: ~1.2244 BPB (9L/512d/1024-vocab tied-embeddings)
- **Current SOTA**: ~1.1194 BPB (as of March 2026)
- **Submission target**: `train_gpt.py` + compressed model weights ≤ 16,000,000 bytes total
- **Evaluation metric**: tokenizer-agnostic bits-per-byte on FineWeb validation set

---

## Workspace Layout

```
parameter-golf/
                   
├── CLAUDE.local.md                  # personal overrides (gitignored)
├── train_gpt.py                     # primary submission script (CUDA/PyTorch)
├── train_gpt_mlx.py                 # Apple Silicon MLX variant
├── requirements.txt                 # package dependencies
├── data/
│   ├── cached_challenge_fineweb.py  # dataset download helper
│   ├── datasets/fineweb10B_sp1024/  # training shards (gitignored)
│   └── tokenizers/                  # tokenizer models (gitignored)
├── records/
│   ├── track_10min_16mb/            # official leaderboard submissions
│   └── track_non_record_16mb/       # interesting non-record submissions
├── prs/                             # tracked PR experiments
└── .claude/
    ├── CLAUDE.md                    # ← you are here
    ├── settings.json
    ├── agents/                      # specialized subagent personas
    ├── commands/                    # slash-command skills (legacy + active)
    ├── rules/                       # modular instruction files
    └── skills/                      # auto-invoked workflow skills
```

---

## Agent System

The project uses a **5-stage pipeline** of specialized agents. Always prefer this over ad-hoc changes:

```
Research → Plan → Implement → Verify → Review
```

| Agent | File | Role |
|-------|------|------|
| Researcher 🔍 | `.claude/agents/researcher.md` | Understand the technique; search PRs/papers/repo |
| Planner 📋 | `.claude/agents/planner.md` | Design exact code changes with line numbers |
| Implementer 💻 | `.claude/agents/implementer.md` | Make incremental, tested changes |
| Verifier ✅ | `.claude/agents/verifier.md` | 5-level verification pyramid |
| Reviewer 🔍 | `.claude/agents/reviewer.md` | Final quality gate before production |

Invoke the full workflow via `/parameter-golf-agent`.

---

## Slash Commands (`.claude/commands/`)

| Command | Purpose |
|---------|---------|
| `/parameter-golf` | Project overview & orientation |
| `/parameter-golf-agent` | Full 5-stage agent orchestration |
| `/parameter-golf-techniques` | Catalogue of known techniques (XSA, EMA, QAT, etc.) |
| `/parameter-golf-implement` | Implementation patterns & code snippets |
| `/parameter-golf-verify` | 5-level verification pyramid |
| `/parameter-golf-submit` | Submission checklist & PR process |

---

## Key Constraints — NEVER Violate

1. **16 MB hard limit** — `code_bytes + zlib_compressed_model_bytes < 16,000,000`
2. **10 min training** — `torchrun --nproc_per_node=8 train_gpt.py` must finish in ≤600 s on 8×H100 SXM
3. **10 min evaluation** — eval pass (including any TTT) must also finish in ≤600 s
4. **No external downloads** — artifact must be fully self-contained at eval time
5. **No training on validation set** — TTT may only see already-scored tokens
6. **Score metric** — always report `val_bpb` (bits-per-byte), NOT `val_loss`

---

## Coding Conventions

### General
- Stay within `train_gpt.py` for all submission-relevant code
- Match the style of existing code (no black formatting rewrites)
- Prefer env-var feature flags (e.g., `XSA_LAST_N = int(os.getenv("XSA_LAST_N", 0))`) for ablations
- Remove all debug prints before committing

### Verification
Run the 5-level pyramid bottom-up before declaring anything done:

```
Level 1 – Syntax:    python3 -c "import ast; ast.parse(open('train_gpt.py').read())"
Level 2 – Smoke:     ITERATIONS=10 VAL_LOSS_EVERY=0 python3 train_gpt.py
Level 3 – Feature:   Confirm technique is ACTUALLY active (logs/hooks/inspection)
Level 4 – 1×H100:    Full 10-min run, check steps/mem/bpb
Level 5 – 8×H100:    Competition run, 3-seed statistical test
```

### Quantization
- Acceptable quantization gap: 0.005–0.03 BPB (alert if >0.03)
- Default precision stack: int6 weights + per-row scales + zstd-22 compression
- QAT: enable late (last 25% of training) to reduce quantization gap

### Submission
- Beat current SOTA by ≥0.005 BPB at p<0.01 (3+ seeds, t-test)
- New folder under `records/track_10min_16mb/YYYY-MM-DD_<name>/`
- Required files: `README.md`, `submission.json`, train logs (3 seeds), `train_gpt.py`

---

## External Resources

- **GitHub repo**: `openai/parameter-golf`
- **Discord**: OpenAI server → `#parameter-golf-discussions`
- **Compute**: RunPod (1×H100 for iteration, 8×H100 SXM for submission)
- **Dataset**: FineWeb 10B tokens, sp1024 vocabulary
- **Compute grants**: https://openai.com/index/parameter-golf/#credit-form

---

## Common Gotchas

- Techniques may be present in code but **silently disabled** by a flag — always verify activation (Level 3)
- EMA replaces SWA in most recent SOTA; don't mix both unless intentional
- Sliding window eval (`stride=64`) is standard — use it for fair comparison
- Val tokens are shared across all submissions; never shuffle or re-seed the val loader
- `bytes_total` in `submission.json` must match the actual exported artifact exactly
- TTT must score under `torch.inference_mode()` BEFORE any `backward()` on that chunk
