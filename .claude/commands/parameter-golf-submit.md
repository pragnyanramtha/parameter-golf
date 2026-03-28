---
name: parameter-golf-submit
description: Step-by-step guide for submitting to Parameter Golf leaderboard
---

# Parameter Golf Submission Process

This skill guides you through the complete submission process for the Parameter Golf challenge.

## Submission Requirements Overview

**Record submissions** (for leaderboard) must:
1. Beat existing SOTA by ≥0.005 nats with p < 0.01 statistical significance
2. Prove val_bpb is correctly calculated (scrutiny if tokenizer changed)
3. Reproducibly run in <10 min on 8xH100 SXM
4. Include all required files (detailed below)

**Non-record submissions** can be:
- Unique/interesting approaches that don't beat SOTA
- In-progress or unoptimized solutions
- Interesting negative results
- Unlimited compute track runs (note in README)

## Step-by-Step Submission Process

### Step 1: Create Submission Folder

**Location**: `records/track_10min_16mb/YYYY-MM-DD_DescriptiveName/`

**Naming convention**:
- Date: YYYY-MM-DD of submission
- Name: Brief descriptor of key technique(s)
- Examples:
  - `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`

```bash
cd parameter-golf
mkdir -p records/track_10min_16mb/2026-03-25_YourTechnique
cd records/track_10min_16mb/2026-03-25_YourTechnique
```

For non-record: Use `records/track_non_record_16mb/` instead.

---

### Step 2: Create README.md

**Required sections**:

#### Header with Key Metrics
```markdown
# Your Technique Name

**val_bpb: 1.1234** (3-seed mean, std 0.0005) | **15.6 MB** | 8×H100 SXM
```

#### Results Table (3 seeds minimum for records)
```markdown
## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | ms/step | val_bpb | Artifact |
|------|-------|---------|---------|----------|
| 1337 | 7,051 | 85.0 | 1.1234 | 15,612,308 |
| 42   | 7,061 | 84.8 | 1.1236 | 15,528,666 |
| 2025 | 7,063 | 85.2 | 1.1235 | 15,639,340 |
| **Mean** | **7,058** | **85.0** | **1.1235 (std 0.0003)** | |
```

#### Key Innovations
Clearly explain what's new in this submission:
```markdown
## Key Innovation: Your Technique

[Concise explanation]

[Code snippet showing the change]

[Why it works]
```

#### Architecture Details
```markdown
## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU² |
| Attention | XSA last 4 layers |
| Embeddings | Tied, BigramHash(2048) |
| Position | Partial RoPE (16/64) |
| [etc.] | [details] |
```

#### Training Configuration
```markdown
## Training

- Optimizer: Muon (matrices) + AdamW (embeddings)
- Learning rates: matrix=0.025, tied_embed=0.035, scalar=0.025
- Momentum: 0.99 (warmup 0.92→0.99 over 1500 steps)
- Weight decay: 0.04 (both optimizers)
- Batch: 786,432 tokens/step, seq_len=2048
- Schedule: 20 warmup, 3500 warmdown
- Grad clip: 0.3
- Weight averaging: EMA(0.997) + SWA(every 50)
```

#### Quantization & Compression
```markdown
## Quantization

- Int6 per-row for MLP + attention weights
- GPTQ-lite with 5-percentile search
- Int8 per-row for embeddings
- FP32 for control tensors (scales, gates)
- zstd level 22 compression
```

#### Run Command
```markdown
## Run Command

\`\`\`bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \\
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 \\
MATRIX_LR=0.025 TIED_EMBED_LR=0.035 \\
ITERATIONS=9000 WARMDOWN_ITERS=3500 \\
SEED=1337 \\
torchrun --standalone --nproc_per_node=8 train_gpt.py
\`\`\`
```

#### Ablation Study (Recommended)
Show incremental contribution:
```markdown
## Ablation

| Change | val_bpb | Delta |
|--------|---------|-------|
| Base (PR #XXX) | 1.1250 | baseline |
| + Your technique A | 1.1245 | -0.0005 |
| + Your technique B | 1.1234 | -0.0011 |
```

#### Credits
Always credit prior work:
```markdown
## Credits

- Base architecture: [PR #XXX](link) by @username
- Technique A: [PR #YYY](link) by @username
- Technique B: Paper title (arXiv:XXXX.XXXXX)
```

---

### Step 3: Create submission.json

**Required fields**:
```json
{
  "name": "Descriptive Technique Name",
  "val_bpb": 1.1234,
  "bytes_total": 15612308,
  "blurb": "One-line summary of key techniques and result. Include major components: architecture, optimizer, quantization, eval method.",
  "author": "Your Name",
  "github_id": "yourusername",
  "date": "YYYY-MM-DD"
}
```

**Field guidelines**:
- `name`: Match README title
- `val_bpb`: Use best seed OR 3-seed mean (document in README)
- `bytes_total`: Exact artifact size in bytes
- `blurb`: Keep under 200 chars, hit key points
- `date`: Submission date, not run date

**Example**:
```json
{
  "name": "LeakyReLU² + Legal Score-First TTT + Parallel Muon",
  "val_bpb": 1.1194,
  "bytes_total": 15990006,
  "blurb": "LeakyReLU(0.5)² activation (-0.003 BPB vs relu²) + legal score-first TTT (PR #461 recipe, 3ep SGD, all blocks unfrozen) + BigramHash(1536) + Parameter Banking + Parallel Muon (PR #399). Built on PR #414 stack. 3-seed mean: 1.1194 (std 0.0006).",
  "author": "abaybektursun",
  "github_id": "abaybektursun",
  "date": "2026-03-23"
}
```

---

### Step 4: Include Training Script

**File**: `train_gpt.py` (or similar name like `train_gpt_ternary.py`)

**Critical requirements**:
- Must run successfully from within the records folder
- Must be completely self-contained
- Include all custom modules, functions inline
- Can import standard packages (add to requirements.txt)
- Should include the full training, quantization, and evaluation pipeline

**Test your script**:
```bash
# From records folder
cd records/track_10min_16mb/2026-03-25_YourTechnique/

# Verify it runs
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"

# Smoke test (if dataset available)
RUN_ID=smoke_test ITERATIONS=10 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

---

### Step 5: Include Training Logs

**Requirements**:
- **SOTA records**: 3+ seed runs showing statistical significance
- **Non-records**: 1+ seed acceptable
- **File naming**: `train.log` or `train_seed1337.log`, `train_seed42.log`, etc.

**What logs must show**:
```
Training progress with step times
Final validation metrics:
  - val_loss (natural log cross-entropy)
  - val_bpb (bits per byte, primary metric)
  - Artifact size in bytes
Statistical evidence:
  - Mean across seeds
  - Standard deviation
  - Must show >0.005 improvement with p<0.01 for records
```

**Capturing logs**:
```bash
# Run with logging
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log

# For 3 seeds
for seed in 1337 42 2025; do
  SEED=$seed torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```

---

### Step 6: Validate Submission Locally

**Pre-submission checklist**:

```bash
# 1. Check folder structure
ls -lh records/track_10min_16mb/2026-03-25_YourTechnique/
# Must contain: README.md, submission.json, train_gpt.py, train*.log

# 2. Validate JSON syntax
python3 -c "import json; print(json.load(open('submission.json')))"

# 3. Check script syntax
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"

# 4. Verify artifact size
# Check bytes_total in submission.json matches artifact size in logs

# 5. Check total folder size
du -sh .
# Should be reasonable (mostly logs, artifact not included in repo)
```

**If validation script exists** (PR #683 added this):
```bash
# From repo root
python3 validate_submission.py records/track_10min_16mb/2026-03-25_YourTechnique/
```

---

### Step 7: Create Pull Request

**Branch naming**: Use descriptive name
```bash
git checkout -b submission/your-technique-name
```

**Add only your submission folder**:
```bash
git add records/track_10min_16mb/2026-03-25_YourTechnique/
git commit -m "Record: Your Technique Name (val_bpb: 1.1234)"

# OR for non-record
git commit -m "Non-record: Your Technique Name (val_bpb: 1.1234)"
```

**Push and create PR**:
```bash
git push origin submission/your-technique-name

# Create PR via gh CLI
gh pr create --title "Record: Your Technique Name (val_bpb: 1.1234)" \
  --body "$(cat <<'EOF'
## Summary

[Concise summary of your submission]

## Results

[Table with seeds and metrics]

## Key Innovation

[What's new]

## Test plan
- [x] 3 seeds run on 8xH100
- [x] All artifacts under 16MB
- [x] Training completes in <10 min
- [x] Eval completes in <10 min
- [x] Statistical significance demonstrated

EOF
)"
```

**PR Title format**:
- Record: `Record: Technique Name (val_bpb: 1.1234)`
- Non-record: `Non-record: Technique Name (val_bpb: 1.1234)`

---

### Step 8: Post-Submission

**Chronological acceptance**: Submissions accepted by PR creation time (first-come, first-served for SOTA).

**Verification**: OpenAI verifies top submissions. Community can raise reproducibility issues.

**Timeline**: Leaderboard may take time to update due to verification.

**If your submission has issues**:
- Respond to PR feedback
- Fix and update (allowed before merge)
- Close and resubmit if major changes needed

---

## Statistical Significance Testing

**For SOTA records**, you must prove ≥0.005-nat improvement with p < 0.01.

**Formula**:
```python
import numpy as np
from scipy import stats

# Your 3 seed results
your_bpbs = [1.1192, 1.1200, 1.1189]  # Example
sota_bpb = 1.1228  # Current SOTA

your_mean = np.mean(your_bpbs)
your_std = np.std(your_bpbs, ddof=1)

# One-sample t-test
t_stat = (your_mean - sota_bpb) / (your_std / np.sqrt(len(your_bpbs)))
p_value = stats.t.cdf(t_stat, df=len(your_bpbs)-1)

print(f"Mean: {your_mean:.4f}")
print(f"Improvement: {sota_bpb - your_mean:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Significant: {p_value < 0.01 and (sota_bpb - your_mean) >= 0.005}")
```

**Include in README**:
```markdown
## Statistical Significance

- Mean improvement: 0.0034 nats
- Standard deviation: 0.0006
- p-value: 0.00234 (< 0.01) ✓
- Improvement > 0.005: Yes ✓
```

---

## Common Submission Mistakes to Avoid

### 1. Missing Files
**Problem**: Forgot train.log or requirements.txt
**Solution**: Use checklist, run validation script

### 2. Wrong Folder Location
**Problem**: Submitted to root or wrong track
**Solution**: Must be in `records/track_10min_16mb/` or `records/track_non_record_16mb/`

### 3. Script Won't Run from Records Folder
**Problem**: Hardcoded paths like `../../data/`
**Solution**: Use environment variables or make paths configurable

```python
# Good
data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")

# Bad
data_path = "../../data/datasets/fineweb10B_sp1024"  # Breaks from records folder
```

### 4. Artifact Size Over Limit
**Problem**: Total bytes > 16,000,000
**Solution**:
- Reduce model size (fewer layers, smaller dim)
- Use more aggressive quantization (int5, ternary)
- Optimize compression (try lzma instead of zstd)
- Remove unnecessary control tensors

### 5. Tokenizer Bugs
**Problem**: Changed tokenizer, val_bpb calculation wrong
**Solution**:
- Submissions editing tokenizer face heavy scrutiny
- Must prove correctness of BPB calculation
- Verify: bytes-per-token counting logic correct
- Test: compare against known-good tokenizer

### 6. Insufficient Statistical Evidence
**Problem**: Only 1-2 seeds, or inter-seed variance too high
**Solution**:
- Run minimum 3 seeds for records
- If std > 0.002, run 4-5 seeds
- Calculate p-value properly

### 7. Non-Reproducible Results
**Problem**: Can't reproduce claimed score
**Solution**:
- Fix all random seeds
- Document exact package versions
- Include complete environment (requirements.txt)
- Test script before submitting

---

## Non-Record Submission Guidelines

**When to submit as non-record**:
1. Doesn't beat SOTA but explores interesting idea
2. Negative result with lessons for community
3. Unlimited compute track (>10 min training)
4. In-progress/unoptimized but working solution
5. Novel architecture worth documenting

**Lower bar** but still require:
- Working code that runs successfully
- Reasonable justification and explanation
- Clear documentation of what was tried
- Honest assessment of results

**Examples of good non-record submissions**:
- PR #670: "Negative results — hardware alignment" - documents 30+ failed experiments
- PR #666: "BitNet Ternary — 65M params" - novel approach, doesn't beat SOTA yet
- PR #645: "Skill Forge — Autonomous ML system" - infrastructure contribution

---

## File Checklist

Before submitting, verify you have:

- [ ] **README.md** with:
  - [ ] Clear title with technique name
  - [ ] Results table with 3 seeds (or 1+ for non-record)
  - [ ] val_bpb, artifact size, hardware info
  - [ ] Key innovations explained
  - [ ] Architecture details
  - [ ] Training hyperparameters
  - [ ] Run command
  - [ ] Ablations (if applicable)
  - [ ] Credits to prior work

- [ ] **submission.json** with:
  - [ ] All required fields (name, val_bpb, bytes_total, blurb, author, github_id, date)
  - [ ] Valid JSON syntax
  - [ ] Accurate metrics matching logs

- [ ] **train_gpt.py** (or similar):
  - [ ] Complete self-contained script
  - [ ] Runs from records folder
  - [ ] Valid Python syntax
  - [ ] Includes dependencies in requirements.txt (if custom packages)

- [ ] **Training logs**:
  - [ ] Minimum 3 for records, 1+ for non-records
  - [ ] Shows complete training run
  - [ ] Includes final metrics
  - [ ] Named clearly (train.log or train_seedXXXX.log)

- [ ] **Optional but recommended**:
  - [ ] requirements.txt (if using non-standard packages)
  - [ ] Ablation logs
  - [ ] Visualizations (ablations.png, etc.)
  - [ ] Additional documentation for complex techniques

---

## Example Minimal Working Submission Structure

```
records/track_10min_16mb/2026-03-25_YourTechnique/
├── README.md              (detailed explanation)
├── submission.json        (metadata)
├── train_gpt.py           (complete training script)
├── train_seed1337.log     (seed 1337 log)
├── train_seed42.log       (seed 42 log)
├── train_seed2025.log     (seed 2025 log)
└── requirements.txt       (optional, if custom packages)
```

---

## Testing Before Submission

### Local Syntax Check
```bash
# Python syntax
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"

# JSON syntax
python3 -c "import json; json.load(open('submission.json'))"
```

### Smoke Test (if possible)
```bash
# Short run to verify script works
RUN_ID=smoke ITERATIONS=100 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Full Reproducibility Test
```bash
# Ideally, re-run one seed to verify reproducibility
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
# Compare output to original log
```

---

## PR Description Template

```markdown
## Summary

[2-3 sentence overview of your submission]

**val_bpb: 1.1234** (3-seed mean) | **15.6 MB** | 8×H100 SXM

## Key Changes from [Previous SOTA/Base]

| Component | Previous | This | Impact |
|-----------|----------|------|--------|
| Technique A | Old way | New way | -0.0010 BPB |
| Technique B | None | Added | -0.0005 BPB |

## Results

[Paste your 3-seed results table]

## Why This Works

[Brief explanation of key innovation]

## Compliance Checklist

- [x] 3 seeds run on 8×H100 SXM
- [x] All training runs complete in ≤600s
- [x] All artifacts ≤16,000,000 bytes
- [x] Eval completes in ≤600s (if applicable)
- [x] Beats SOTA by ≥0.005 nats with p<0.01 (for records)
- [x] No validation data access during training
- [x] All required files included
- [x] Script runs from records folder

## Credits

[List prior work and techniques used]
```

---

## Timeline and Review Process

**Acceptance timeline**:
- PRs accepted chronologically by creation time
- First valid SOTA PR gets priority
- Later PRs with similar scores may be rejected or accepted as non-record

**Review process**:
1. Automated checks (syntax, file presence)
2. Manual review of technique and code
3. Reproducibility verification (for top submissions)
4. Leaderboard update

**Typical wait time**: Hours to days depending on reviewer availability and queue

---

## After Acceptance

**Merged submissions**:
- Appear on main README leaderboard
- Code becomes reference for others
- You're credited in future work building on yours

**Best practices post-merge**:
- Monitor issues on your PR
- Respond to reproducibility questions
- Help others understand your techniques

---

## Special Cases

### Tokenizer Changes
If you modified the tokenizer:
- **Extra scrutiny**: BPB calculation bugs can unjustly improve score
- **Requirements**:
  - Prove correctness of bytes-per-token counting
  - Show validation against baseline tokenizer
  - Document exact changes and rationale
  - Include tokenizer training code/config

### Custom Libraries
If you created custom libraries:
- Must not unjustly violate rules
- Can't sneak extra compute or massive code size increase
- Document in requirements.txt
- Explain in README why library needed
- Standard libraries (FlashAttention, etc.) are fine

### External Compute
If you trained auxiliary components offline:
- **Tuning hyperparameters**: Acceptable
- **Brute-forcing seeds**: Not acceptable
- **Pre-training components**: Must fit in 16MB or declare as non-record
- **Use judgment**: Ask if unsure

---

## Quick Reference: Submission Commands

```bash
# 1. Create folder
mkdir -p records/track_10min_16mb/$(date +%Y-%m-%d)_YourTechnique
cd records/track_10min_16mb/$(date +%Y-%m-%d)_YourTechnique

# 2. Copy your train script
cp /path/to/your/train_gpt.py .

# 3. Run 3 seeds
for seed in 1337 42 2025; do
  SEED=$seed torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${seed}.log
done

# 4. Create submission.json
cat > submission.json <<EOF
{
  "name": "Your Technique",
  "val_bpb": 1.1234,
  "bytes_total": 15600000,
  "blurb": "Brief summary",
  "author": "Your Name",
  "github_id": "yourusername",
  "date": "$(date +%Y-%m-%d)"
}
EOF

# 5. Write README.md
# [Use your text editor]

# 6. Validate
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"
python3 -c "import json; print(json.load(open('submission.json')))"

# 7. Commit and PR
cd ../../..  # Back to repo root
git add records/track_10min_16mb/$(date +%Y-%m-%d)_YourTechnique/
git commit -m "Record: Your Technique (val_bpb: 1.1234)"
git push origin submission/your-technique-name
gh pr create --web
```

---

## Getting Help

**Before submitting**:
- Ask in Discord #parameter-golf-discussions
- Review similar submissions for format guidance
- Check recent PRs for current expectations

**During review**:
- Respond promptly to reviewer questions
- Provide additional logs/evidence if requested
- Be open to feedback and suggestions

**After rejection/issues**:
- Understand the reason (artifact size, reproducibility, significance, etc.)
- Fix issues and resubmit
- Or pivot to non-record submission

---

## Competition Participant Form

Optional but recommended: https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

**Benefits**:
- Helps OpenAI attribute submissions
- Reach out about opportunities
- Not required to participate

**Hiring**: Small cohort of early-career researchers in June 2026 for exceptional participants.

---

## Related Skills

- Use `/parameter-golf` for general challenge overview
- Use `/parameter-golf-techniques` for detailed technique implementations
- Use `/parameter-golf-verify` for validation and testing strategies
