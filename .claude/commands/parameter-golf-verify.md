---
name: parameter-golf-verify
description: Testing, verification, and debugging guide for Parameter Golf submissions
---

# Parameter Golf Verification & Testing

This skill provides comprehensive guidance for testing, verifying, and debugging Parameter Golf submissions.

## Testing Pyramid

```
         3-Seed Statistical Test (8xH100, ~2 hours)
                      /\
                     /  \
                    /    \
        Full Run Test (8xH100, 10 min)
                  /      \
                 /        \
               /          \
    1xH100 Ablation Test (10 min)
              /            \
             /              \
           /                \
  Local Smoke Test (100-1000 steps)
          /                  \
         /                    \
       /                      \
  Syntax Check (instant)
```

Work bottom-up: fix issues at each level before moving to next.

---

## Level 1: Syntax Validation

**Goal**: Verify code is valid Python and JSON

**Time**: Instant

**Commands**:
```bash
# Python syntax
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"
echo "✓ Python syntax valid"

# JSON syntax
python3 -c "import json; print(json.load(open('submission.json')))"
echo "✓ JSON valid"

# Check for common issues
python3 << 'EOF'
import ast
with open('train_gpt.py') as f:
    tree = ast.parse(f.read())

# Check for suspicious patterns
imports = [node.names[0].name for node in ast.walk(tree)
           if isinstance(node, ast.Import)]
print(f"Imports: {', '.join(imports)}")

# Check line count
with open('train_gpt.py') as f:
    lines = len(f.readlines())
print(f"Lines: {lines} {'✓' if lines <= 1500 else '✗ Over 1500!'}")
EOF
```

**What to check**:
- [ ] No syntax errors
- [ ] Valid imports (no missing dependencies)
- [ ] Under 1500 lines (for baseline script)
- [ ] JSON fields present and correct types

---

## Level 2: Local Smoke Test

**Goal**: Verify training runs without errors

**Time**: 1-5 minutes

**Environment**: Local machine or single GPU

**Commands**:
```bash
# Minimal smoke test (10 iterations, no validation)
RUN_ID=smoke_test \
ITERATIONS=10 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
python3 train_gpt.py

# Slightly longer smoke test (100 iterations)
RUN_ID=smoke_100 \
ITERATIONS=100 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
python3 train_gpt.py

# With validation (if dataset available)
RUN_ID=smoke_val \
ITERATIONS=200 \
VAL_LOSS_EVERY=100 \
TRAIN_LOG_EVERY=50 \
python3 train_gpt.py
```

**What to verify**:
- [ ] Script starts without import errors
- [ ] Model initializes without OOM
- [ ] Training loop runs without crashes
- [ ] Loss decreases (should drop from ~8 to ~3-4 in 100 steps)
- [ ] No NaN or Inf in losses
- [ ] If validation runs, produces val_loss and val_bpb

**Common issues**:
- **Import errors**: Add missing packages to requirements.txt
- **Shape mismatches**: Check tensor dimensions in custom modules
- **OOM**: Reduce batch size or sequence length for testing
- **CUDA errors**: Verify CUDA availability, driver compatibility

---

## Level 3: 1xH100 Ablation Test

**Goal**: Full 10-minute run on single H100 to verify end-to-end pipeline

**Time**: 10 minutes

**Environment**: RunPod 1xH100 or equivalent

**Expected results**:
- ~900-1100 steps in 600s
- Final val_bpb: 1.25-1.35 (single GPU, not competitive)
- Artifact size: Should be under 16MB

**Commands**:
```bash
# Setup on RunPod
cd /workspace
git clone https://github.com/yourusername/parameter-golf.git
cd parameter-golf
git checkout your-branch

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Run full test
RUN_ID=1xh100_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee test_1xh100.log
```

**What to verify**:
- [ ] Completes within 600s (check wallclock in log)
- [ ] No CUDA errors or OOM
- [ ] Produces final val_loss and val_bpb
- [ ] Artifact exported successfully
- [ ] Artifact size reported and < 16MB
- [ ] Quantization roundtrip succeeds (if implemented)

**Parse results**:
```bash
# Extract key metrics from log
grep "step_avg:" test_1xh100.log | tail -1
grep "val_bpb:" test_1xh100.log | tail -1
grep "bytes" test_1xh100.log | tail -1
```

---

## Level 4: Full 8xH100 Run

**Goal**: Competition-scale run to verify performance and timing

**Time**: 10 minutes

**Environment**: RunPod 8xH100 SXM

**Expected results**:
- ~6800-7200 steps in 600s (83-92ms/step)
- Final val_bpb: Competitive (depends on techniques)
- Artifact size: < 16MB
- Peak memory: ~15-25GB per GPU

**Commands**:
```bash
# On 8xH100 pod
cd /workspace/parameter-golf
git checkout your-branch
python3 data/cached_challenge_fineweb.py --variant sp1024  # Full dataset

# Run with optimal settings
RUN_ID=8xh100_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

**What to verify**:
- [ ] Step time: 82-95ms/step (slower is red flag)
- [ ] Steps completed: >6800 (more is better)
- [ ] No stragglers (all GPUs finish together)
- [ ] Peak memory: <80GB per GPU
- [ ] Val_bpb: Competitive with similar techniques
- [ ] Artifact size: <16,000,000 bytes exactly
- [ ] No warnings about quantization errors

**Monitoring during run**:
```bash
# In another terminal, monitor progress
watch -n 5 'tail -20 train_seed1337.log'

# Check GPU utilization
watch -n 1 nvidia-smi
```

---

## Level 5: Multi-Seed Statistical Test

**Goal**: Prove statistical significance across random seeds

**Time**: ~30-40 minutes (3 seeds)

**Requirements for SOTA records**:
- Minimum 3 seeds
- Mean improvement ≥0.005 BPB over current SOTA
- p-value < 0.01 via t-test

**Commands**:
```bash
# Run 3 standard seeds
for seed in 1337 42 2025; do
  RUN_ID=8xh100_seed${seed} SEED=${seed} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${seed}.log
done

# Compute statistics
python3 << 'EOF'
import re
import numpy as np
from scipy import stats

# Extract val_bpb from logs
seeds = [1337, 42, 2025]
bpbs = []
artifacts = []

for seed in seeds:
    with open(f'train_seed{seed}.log') as f:
        log = f.read()
        # Find final val_bpb
        match = re.search(r'final.*val_bpb[:\s]+([\d.]+)', log)
        if match:
            bpbs.append(float(match.group(1)))
        # Find artifact size
        match = re.search(r'bytes_total[:\s]+([\d]+)', log)
        if match:
            artifacts.append(int(match.group(1)))

bpbs = np.array(bpbs)
artifacts = np.array(artifacts)

print(f"\n{'='*60}")
print(f"STATISTICAL SUMMARY")
print(f"{'='*60}")
print(f"Seeds: {seeds}")
print(f"BPBs: {bpbs}")
print(f"\nMean BPB: {bpbs.mean():.6f}")
print(f"Std BPB: {bpbs.std(ddof=1):.6f}")
print(f"Min BPB: {bpbs.min():.6f}")
print(f"Max BPB: {bpbs.max():.6f}")
print(f"\nMean Artifact: {artifacts.mean():.0f} bytes")
print(f"Max Artifact: {artifacts.max()} bytes")
print(f"Under 16MB: {'✓' if artifacts.max() < 16_000_000 else '✗ OVER LIMIT!'}")

# Test against SOTA (update this value)
sota_bpb = 1.1194  # Current SOTA as of March 2026
improvement = sota_bpb - bpbs.mean()
t_stat = improvement / (bpbs.std(ddof=1) / np.sqrt(len(bpbs)))
p_value = stats.t.cdf(-abs(t_stat), df=len(bpbs)-1) * 2  # Two-tailed

print(f"\n{'='*60}")
print(f"COMPARISON TO SOTA ({sota_bpb:.4f})")
print(f"{'='*60}")
print(f"Improvement: {improvement:.6f} BPB")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Significant (p<0.01): {'✓' if p_value < 0.01 else '✗'}")
print(f"Beats SOTA by ≥0.005: {'✓' if improvement >= 0.005 else '✗'}")
print(f"\n{'='*60}")
print(f"SUBMISSION READY: {'✓ YES' if p_value < 0.01 and improvement >= 0.005 and artifacts.max() < 16_000_000 else '✗ NO'}")
print(f"{'='*60}\n")
EOF
```

**If variance too high** (std > 0.002):
- Run 4-5 seeds instead of 3
- Check for sources of non-determinism
- Verify proper seeding of all RNGs

---

## Reproducibility Verification

**Goal**: Ensure identical results on repeated runs

**Critical for**: SOTA submissions under scrutiny

### Check 1: Deterministic Training

**Run same seed twice**:
```bash
# First run
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee run1.log

# Second run (same seed)
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee run2.log

# Compare
diff <(grep "train_loss:" run1.log) <(grep "train_loss:" run2.log)
# Should be identical or very close (< 1e-5 difference)
```

**If non-deterministic**:
- Check RNG seeding (torch, numpy, random)
- Verify no undefined behavior in custom code
- Check for timing-dependent logic
- Verify torch.backends.cudnn.deterministic settings

---

### Check 2: Environment Verification

**Document exact environment**:
```bash
# Create environment snapshot
python3 << 'EOF' > environment.txt
import torch
import sys
import subprocess

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

try:
    import flash_attn
    print("FlashAttention:", flash_attn.__version__)
except:
    print("FlashAttention: not installed")

result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
                        '--format=csv,noheader'],
                       capture_output=True, text=True)
print("Driver:", result.stdout.strip())
EOF

cat environment.txt
```

**Include in submission** if using non-standard setup.

---

### Check 3: Artifact Size Verification

**Verify reported size matches reality**:
```bash
# Find the exported artifact
find . -name "*.ptz" -o -name "*.pt" | while read f; do
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
    echo "$f: $size bytes"
done

# Verify against submission.json
python3 << 'EOF'
import json
with open('submission.json') as f:
    meta = json.load(f)
    claimed_size = meta['bytes_total']
    print(f"Claimed size: {claimed_size:,} bytes")
    print(f"Under 16MB: {'✓' if claimed_size < 16_000_000 else '✗ OVER!'}")
EOF

# Check code size contribution
wc -c train_gpt.py
```

---

## Quantization Verification

**Goal**: Verify quantization doesn't break model

### Test 1: Quantization Gap Measurement

**Acceptable gap**: 0.005-0.03 BPB

```python
# In your eval code, measure pre-quant and post-quant
print(f"Pre-quant val_bpb: {pre_quant_bpb:.4f}")
print(f"Post-quant val_bpb: {post_quant_bpb:.4f}")
print(f"Quantization gap: {post_quant_bpb - pre_quant_bpb:.4f}")

# Alert if gap too large
if post_quant_bpb - pre_quant_bpb > 0.03:
    print("⚠️  WARNING: Large quantization gap! Consider:")
    print("  - Enabling Late QAT")
    print("  - Using GPTQ-lite")
    print("  - Checking quantization implementation for bugs")
```

**If gap > 0.03**:
- Enable Late QAT
- Use GPTQ-lite clip search
- Check STE implementation
- Verify per-row scales computed correctly

---

### Test 2: Quantization Roundtrip

**Verify dequantization inverts quantization**:
```python
def test_quantization_roundtrip(model):
    """Verify quant->dequant produces reasonable model."""
    # Save original weights
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Quantize
    quantized = quantize_state_dict(orig_state)

    # Dequantize
    restored = dequantize_state_dict(quantized)

    # Measure error
    for name in orig_state:
        orig = orig_state[name]
        rest = restored[name]

        if orig.dtype in [torch.float32, torch.bfloat16]:
            mse = ((orig - rest) ** 2).mean().item()
            max_err = (orig - rest).abs().max().item()
            print(f"{name:40s} MSE: {mse:.6f}, Max: {max_err:.4f}")

test_quantization_roundtrip(model)
```

**Expected errors**:
- Int8: MSE < 0.001, Max < 0.1
- Int6: MSE < 0.01, Max < 0.5
- Int5: MSE < 0.05, Max < 1.0

---

### Test 3: Compression Ratio Check

**Verify compression is effective**:
```bash
# Check compression ratios
python3 << 'EOF'
import torch
import zlib

# Load quantized model
state = torch.load('model.int6.ptz')

for name, tensor in state['quantized'].items():
    raw_bytes = tensor.numel()
    compressed = len(zlib.compress(tensor.numpy().tobytes(), level=9))
    ratio = raw_bytes / compressed
    print(f"{name:40s} Ratio: {ratio:.2f}x")
EOF
```

**Typical ratios**:
- Int8 matrices: 1.2-1.4x
- Int6 matrices: 1.5-1.9x
- Int5 matrices: 1.8-2.2x
- Ternary: 2.5-3.5x with base-3 packing

**If ratio < 1.2x**: Weights might not be well-distributed, check:
- Weight initialization
- Weight decay application
- Training convergence

---

## Evaluation Verification

### Test 1: BPB Calculation Correctness

**Critical if you modified tokenizer!**

```python
def verify_bpb_calculation(tokenizer_path, val_tokens):
    """Verify bits-per-byte calculation is correct."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # Decode tokens back to text
    token_ids = val_tokens[:1000].tolist()  # Sample
    text = sp.decode(token_ids)
    text_bytes = len(text.encode('utf-8'))

    # Count bytes using your BPB logic
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, vocab_size, device)

    computed_bytes = 0
    for i in range(1, len(token_ids)):
        prev_id = token_ids[i-1]
        cur_id = token_ids[i]
        computed_bytes += base_bytes_lut[cur_id].item()
        if has_leading_space_lut[cur_id] and not is_boundary_token_lut[prev_id]:
            computed_bytes += 1  # Leading space byte

    # Should match
    print(f"Text bytes (ground truth): {text_bytes}")
    print(f"Computed bytes (your logic): {computed_bytes}")
    print(f"Match: {'✓' if abs(text_bytes - computed_bytes) <= 1 else '✗ BUG!'}")

    return abs(text_bytes - computed_bytes) <= 1
```

**If mismatch**: Your BPB calculation has a bug. This will disqualify submission.

---

### Test 2: Sliding Window Consistency

**Verify sliding window scores same tokens correctly**:
```python
def test_sliding_window_consistency(model, val_tokens):
    """Verify each token scored exactly once with correct context."""
    scored_positions = set()
    seq_len = 1024
    stride = 64

    for start in range(0, len(val_tokens) - seq_len, stride):
        # Positions scored in this window
        scored_start = start + (seq_len - stride)
        scored_end = start + seq_len
        for pos in range(scored_start, scored_end):
            if pos in scored_positions:
                print(f"⚠️  Position {pos} scored multiple times!")
            scored_positions.add(pos)

    total_tokens = len(val_tokens) - 1
    expected_scored = ((total_tokens - seq_len) // stride) * stride + \
                     min(stride, total_tokens - seq_len - \
                         ((total_tokens - seq_len) // stride) * stride)

    print(f"Tokens scored: {len(scored_positions)}")
    print(f"Expected: ~{expected_scored}")
    print(f"Match: {'✓' if abs(len(scored_positions) - expected_scored) < 100 else '✗'}")
```

---

### Test 3: TTT Legality Check

**Verify TTT never trains before scoring**:
```python
def verify_ttt_legal(ttt_eval_function):
    """
    Verify TTT uses inference_mode for scoring.
    This is a code inspection check - grep for patterns.
    """
    import inspect
    source = inspect.getsource(ttt_eval_function)

    # Check for score-first pattern
    has_inference_mode = 'torch.inference_mode()' in source
    has_score_before_train = source.find('inference_mode') < source.find('backward()')

    print(f"Uses inference_mode: {'✓' if has_inference_mode else '✗ ILLEGAL!'}")
    print(f"Scores before training: {'✓' if has_score_before_train else '✗ ILLEGAL!'}")

    if not (has_inference_mode and has_score_before_train):
        print("\n⚠️  TTT IMPLEMENTATION ILLEGAL!")
        print("Must score under inference_mode BEFORE training on chunk!")

    return has_inference_mode and has_score_before_train
```

**Manual inspection**:
- Verify `torch.inference_mode()` context during scoring
- Verify no `backward()` or `optimizer.step()` during scoring
- Verify last chunk scored but never trained on
- Verify no cross-document leakage (for LoRA TTT)

---

## Performance Profiling

### Profile 1: Step Time Breakdown

**Goal**: Identify bottlenecks

```python
import time
import torch.cuda.nvtx as nvtx

# In training loop
times = {'forward': [], 'backward': [], 'optimizer': []}

for step in range(100, 200):  # Profile 100 steps
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    nvtx.range_push("forward")
    loss = model(x, y)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    nvtx.range_pop()

    nvtx.range_push("backward")
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    nvtx.range_pop()

    nvtx.range_push("optimizer")
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    nvtx.range_pop()

    times['forward'].append((t1 - t0) * 1000)
    times['backward'].append((t2 - t1) * 1000)
    times['optimizer'].append((t3 - t2) * 1000)

# Print summary
for phase, timings in times.items():
    print(f"{phase:12s}: {np.mean(timings):6.2f}ms ± {np.std(timings):5.2f}ms")
```

**Typical breakdown** (8xH100, 512d, seq2048):
- Forward: 25-30ms (30-35%)
- Backward: 35-40ms (40-45%)
- Optimizer: 20-25ms (25-30%)
- Total: 82-92ms

**If slower than expected**:
- Forward slow: Check attention implementation, use FlashAttention
- Backward slow: Check custom autograd functions, verify torch.compile
- Optimizer slow: Check Muon Newton-Schulz steps (try 3 instead of 5)

---

### Profile 2: Memory Usage

**Goal**: Verify not wasting memory

```python
def log_memory_stats(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# In training
log_memory_stats("After model init")
log_memory_stats("After first forward")
log_memory_stats("After first backward")
log_memory_stats("After warmup")
```

**Expected** (8xH100, 11L/512d/seq2048):
- After init: ~2-4GB
- After forward: ~15-20GB
- After backward: ~20-25GB
- Peak: ~20-25GB per GPU

**If using >50GB per GPU**:
- Memory leak (check for retained graphs)
- Unnecessary tensor copies
- EMA model not managed properly

---

## Validation Against Known Baselines

**Goal**: Verify your implementation produces expected results for known configs

### Baseline Reproduction Test

**Run baseline config and verify**:
```bash
# Standard baseline (from README)
RUN_ID=baseline_check \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
MLP_MULT=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Expected: val_bpb ~1.2244
# If you get significantly different, investigate!
```

---

### Technique Ablation Test

**Verify each technique works independently**:
```bash
# Base
BASE_CONFIG="NUM_LAYERS=9 MODEL_DIM=512 ITERATIONS=1000"

# Test each technique individually
echo "Testing baseline..."
$BASE_CONFIG python3 train_gpt.py > log_base.txt

echo "Testing with XSA..."
$BASE_CONFIG XSA_LAST_N=4 python3 train_gpt.py > log_xsa.txt

echo "Testing with LeakyReLU²..."
$BASE_CONFIG ACTIVATION=leaky_relu2 python3 train_gpt.py > log_leaky.txt

echo "Testing with EMA..."
$BASE_CONFIG EMA_ENABLED=1 python3 train_gpt.py > log_ema.txt

# Compare final losses (each should be ≤ baseline)
for log in log_*.txt; do
    loss=$(grep "train_loss:" "$log" | tail -1 | awk '{print $NF}')
    echo "$log: $loss"
done
```

---

## Common Issues and Debugging

### Issue 1: Training Loss Not Decreasing

**Symptoms**: Loss stays high or increases

**Debug checklist**:
- [ ] Check learning rates (might be too low)
- [ ] Verify optimizer actually stepping
- [ ] Check for frozen parameters
- [ ] Verify gradients flowing (print grad norms)
- [ ] Check batch size (might be too small)

**Debug code**:
```python
# After backward, before optimizer step
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5
print(f"Step {step}, grad_norm: {total_norm:.4f}")

# If grad_norm is 0 or very small (<0.01), gradients not flowing!
```

---

### Issue 2: OOM (Out of Memory)

**Symptoms**: CUDA out of memory error

**Solutions in order of preference**:

1. **Reduce batch size**:
```bash
TRAIN_BATCH_TOKENS=262144  # Half of 524k
```

2. **Reduce sequence length**:
```bash
TRAIN_SEQ_LEN=1024  # Instead of 2048
```

3. **Reduce model size**:
```bash
NUM_LAYERS=9  # Instead of 11
MLP_MULT=2    # Instead of 3
```

4. **Enable gradient accumulation** (in code):
```python
grad_accum_steps = 2
optimizer.zero_grad()
for i in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps
    loss.backward()
optimizer.step()
```

5. **Clear cache after warmup**:
```python
if step == warmup_steps:
    torch.cuda.empty_cache()
```

---

### Issue 3: Slower Than Expected Steps

**Symptoms**: >95ms/step on 8xH100 (expect 82-92ms)

**Debug checklist**:
- [ ] torch.compile enabled?
- [ ] FlashAttention being used?
- [ ] Unnecessary synchronization points?
- [ ] Too many print statements?
- [ ] Large custom operations not optimized?

**Profile to find bottleneck**:
```bash
# Use PyTorch profiler
python3 << 'EOF'
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for step in range(10):
        loss = model(x, y)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
EOF
```

**Common fixes**:
- Remove print statements from inner loop
- Use FlashAttention 3 instead of standard attention
- Reduce Newton-Schulz steps (5 → 3)
- Verify no CPU-GPU sync in hot path

---

### Issue 4: Large Artifact Size

**Symptoms**: Artifact > 16,000,000 bytes

**Solutions**:

1. **Check compression**:
```python
# Try different compression levels
for level in [9, 15, 22]:
    compressed = zlib.compress(bytes, level=level)
    print(f"Level {level}: {len(compressed):,} bytes")

# Try lzma instead of zstd
import lzma
compressed_lzma = lzma.compress(bytes, preset=6)
print(f"LZMA: {len(compressed_lzma):,} bytes")
```

2. **More aggressive quantization**:
```bash
# Try int5 instead of int6 for MLPs
# Or ternary for extreme compression
```

3. **Reduce model size**:
```bash
NUM_LAYERS=10         # Instead of 11
MLP_MULT=2.5          # Instead of 3
BIGRAM_VOCAB_SIZE=1536  # Instead of 2048
```

4. **Remove unnecessary parameters**:
- Check for unused modules
- Remove debug parameters
- Slim control tensors

5. **Optimize code size**:
```bash
# Check code contribution
wc -c train_gpt.py
# Should be 45-70KB. If >100KB, code too large!

# Remove comments and docstrings for submission (keep development version)
```

---

### Issue 5: TTT Makes Results Worse

**Symptoms**: Post-TTT BPB > Pre-TTT BPB

**Possible causes**:
1. **Illegal TTT**: Training before scoring
2. **Wrong LR**: Too high (divergence) or too low (no effect)
3. **Wrong optimizer**: Adam might need different hyperparams than SGD
4. **Overfitting**: Too many epochs on small chunks
5. **Chunk size**: Too small or too large

**Debug**:
```python
# Log per-chunk improvement
for chunk_idx, chunk in enumerate(chunks):
    pre_ttt_loss = score_chunk(model, chunk)
    adapt_on_chunk(model, chunk)
    post_ttt_loss = score_chunk(model, chunk)  # Shouldn't do this in legal TTT!

    improvement = pre_ttt_loss - post_ttt_loss
    print(f"Chunk {chunk_idx}: {improvement:+.4f} (positive = improvement)")
```

**Tune**:
- Try LR: 0.001, 0.002, 0.005
- Try epochs: 1, 3, 5, 10, 20, 30
- Try momentum: 0.0, 0.9, 0.95
- Try chunk size: 16K, 32K, 64K

---

## Reproducibility Checklist

Before claiming a result is reproducible:

- [ ] **Fixed seeds**: All RNGs seeded deterministically
  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  ```

- [ ] **No randomness in eval**: Validation should be fully deterministic
  ```python
  model.eval()
  torch.cuda.manual_seed(seed)  # Reset before eval
  ```

- [ ] **Environment documented**: Package versions in requirements.txt
  ```bash
  pip freeze > requirements.txt
  ```

- [ ] **No timing-dependent behavior**: Logic shouldn't depend on wallclock timing
  ```python
  # Bad: Different behavior based on time
  if time.time() % 2 == 0:
      do_something()

  # Good: Deterministic based on step count
  if step % 100 == 0:
      do_something()
  ```

- [ ] **No external randomness**: No calls to random APIs, no reading uncontrolled files

- [ ] **NCCL determinism** (for distributed):
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  # In Python:
  torch.use_deterministic_algorithms(True, warn_only=True)
  ```

---

## Validation Log Analysis

**Extract key metrics from logs**:
```bash
# Create summary from training log
python3 << 'EOF'
import re

with open('train_seed1337.log') as f:
    log = f.read()

# Extract metrics
def extract(pattern, log):
    match = re.search(pattern, log)
    return match.group(1) if match else "Not found"

print("="*60)
print("TRAINING SUMMARY")
print("="*60)

steps = extract(r'step[:\s]+(\d+).*final', log)
step_avg = extract(r'step_avg[:\s]+([\d.]+)ms', log)
val_loss = extract(r'final.*val_loss[:\s]+([\d.]+)', log)
val_bpb = extract(r'final.*val_bpb[:\s]+([\d.]+)', log)
artifact = extract(r'bytes_total[:\s]+(\d+)', log)

print(f"Steps completed: {steps}")
print(f"Step average: {step_avg}ms")
print(f"Final val_loss: {val_loss}")
print(f"Final val_bpb: {val_bpb}")
print(f"Artifact size: {artifact} bytes")

# Compute steps per second
if step_avg != "Not found":
    sps = 1000.0 / float(step_avg)
    print(f"Steps/second: {sps:.2f}")

# Check under limits
if artifact != "Not found":
    under_limit = int(artifact) < 16_000_000
    print(f"Under 16MB: {'✓' if under_limit else '✗ OVER LIMIT!'}")

print("="*60)
EOF
```

---

## Timing Budget Analysis

**Goal**: Ensure training + eval under respective limits

**Training budget**: 600s (10 minutes)
**Eval budget**: 600s (10 minutes, separate from training)

```python
# Parse timing from log
def check_timing_budget(log_file):
    with open(log_file) as f:
        log = f.read()

    # Training time
    train_match = re.search(r'train.*time[:\s]+([\d.]+)s', log)
    train_time = float(train_match.group(1)) if train_match else None

    # Eval components
    quant_match = re.search(r'quantization.*time[:\s]+([\d.]+)s', log)
    sliding_match = re.search(r'sliding.*time[:\s]+([\d.]+)s', log)
    ttt_match = re.search(r'TTT.*time[:\s]+([\d.]+)s', log)

    quant_time = float(quant_match.group(1)) if quant_match else 0
    sliding_time = float(sliding_match.group(1)) if sliding_match else 0
    ttt_time = float(ttt_match.group(1)) if ttt_match else 0

    eval_time = quant_time + sliding_time + ttt_time

    print(f"Training time: {train_time:.1f}s / 600s {'✓' if train_time <= 600 else '✗'}")
    print(f"Eval time: {eval_time:.1f}s / 600s {'✓' if eval_time <= 600 else '✗'}")
    print(f"  - Quantization: {quant_time:.1f}s")
    print(f"  - Sliding eval: {sliding_time:.1f}s")
    print(f"  - TTT: {ttt_time:.1f}s")

check_timing_budget('train_seed1337.log')
```

**Typical eval time breakdown**:
- Quantization: 5-20s
- Standard eval: 15-20s
- Sliding window (stride=64): 90-120s
- TTT (3 epochs): 400-450s
- TTT (30 epochs): ~700s (over budget!)

**If over eval budget**:
- Reduce TTT epochs
- Increase sliding stride (64 → 128)
- Optimize quantization (cache calculations)
- Skip redundant eval modes

---

## Continuous Integration Testing

**For iterative development**:

```bash
#!/bin/bash
# test_pipeline.sh - Quick CI test

set -e  # Exit on error

echo "Stage 1: Syntax check"
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"

echo "Stage 2: 10-step smoke test"
ITERATIONS=10 VAL_LOSS_EVERY=0 python3 train_gpt.py

echo "Stage 3: 100-step convergence test"
ITERATIONS=100 VAL_LOSS_EVERY=50 python3 train_gpt.py > test_100.log

# Extract final loss
final_loss=$(grep "train_loss:" test_100.log | tail -1 | awk '{print $NF}')
echo "Final loss after 100 steps: $final_loss"

# Should drop from ~8 to ~3-4
if (( $(echo "$final_loss > 5" | bc -l) )); then
    echo "✗ Loss not decreasing properly!"
    exit 1
fi

echo "✓ All tests passed"
```

---

## Pre-Submission Final Checklist

Run through this checklist before submitting PR:

### Files
- [ ] README.md present and complete
- [ ] submission.json present with all required fields
- [ ] train_gpt.py (or similar) present and runnable
- [ ] 3+ training logs for records (1+ for non-records)
- [ ] requirements.txt if using custom packages

### Correctness
- [ ] Python syntax valid
- [ ] JSON syntax valid
- [ ] Script runs from records folder
- [ ] All paths relative or configurable
- [ ] Random seeding implemented

### Performance
- [ ] Completes training in ≤600s on 8xH100
- [ ] Achieves competitive val_bpb
- [ ] Artifact ≤16,000,000 bytes
- [ ] Eval completes in ≤600s (if applicable)

### Statistical
- [ ] 3+ seeds run (for records)
- [ ] Standard deviation < 0.002 (acceptable variance)
- [ ] Mean improvement ≥0.005 over SOTA (for records)
- [ ] p-value < 0.01 (for records)

### Documentation
- [ ] Clear explanation of techniques
- [ ] Run command documented
- [ ] Credits to prior work
- [ ] Ablations or justification
- [ ] Honest assessment of results

### Legal
- [ ] No training on validation before scoring
- [ ] TTT is score-first if used
- [ ] No external compute violations
- [ ] No validation data access during training (except paid for)
- [ ] No cheating on test loss

---

## Post-Submission Monitoring

### Watch for Issues Raised

**If reproducibility issue raised**:
1. Verify you can reproduce on fresh environment
2. Provide detailed environment info
3. Share additional logs if helpful
4. Fix bugs and update if needed (before merge)

**If performance questioned**:
1. Provide profiling data
2. Share intermediate checkpoints
3. Document any non-obvious optimizations

---

## Advanced Verification: Numerical Stability

**For novel quantization schemes or optimizers**:

```python
def test_numerical_stability(model, test_batches=100):
    """Check for gradients exploding/vanishing."""
    grad_norms = []

    for _ in range(test_batches):
        x, y = get_batch()
        loss = model(x, y)
        loss.backward()

        # Measure gradient norms per layer
        norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()

        grad_norms.append(norms)
        optimizer.zero_grad()

    # Analyze
    for name in grad_norms[0].keys():
        layer_norms = [step[name] for step in grad_norms]
        mean = np.mean(layer_norms)
        std = np.std(layer_norms)
        print(f"{name:40s} Mean: {mean:.4f}, Std: {std:.4f}")

        if mean < 1e-6:
            print(f"  ⚠️  Vanishing gradients in {name}!")
        if mean > 100:
            print(f"  ⚠️  Exploding gradients in {name}!")
```

---

## Verification for Specific Techniques

### Verifying Sliding Window Correctness

```bash
# Run with and without sliding window
BASE="ITERATIONS=5000"

# Standard eval
$BASE EVAL_STRIDE=0 python3 train_gpt.py > log_standard.log

# Sliding eval
$BASE EVAL_STRIDE=64 python3 train_gpt.py > log_sliding.log

# Compare
echo "Standard eval:"
grep "val_bpb:" log_standard.log | tail -1

echo "Sliding eval:"
grep "val_bpb:" log_sliding.log | tail -1

# Sliding should be ~0.03 BPB better
```

---

### Verifying EMA Correctness

**Test that EMA produces different (better) results than live model**:
```python
# After training with EMA
live_val_bpb = eval_val(args, model, ...)
ema_val_bpb = eval_val(args, ema_model, ...)

print(f"Live model: {live_val_bpb:.4f}")
print(f"EMA model: {ema_val_bpb:.4f}")
print(f"Improvement: {live_val_bpb - ema_val_bpb:.4f}")

# EMA should be better (0.003-0.010 typical)
assert ema_val_bpb < live_val_bpb, "EMA should improve results!"
```

---

### Verifying Quantization Doesn't Break Model

```python
# Compare pre-quant and post-quant predictions
def test_quantization_effect(model, quantized_model, test_inputs):
    model.eval()
    quantized_model.eval()

    with torch.no_grad():
        orig_logits = model(test_inputs)
        quant_logits = quantized_model(test_inputs)

        # Measure prediction difference
        prob_diff = (F.softmax(orig_logits, dim=-1) -
                    F.softmax(quant_logits, dim=-1)).abs().mean()

        print(f"Average probability difference: {prob_diff:.6f}")
        # Should be < 0.05 typically
```

---

## Quick Reference: Expected Metrics

**Baseline (9L/512d/1024vocab)**:
- Steps: 13,780 (8xH100, 600s)
- Step time: 43ms
- Val_bpb: 1.2244
- Artifact: 15.86MB

**Current SOTA** (as of March 2026):
- Steps: 7,179 (8xH100, 600s)
- Step time: 83ms
- Val_bpb: 1.1194
- Artifact: 15.98MB
- Techniques: 11L, LeakyReLU², Legal TTT, Parallel Muon, XSA, Partial RoPE, EMA, GPTQ-lite

**Your results should**:
- Match step time ±10ms for similar architecture
- Beat or match comparable techniques
- Stay under 16MB
- Show consistent improvement over baseline techniques

---

## Related Skills

- Use `/parameter-golf` for challenge overview
- Use `/parameter-golf-techniques` for technique catalog
- Use `/parameter-golf-implement` for implementation patterns
- Use `/parameter-golf-submit` for submission process
