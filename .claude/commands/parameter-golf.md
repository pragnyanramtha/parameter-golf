---
name: parameter-golf
description: Comprehensive guide for OpenAI's Parameter Golf challenge - train the best LM under 16MB
---

# Parameter Golf Skill

You are helping with OpenAI's **Parameter Golf Challenge**: train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (bits per byte, or BPB).

## Challenge Overview

**Objective**: Optimize L(N) - lowest validation loss given fixed parameters (N), unconstrained by data, compute (within 10-min limit), steps, or architecture.

**Constraints**:
- **16MB artifact limit**: Code bytes + compressed model bytes ≤ 16,000,000 bytes (decimal, not MiB)
- **10 minute training**: Must train in ≤600 seconds on 8xH100 SXM
- **10 minute evaluation**: Evaluation must complete in ≤600 seconds (additional to training time)
- **Evaluation metric**: Bits per byte (BPB) on FineWeb validation set (tokenizer-agnostic)

**Current SOTA**: 1.1194 BPB (as of March 2026)

## Repository Structure

```
parameter-golf/
├── train_gpt.py           # Baseline training script (for newcomers, ~1500 line limit)
├── train_gpt_mlx.py       # MLX version for Apple Silicon local iteration
├── data/
│   ├── cached_challenge_fineweb.py  # Dataset downloader
│   ├── tokenizer_specs.json
│   └── README.md
├── records/
│   ├── track_10min_16mb/      # Competition submissions
│   └── track_non_record_16mb/ # Unlimited compute/interesting approaches
└── requirements.txt
```

## Key Concepts

### Architecture Components

1. **Model Size**: Typically 9-11 layers, 512-768 dim, 8 heads, 4 KV heads (GQA)
2. **Vocabulary**: Usually 1024-8192 tokens (smaller vocab = fewer embedding params)
3. **Tied Embeddings**: Input and output embeddings share weights (saves ~50% embedding params)
4. **Sequence Length**: 1024-4096 tokens (longer context = better eval, slower training)

### Critical Techniques (Ordered by Impact)

#### Evaluation Techniques (Largest Gains)
1. **Sliding Window Eval** (~0.03 BPB): Score tokens with full context using overlapping windows (stride=64 typical)
2. **Test-Time Training (TTT)** (~0.02-0.04 BPB): Adapt model during evaluation on already-scored tokens
   - **Legal TTT**: Score-first protocol - score tokens under `torch.inference_mode()`, then train on scored chunks
   - Typical: 3-30 epochs, SGD with momentum 0.9, LR 0.002-0.005, 32K token chunks
3. **N-gram Eval Cache** (~0.02 BPB): Interpolate neural predictions with n-gram statistics from scored tokens

#### Architecture Innovations
1. **LeakyReLU² Activation** (~0.003 BPB): `F.leaky_relu(x, 0.5).square()` preserves negative gradients, eliminates dead neurons
2. **Exclusive Self-Attention (XSA)** (~0.005 BPB): Subtract component aligned with token's own value vector
3. **Partial RoPE** (~0.002 BPB): Apply rotary embeddings to only 16/64 head dims (25%), rest attend position-invariant
4. **U-Net Skip Connections**: Encoder-decoder with learned skip weights connecting corresponding layers
5. **SmearGate** (~0.01 BPB): Learned gate blending current token embedding with previous token
6. **BigramHash Embedding** (~0.01 BPB): Hash table (2048-10240 buckets) mapping token pairs to embeddings
7. **Layer Normalization Scale**: Scale RMSNorm outputs by `1/sqrt(layer_idx+1)` to damp deep layers
8. **Value Embedding (VE)**: Shared value embeddings across layers with per-layer learned scales

#### Quantization & Compression
1. **Int6 Quantization** (~50% size reduction): Per-row quantization to [-31, 31] with per-row scales
2. **GPTQ-lite**: Try 5-15 clip percentiles per row, pick minimum reconstruction MSE
3. **Late QAT (Quantization-Aware Training)**: Enable STE fake-quantization in final 10-15% of training
4. **Mixed Precision**: Int5 for MLPs, Int6 for attention, FP16 for embeddings/control tensors
5. **Ternary/BitNet** (b1.58): Extreme quantization {-1, 0, +1} enables 3x more parameters (~65M)
6. **Compression**: zstd-22 (standard) or lzma (2-5% tighter) after quantization

#### Optimizer & Training
1. **Muon Optimizer**: Orthogonalizes gradients via Newton-Schulz iteration (~5 steps)
   - Typical: momentum 0.95-0.99, warmup from 0.85-0.92 over 500-1500 steps
   - **Parameter Banking**: Batched orthogonalization for 4x speedup
2. **Weight Decay**: 0.04 typical, applied to both Muon (matrices) and AdamW (embeddings/scalars)
3. **EMA (Exponential Moving Average)**: Shadow model with decay 0.997-0.9985, smoother than SWA
4. **SWA (Stochastic Weight Averaging)**: Collect checkpoints during warmdown (every 30-50 steps)
5. **Warmdown**: Linear LR decay over final 3000-3500 iterations (critical for convergence)
6. **Gradient Clipping**: 0.3 typical
7. **Orthogonal Initialization**: All weight matrices initialized orthogonally for faster convergence
8. **MLP Expansion**: 3-4x (vs baseline 2x) funded by quantization savings

#### Advanced Architecture
1. **Depth Recurrence**: Loop 3-6 unique layers 2-3x for 12-18 effective layers
2. **SwiGLU**: Gated activation `silu(gate) * up` (used in LLaMA/Mistral)
3. **Differential Attention**: Novel attention mechanism (slow without custom kernels)
4. **Mixture of Experts (MoE)**: Soft/dense gating preferred over sparse (routing collapse)

### Key Implementation Details

#### Data Loading
- **TokenStream**: Sequential shard reading with wraparound
- **DistributedTokenLoader**: Per-rank disjoint spans from shared stream
- **Validation**: Fixed 50k-document fineweb_val split

#### Evaluation Modes
1. **Standard**: Non-overlapping chunks (baseline)
2. **Sliding Window**: Overlapping windows with stride (64 typical)
3. **Test-Time Training**: Adapt on scored chunks during eval

#### Compression Pipeline
```
FP32/BF16 trained model
  → Quantize (int5/int6/int8/ternary)
  → Pack into int8 containers
  → Compress (zstd-22 or lzma)
  → Export artifact
```

## Common Hyperparameters (SOTA as of March 2026)

```bash
# Model
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
VOCAB_SIZE=1024
TRAIN_SEQ_LEN=2048
TIE_EMBEDDINGS=1

# Training
ITERATIONS=9000
TRAIN_BATCH_TOKENS=786432
WARMUP_STEPS=20
WARMDOWN_ITERS=3500
MAX_WALLCLOCK_SECONDS=600

# Optimizer (Muon for matrices)
MATRIX_LR=0.025
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_WD=0.04

# Optimizer (AdamW for embeddings/scalars)
TIED_EMBED_LR=0.035
SCALAR_LR=0.025
ADAM_WD=0.04

# Architecture features
BIGRAM_VOCAB_SIZE=2048  # BigramHash buckets
XSA_LAST_N=4            # XSA on last 4 layers
ROPE_DIMS=16            # Partial RoPE (16/64 dims)
LN_SCALE=1              # Enable 1/sqrt(i+1) scaling

# EMA/SWA
EMA_ENABLED=1
EMA_DECAY=0.997
SWA_ENABLED=1
SWA_EVERY=50

# Quantization
LATE_QAT=1
LATE_QAT_THRESHOLD=0.15

# Evaluation
EVAL_STRIDE=64          # Sliding window stride
```

## File Requirements for Submissions

Every submission must include in `records/track_10min_16mb/YYYY-MM-DD_DescriptiveName/`:

1. **README.md**: Detailed explanation of techniques, results, ablations
2. **submission.json**:
   ```json
   {
     "name": "Descriptive Name",
     "val_bpb": 1.1234,
     "bytes_total": 15900000,
     "blurb": "Brief summary of techniques",
     "author": "Your Name",
     "github_id": "yourusername",
     "date": "YYYY-MM-DD"
   }
   ```
3. **train_gpt.py**: Complete training script (must run from within records folder)
4. **train.log** (or train_seedXXX.log): Training logs proving statistical significance
   - SOTA records require 3+ seeds showing p < 0.01 for 0.005-nat improvement

## Typical Workflow

### Local Iteration (Mac with Apple Silicon)
```bash
# Clone and setup
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download dataset (smaller subset for testing)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Run MLX training
RUN_ID=mlx_test ITERATIONS=200 python3 train_gpt_mlx.py
```

### Remote Training (8xH100 on RunPod)
```bash
# On RunPod pod with Parameter Golf template
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download full dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Launch training
RUN_ID=my_experiment \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Important Rules

### DO
- Use any package/library (include in requirements.txt)
- Train on validation set (bits must fit in 16MB budget)
- Evaluate at any sequence length
- Use test-time training on already-scored tokens
- Tune hyperparameters offline
- Submit non-record runs for interesting/negative results

### DON'T
- Access validation data during training (except if paid for in 16MB limit)
- Train on validation set before evaluation (TTT only on scored tokens)
- Take >10 min for training on 8xH100
- Take >10 min for evaluation
- Submit artifacts >16,000,000 bytes
- Cheat on test loss or validation access

### Submission Acceptance Criteria

**SOTA Records**:
1. Beat existing SOTA by ≥0.005 nats with p < 0.01 statistical significance
2. Prove val_bpb is correctly calculated (especially if tokenizer changed)
3. Reproducibly run in <10 min on 8xH100s

**Non-Record Submissions**:
- Unique/interesting approaches
- In-progress or unoptimized solutions
- Interesting negative results
- Unlimited compute track (note in README)

## Key Research Papers Referenced

- Neural Scaling Laws (Kaplan et al., 2020): arXiv:2001.08361
- Muon Optimizer: https://kellerjordan.github.io/posts/muon/
- BitNet b1.58 (ternary quantization)
- Differential Attention (ICLR 2025): arXiv:2410.05258
- Value Residual Learning/ResFormer (ACL 2025): arXiv:2410.17897
- GPTQ quantization techniques
- YaRN positional embeddings
- FlashAttention 3 (Hopper-optimized)

## Common Pitfalls

1. **torch.compile issues**:
   - Class attributes get constant-folded at first trace (Late QAT bug)
   - `torch.autograd.Function` bypasses Inductor (2-3x slower backward)
   - Flipping flags mid-training causes recompilation → OOM

2. **Quantization gaps**:
   - STE (Straight-Through Estimator) during QAT must match actual quantizer
   - Control tensors (scales, gates) should stay FP16/FP32
   - Post-training quantization can lose 0.01-0.03 BPB without QAT

3. **Evaluation gotchas**:
   - Sliding window dramatically improves BPB (~0.03) but adds eval time
   - TTT must be score-first (legal) - never train before scoring
   - Document boundaries matter for TTT

4. **Memory constraints**:
   - Peak memory ~20GB per H100 for 11L/512d/seq2048
   - FlashAttention essential for long sequences
   - Gradient accumulation if OOM

## Environment & Hardware

- **Baseline**: RunPod 8xH100 SXM 80GB HBM3
- **Software**: PyTorch 2.9.1+cu128, FlashAttention 3, CUDA 12.8
- **Template**: [RunPod Parameter Golf Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- **Compute Credits**: OpenAI offers $1M in credits - apply at https://openai.com/index/parameter-golf/#credit-form

## Sub-Skills

Use these specialized sub-skills for specific tasks:
- `/parameter-golf-techniques` - Deep dive into all techniques with implementation examples
- `/parameter-golf-submit` - Step-by-step submission process and validation
- `/parameter-golf-implement` - How to implement specific techniques in train_gpt.py
- `/parameter-golf-verify` - Testing, verification, and reproducibility checks
- `/parameter-golf-debug` - Common issues and debugging strategies

## Support

- **Discord**: OpenAI Discord server → #parameter-golf-discussions, #parameter-golf-announcements
- **Issues**: Raise reproducibility issues on GitHub
- **Verification**: Top leaderboard entries independently verified by OpenAI
