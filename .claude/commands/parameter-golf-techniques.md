---
name: parameter-golf-techniques
description: Comprehensive catalog of all Parameter Golf techniques with implementations
---

# Parameter Golf Techniques Catalog

This skill provides detailed information on all techniques used in Parameter Golf submissions, organized by category with implementation guidance and expected impact.

## Evaluation Techniques (Highest Impact per Cost)

### 1. Sliding Window Evaluation (~0.03 BPB improvement)

**Impact**: ~0.03 BPB | **Cost**: +50-120s eval time | **Complexity**: Low

**Problem**: Standard evaluation chops validation into non-overlapping chunks. First token in each chunk has zero context (average ~512 tokens/token).

**Solution**: Overlapping windows with stride < seq_len. Only score tokens with near-full context.

**Implementation**:
```python
# Key parameters
EVAL_STRIDE=64          # Advance 64 tokens per window
TRAIN_SEQ_LEN=1024      # 1024-token window
# Result: Each token scored with 960+ tokens of context

# Core loop (simplified)
for start in range(0, total_tokens - seq_len, stride):
    window = tokens[start:start + seq_len + 1]
    x, y = window[:-1], window[1:]
    # Score only the rightmost 'stride' tokens (they have most context)
    logits = model(x)
    loss += cross_entropy(logits[-stride:], y[-stride:])
```

**Tuning**: stride=32 denser but slower, stride=64 standard, stride=128 faster but less accurate

**First appeared**: PR #19 (Matthew Li) - 1.1925 BPB

---

### 2. Test-Time Training (TTT) (~0.02-0.04 BPB improvement)

**Impact**: 0.02-0.04 BPB | **Cost**: +400-500s eval time | **Complexity**: High

**Concept**: Adapt the model during evaluation on already-scored validation tokens. Think of it as "online learning during the test."

**Legal Score-First Protocol** (Required for Competition):
1. Split validation into chunks (32K tokens typical)
2. **For each chunk**:
   - **SCORE FIRST**: Eval under `torch.inference_mode()` (hard guarantee no weight mutation)
   - **THEN TRAIN**: Adapt on the already-scored chunk
3. Last chunk scored but never trained on
4. Each chunk scored by model adapted only on previous chunks

**Implementation**:
```python
# Chunk and score-first loop
chunk_size = 32768
chunks = split_tokens_into_chunks(val_tokens, chunk_size)

adapted_model = copy.deepcopy(model)
optimizer = torch.optim.SGD(adapted_model.parameters(),
                            lr=0.002, momentum=0.9)

for i, chunk in enumerate(chunks):
    # SCORE FIRST - no gradients possible
    with torch.inference_mode():
        loss, bpb = score_chunk(adapted_model, chunk)
        accumulate_metrics(loss, bpb)

    # TRAIN on already-scored chunk (except last)
    if i < len(chunks) - 1:
        for epoch in range(3):  # 3-30 epochs typical
            train_loss = adapted_model(chunk)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
```

**Hyperparameters** (from SOTA):
- Chunk size: 32,768 tokens
- Optimizer: SGD + momentum 0.9
- Learning rate: 0.002 (cosine decay across chunks)
- Epochs per chunk: 3-30 (more epochs = better but slower)
- Frozen blocks: None (adapt all blocks) or last N blocks only
- Gradient clip: 1.0

**Variants**:
- **LoRA TTT**: Use rank-8 LoRA adapters instead of full fine-tuning (5x faster)
- **Cosine LR schedule**: Decay LR across chunks for stability
- **AdamW TTT**: AdamW instead of SGD (needs careful tuning)

**Key Rules**:
- NEVER train before scoring a token
- NEVER carry adaptation across documents (reset between docs for LoRA TTT)
- Eval time budget: Must complete in <10 minutes on 8xH100

**First appeared**: PR #17 (samacqua) - LoRA TTT 1.1928 BPB
**Current best**: PR #549+ Legal TTT variants - 1.0781-1.0920 BPB range

---

### 3. N-gram Evaluation Cache (~0.02 BPB improvement)

**Impact**: 0.02 BPB | **Cost**: Minimal CPU time | **Complexity**: Medium

**Concept**: Build n-gram frequency cache from already-scored tokens, interpolate with neural predictions.

**Implementation**:
```python
# Online 5-gram cache during sliding eval
cache = defaultdict(lambda: defaultdict(float))

for token in validation_stream:
    context = tuple(last_4_tokens)  # 4-gram context

    # Get neural prediction
    neural_logits = model(context)
    neural_probs = softmax(neural_logits)

    # Get n-gram prediction from cache
    if context in cache:
        ngram_dist = cache[context]
        # Mix predictions (alpha=0.20 typical)
        mixed_probs = 0.8 * neural_probs + 0.2 * ngram_dist
    else:
        mixed_probs = neural_probs

    # Score with mixed prediction
    loss = -log(mixed_probs[token])

    # AFTER scoring, update cache with this token
    cache[context][token] += 1
```

**Key details**:
- Only use backward-looking tokens (already scored)
- Fixed interpolation weight (0.15-0.25 typical)
- Hash table for memory efficiency (4M buckets typical)
- Minimum count threshold (2-5) to reduce noise

**First appeared**: PR #659 (deanbrr) - 1.0920 BPB with 5-gram cache

---

## Architecture Techniques

### 4. LeakyReLU² Activation (~0.003 BPB improvement)

**Impact**: 0.003 BPB | **Cost**: Zero | **Complexity**: Trivial

**Problem**: Standard ReLU² has dead neurons (negative values zeroed, no gradient).

**Solution**: Leaky version preserves negative gradient flow.

**Implementation**:
```python
# Before (relu²)
x = torch.relu(self.fc(x)).square()

# After (leaky relu²)
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

**Why it works**:
- Squaring still produces non-negative outputs (maintains relu² inductive bias)
- Negative slope 0.5 preserves gradient flow
- No dead neurons in deep networks

**Ablation**: Tested slopes 0.01, 0.1, 0.5, 1.0 - slope 0.5 optimal

**First appeared**: PR #493 (parinzee), PR #518 (sofiabod)

---

### 5. Exclusive Self-Attention (XSA) (~0.005 BPB improvement)

**Impact**: 0.005 BPB | **Cost**: ~2ms/step | **Complexity**: Medium

**Concept**: Subtract the component of attention output aligned with token's own value vector. Forces attention to capture only orthogonal information.

**Implementation**:
```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split([...], dim=-1)

        # Standard attention
        attn_out = flash_attn_func(q, k, v, causal=True)

        # XSA: subtract own-value component
        if self.use_xsa:
            # Efficient GQA-aware implementation
            v_per_head = v.view(B, T, self.num_kv_heads, self.head_dim)
            # Compute dot product of attn_out with own value
            v_expanded = v_per_head.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=2
            )
            attn_out_reshaped = attn_out.view(B, T, self.num_heads, self.head_dim)
            own_component = (attn_out_reshaped * v_expanded).sum(dim=-1, keepdim=True) * v_expanded
            attn_out = attn_out - own_component.view(B, T, C)

        return self.out_proj(attn_out)
```

**Common config**: Apply to last 4 layers only (XSA_LAST_N=4)

**Why it works**: Prevents attention collapse where tokens attend primarily to themselves

**First appeared**: PR #198 (jfprincz) - 1.1318 BPB

---

### 6. Partial RoPE (~0.002 BPB improvement)

**Impact**: 0.002 BPB | **Cost**: Zero | **Complexity**: Low

**Concept**: Apply rotary position embeddings to only first N head dimensions. Remaining dims attend position-invariant.

**Implementation**:
```python
# Instead of rotating all 64 dims, rotate only 16
ROPE_DIMS=16  # vs full head_dim=64

def apply_rotary_emb(q, k, freqs_cos, freqs_sin, rope_dims=16):
    # Split into rotary and pass-through portions
    q_rot = q[..., :rope_dims]
    q_pass = q[..., rope_dims:]
    k_rot = k[..., :rope_dims]
    k_pass = k[..., rope_dims:]

    # Apply RoPE only to first rope_dims
    q_rot = (q_rot * freqs_cos) + (rotate_half(q_rot) * freqs_sin)
    k_rot = (k_rot * freqs_cos) + (rotate_half(k_rot) * freqs_sin)

    # Concatenate back
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k
```

**Typical configs**:
- 16/64 dims (25%) - SOTA
- 24/64 dims (37.5%) - alternative
- Adaptive: 3/4 of head_dim for head_dim>32

**First appeared**: PR #287 (jfprincz) - 1.1248 BPB

---

### 7. U-Net Skip Connections (~0.01 BPB improvement)

**Impact**: 0.01 BPB | **Cost**: ~512 params | **Complexity**: Low

**Concept**: Split transformer into encoder/decoder halves with learned skip connections.

**Implementation**:
```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_idx, num_layers, ...):
        # For encoder layers (first half), save outputs
        self.is_encoder = layer_idx < num_layers // 2
        # For decoder layers (second half), receive skip connections
        self.skip_weight = nn.Parameter(torch.ones(1)) if not self.is_encoder else None

    def forward(self, x, encoder_states=None):
        # Standard transformer ops
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        # Skip connection logic
        if self.is_encoder:
            return x, x  # Save encoder state
        elif self.skip_weight is not None and encoder_states is not None:
            # Add skip connection from corresponding encoder layer
            x = x + self.skip_weight * encoder_states
        return x, None
```

**Typical config**: 11 layers → 5 encoder + 6 decoder

**Why it works**: Decoder accesses early representations directly, not just through residual stream

---

### 8. SmearGate (~0.01 BPB improvement)

**Impact**: 0.01 BPB | **Cost**: ~512 params | **Complexity**: Low

**Concept**: Learned per-dimension gate blending current token embedding with previous token's embedding. Injects bigram context before transformer.

**Implementation**:
```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize to ~0.95 (mostly current token)
        self.gate = nn.Parameter(torch.full((dim,), 3.0))

    def forward(self, x):
        # x: [B, T, dim]
        gate = torch.sigmoid(self.gate)  # Per-dimension blend weight

        # Shift to get previous token embeddings
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))  # Zero-pad at start

        # Blend: mostly current, some previous
        return gate * x + (1 - gate) * prev
```

**Usage**: Apply after embedding lookup and before first layer norm

**Why it works**: Transformer must discover bigram relationships through attention; SmearGate provides this signal directly

**First appeared**: PR #162 (raahilshah)

---

### 9. BigramHash Embedding (~0.01 BPB improvement)

**Impact**: 0.01 BPB | **Cost**: bucket_count × dim params | **Complexity**: Low

**Concept**: Hash consecutive token pairs into learned embedding table. Direct access to token-pair features at minimal cost.

**Implementation**:
```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets=2048, dim=128, model_dim=512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embeddings = nn.Embedding(num_buckets, dim)
        self.proj = nn.Linear(dim, model_dim, bias=False)
        # Hash constants (large primes work well)
        self.hash_a = 92821

    def forward(self, token_ids):
        # token_ids: [B, T]
        prev_ids = F.pad(token_ids[:, :-1], (1, 0))  # Shift

        # Hash: (prev * A + cur) % buckets
        hash_idx = (prev_ids * self.hash_a + token_ids) % self.num_buckets

        # Look up and project
        bigram_emb = self.embeddings(hash_idx)
        return self.proj(bigram_emb)
```

**Typical configs**:
- 2048 buckets × 128 dim = 262K params (standard)
- 4096-10240 buckets for lower collision rate (+0.001-0.002 BPB)

**Why it works**: Directly encodes token-pair statistics that attention would need steps to learn

**First appeared**: PR #162 (raahilshah)

---

### 10. Exponential Moving Average (EMA) (~0.006 BPB improvement)

**Impact**: 0.006 BPB | **Cost**: 2x model memory | **Complexity**: Low

**Concept**: Maintain shadow model that smoothly averages parameters over training.

**Implementation**:
```python
class EMAModel:
    def __init__(self, model, decay=0.997):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model):
        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )

# Usage
ema = EMAModel(model, decay=0.997)
for step in training_loop:
    train_step(model)
    ema.update(model)  # Update every step

# Use EMA weights for evaluation
eval_model = ema.shadow
```

**Typical decays**:
- 0.997: 7000-step runs (standard SOTA)
- 0.9985: Longer runs
- 0.97: Short 200-step ablations

**Why better than SWA**: Continuous smoothing vs discrete checkpoints, better generalization

**First appeared**: PR #287 (jfprincz) - 1.1271 BPB (replacing SWA)

---

### 11. Layer Normalization Scale (~0.002 BPB improvement)

**Impact**: 0.002 BPB | **Cost**: Zero | **Complexity**: Trivial

**Concept**: Scale RMSNorm outputs by `1/sqrt(layer_idx+1)` to damp deeper layers' contributions.

**Implementation**:
```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_idx, use_ln_scale=True, ...):
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1) if use_ln_scale else 1.0

    def forward(self, x):
        # Apply scale after both attention and MLP norms
        x = x + self.ln_scale * self.attn(self.ln1(x))
        x = x + self.ln_scale * self.mlp(self.ln2(x))
        return x
```

**Why it works**: Stabilizes training and convergence in deep (11L+) models. Prevents deep layer dominance.

**First appeared**: PR #287 (jfprincz) - 1.1248 BPB

---

### 12. Grouped Query Attention (GQA) (~0.02 BPB improvement over MHA)

**Impact**: 0.02 BPB vs MHA | **Cost**: None (saves params) | **Complexity**: Standard

**Concept**: Multiple query heads share fewer key/value heads. Reduces KV cache and parameters.

**Implementation**:
```python
# Standard config
NUM_HEADS=8      # Query heads
NUM_KV_HEADS=4   # Key/Value heads (2:1 ratio)

# In attention
q = q.view(B, T, num_heads, head_dim)
k = k.view(B, T, num_kv_heads, head_dim)
v = v.view(B, T, num_kv_heads, head_dim)

# Expand KV to match Q heads
k = k.repeat_interleave(num_heads // num_kv_heads, dim=2)
v = v.repeat_interleave(num_heads // num_kv_heads, dim=2)
```

**Parameter savings**: ~25-40% fewer KV parameters with 2:1 or 4:1 ratios

**Standard in all SOTA submissions** since baseline

---

## Quantization Techniques

### 13. GPTQ-lite (~0.0006 BPB improvement)

**Impact**: 0.0006 BPB | **Cost**: Zero training time | **Complexity**: Medium

**Concept**: Instead of fixed row-max clipping for quantization, search multiple clip percentiles per row and pick the one minimizing reconstruction error.

**Implementation**:
```python
def gptq_lite_quantize(weight_matrix, bits=6):
    # Try multiple clip percentiles
    candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
    best_q = None
    best_mse = float('inf')

    for percentile in candidates:
        # Per-row clipping
        clip_vals = torch.quantile(weight_matrix.abs(), percentile, dim=1)
        clipped = torch.clamp(weight_matrix, -clip_vals[:, None], clip_vals[:, None])

        # Quantize to int6 [-31, 31]
        scale = clip_vals / 31.0
        q = torch.round(clipped / scale[:, None]).clamp(-31, 31)

        # Dequantize and measure error
        reconstructed = q * scale[:, None]
        mse = ((weight_matrix - reconstructed) ** 2).mean()

        if mse < best_mse:
            best_mse = mse
            best_q = (q, scale)

    return best_q
```

**Variants**:
- 5 candidates (fast, standard)
- 15 candidates (thorough, +0.0002 BPB)
- Per-layer vs global search

**First appeared**: PR #374 (signalrush)

---

### 14. Late QAT (Quantization-Aware Training) (~0.003 BPB improvement)

**Impact**: 0.003 BPB (reduces quant gap) | **Cost**: Small slowdown | **Complexity**: Medium

**Concept**: Enable fake quantization (STE) in final 10-15% of training so model learns weights robust to quantization.

**Implementation**:
```python
class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.qat_enabled = False
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1  # 31 for int6

    def enable_qat(self):
        self.qat_enabled = True

    def forward(self, x):
        w = self.weight
        if self.qat_enabled and self.training:
            # Straight-Through Estimator (STE)
            scale = w.abs().max(dim=1).values / self.qmax
            w_q = torch.round(w / scale[:, None]).clamp(-self.qmax, self.qmax)
            w = w_q * scale[:, None]  # Forward uses quantized
            # Backward flows through as if identity (STE)
        return F.linear(x, w)

# Enable when LR scale drops below threshold
if lr_scale < 0.15:
    for module in model.modules():
        if hasattr(module, 'enable_qat'):
            module.enable_qat()
```

**CRITICAL BUG**: torch.compile constant-folds class attributes! Use instance attributes or compile-time checks.

**Typical thresholds**: 0.10-0.15 (activate when LR scale drops below this)

---

### 15. Mixed Precision Quantization (~0.01 BPB improvement)

**Impact**: 0.01 BPB | **Cost**: None (saves space) | **Complexity**: Medium

**Strategy**: Different precisions for different parameter types based on sensitivity.

**Standard config**:
```python
# Int5 for MLPs (most compressible, less sensitive)
mlp_weights: int5 [-16, 15]  # ~1.88x zstd compression

# Int6 for attention (precision-sensitive)
attn_weights: int6 [-31, 31]  # ~1.51x zstd compression

# FP16 for embeddings and control tensors
embeddings: fp16
control_tensors: fp16/fp32  # scales, gates, small params
```

**Benefits**:
- Int5 MLPs save ~1.86MB vs uniform int6
- Can fund 10th layer or wider MLPs
- Minimal accuracy loss if applied correctly

**First appeared**: PR #414 (thwu1) - 1.1428 BPB

---

### 16. Ternary/BitNet Quantization (~3x parameter count)

**Impact**: Enables 65-74M params in 16MB | **Cost**: Slower training | **Complexity**: High

**Concept**: Extreme quantization to {-1, 0, +1} (~1.58 bits/param). Train 3x more parameters in same budget.

**Implementation**:
```python
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.group_size = group_size

    def ternary_quantize(self, w):
        # Group-wise absolute mean scaling
        groups = w.reshape(-1, self.group_size)
        scale = groups.abs().mean(dim=1, keepdim=True)

        # Quantize to {-1, 0, +1}
        w_scaled = w / (scale.reshape(-1, 1) + 1e-6)
        w_ternary = torch.sign(w_scaled) * (w_scaled.abs() > 0.5).float()

        # Dequantize
        return w_ternary * scale.reshape(-1, 1)

    def forward(self, x):
        if self.training:
            # STE: forward uses ternary, backward flows through
            w = self.ternary_quantize(self.weight)
        else:
            w = self.weight
        return F.linear(x, w)
```

**Challenges**:
- Oscillator/recurrent parameters accumulate quantization drift
- Need careful STE implementation
- Some techniques incompatible (EMA, certain Muon WD configs)

**Compression**: Base-3 packing + LZMA gives ~39% reduction over int8+zlib

**First appeared**: PR #424 (Ciprian-Florin Ifrim) - Ternary submissions in notable runs

---

## Optimizer Techniques

### 17. Muon Optimizer (Standard in SOTA)

**Impact**: Essential baseline | **Cost**: ~5-10ms/step for Newton-Schulz | **Complexity**: Medium

**Concept**: Orthogonalize gradient updates via Newton-Schulz iteration. Equivalent to steepest descent under spectral norm.

**Implementation** (simplified):
```python
def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize 2D gradient matrix."""
    X = G.bfloat16() / (G.norm() + 1e-7)

    # Transpose if needed (prefer shorter dimension)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Newton-Schulz iteration
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Apply momentum
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(p.grad)
                    g = p.grad.add(buf, alpha=momentum)  # Nesterov

                    # Orthogonalize (only for 2D matrices)
                    if g.ndim == 2:
                        g = zeropower_via_newtonschulz5(g, steps=5)
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5  # Scale correction

                    # Update
                    p.add_(g, alpha=-lr)
```

**Typical hyperparameters**:
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_START=0.92` (warmup over 500-1500 steps)
- `MUON_BACKEND_STEPS=5` (Newton-Schulz iterations, 3 for ternary models)
- `MUON_WD=0.04` (decoupled weight decay)

**Optimizations**:
- **Parameter Banking** (PR #399): Batch orthogonalization for 4x speedup (83ms vs 85ms/step)
- **Parallel Muon**: Async reduce-scatter → local NS → async all-gather

**Used in**: All SOTA submissions

---

### 18. Decoupled Weight Decay (~0.01 BPB improvement)

**Impact**: 0.01 BPB | **Cost**: Zero | **Complexity**: Trivial

**Concept**: Apply weight decay before gradient update (AdamW-style) rather than as L2 regularization.

**Implementation**:
```python
# Before gradient update
for p in model.parameters():
    p.mul_(1 - weight_decay * learning_rate)

# Then apply gradient update
optimizer.step()
```

**Typical values**: WD=0.04 for both Muon and AdamW optimizers

**Why it works**: Keeps weights smaller and better-distributed → better quantization and compression

---

### 19. Stochastic Weight Averaging (SWA) (~0.005 BPB improvement)

**Impact**: 0.005 BPB | **Cost**: Storage for checkpoints | **Complexity**: Low

**Concept**: Average multiple checkpoints from warmdown phase.

**Implementation**:
```python
# Collect checkpoints during warmdown
swa_checkpoints = []
warmdown_start = iterations - warmdown_iters

for step in range(iterations):
    train_step()

    # Collect during warmdown
    if step >= warmdown_start and (step - warmdown_start) % swa_every == 0:
        swa_checkpoints.append(copy.deepcopy(model.state_dict()))

# Average at end
averaged_state = {}
for key in swa_checkpoints[0].keys():
    averaged_state[key] = torch.stack([
        ckpt[key] for ckpt in swa_checkpoints
    ]).mean(dim=0)
```

**Variants**:
- **Tight SWA**: Only collect when LR scale < 0.2
- **SWA with start_frac**: Only collect from last 40-60% of warmdown (more converged checkpoints)
- **SWA_EVERY**: 30-50 steps typical

**Often combined with EMA**: EMA for continuous smoothing, SWA for discrete high-quality checkpoints

---

## Advanced Architecture Techniques

### 20. Value Embedding (VE) (~0.002 BPB improvement)

**Impact**: 0.002 BPB | **Cost**: ~128 × num_layers params | **Complexity**: Medium

**Concept**: Share value embeddings across specified layers with per-layer learned scales.

**Implementation**:
```python
class SharedValueEmbedding(nn.Module):
    def __init__(self, dim=128, model_dim=512, layers=[9, 10]):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(model_dim, dim))
        self.scales = nn.ParameterDict({
            str(i): nn.Parameter(torch.ones(1)) for i in layers
        })

    def get_for_layer(self, layer_idx):
        if str(layer_idx) in self.scales:
            return self.scales[str(layer_idx)] * self.embedding
        return None

# In attention layer
if ve := value_embedding.get_for_layer(self.layer_idx):
    v = v + ve  # Add shared component
```

**Typical config**: VE_DIM=128, VE_LAYERS=9,10 (last 2 layers)

---

### 21. Orthogonal Initialization (~0.005 BPB improvement)

**Impact**: 0.005 BPB | **Cost**: Zero | **Complexity**: Trivial

**Concept**: Initialize weight matrices as orthogonal matrices (all singular values = 1).

**Implementation**:
```python
# For all weight matrices
nn.init.orthogonal_(linear_layer.weight)

# With muP-style output scaling
nn.init.orthogonal_(linear_layer.weight)
linear_layer.weight.mul_(model_dim ** -0.5)  # Scale for muP
```

**Why it works**:
- Uniform gradient flow at initialization (no vanishing/exploding)
- Muon orthogonalizes updates, so starting orthogonal means immediate useful updates
- Critical with only ~7000 steps in 10-min budget

**Standard in all SOTA submissions** since PR #162

---

### 22. Depth Recurrence (~0.01 BPB improvement, enables wider models)

**Impact**: Variable | **Cost**: Complexity | **Complexity**: High

**Concept**: Loop N unique layers M times for N×M effective layers. Saves N×(M-1) layers worth of parameters.

**Implementation**:
```python
class RecurrentTransformer(nn.Module):
    def __init__(self, num_unique_layers=6, num_loops=2):
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_unique_layers)])
        self.num_loops = num_loops

        # Per-loop conditioning
        self.loop_scale = nn.ParameterList([
            nn.Parameter(torch.ones(num_unique_layers))
            for _ in range(num_loops)
        ])
        self.loop_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(num_unique_layers))
            for _ in range(num_loops)
        ])

    def forward(self, x):
        for loop_idx in range(self.num_loops):
            for layer_idx, layer in enumerate(self.layers):
                # Condition on loop iteration
                scale = self.loop_scale[loop_idx][layer_idx]
                bias = self.loop_bias[loop_idx][layer_idx]
                x = scale * layer(x) + bias
        return x
```

**Example**: 6 unique layers × 2 loops = 12 effective layers, saves ~7.5MB

**LR scaling**: Divide matrix/scalar LRs by `sqrt(num_loops)`

---

### 23. Differential Attention (~0.01 BPB improvement, but slow)

**Impact**: 0.01 BPB | **Cost**: 2-3x slower without custom kernels | **Complexity**: High

**Concept**: Subtract two attention heads to amplify focused attention and suppress noise.

**Status**: Novel for competition but requires FlashAttention integration for speed. Several PRs exploring.

**Paper**: arXiv:2410.05258 (ICLR 2025)

---

## Training Tricks

### 24. Warmdown Schedule (Essential)

**Impact**: Critical for convergence | **Cost**: Zero | **Complexity**: Trivial

**Concept**: Linear decay of learning rate over final 3000-3500 iterations.

```python
warmdown_iters = 3500
current_iter = step

if current_iter > iterations - warmdown_iters:
    warmdown_progress = (current_iter - (iterations - warmdown_iters)) / warmdown_iters
    lr_scale = 1.0 - warmdown_progress
else:
    lr_scale = 1.0

actual_lr = base_lr * lr_scale
```

**Why critical**: Models converge dramatically in final warmdown steps. Longer warmdown = better convergence.

**Tuning**: 3000 standard, 3500+ for advanced runs

---

### 25. Gradient Clipping (~0.002 BPB improvement)

**Impact**: 0.002 BPB | **Cost**: Zero | **Complexity**: Trivial

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
```

**Typical values**: 0.3 (standard), 1.0 (for TTT)

---

### 26. Learning Rate Schedules

**Warmup**: 20 steps typical (very short due to 10-min constraint)

**Separate LRs by parameter type**:
```python
TIED_EMBED_LR=0.035      # Tied embeddings (highest)
MATRIX_LR=0.025          # Muon-optimized matrices
SCALAR_LR=0.025          # Scalars, biases
HEAD_LR=0.008            # Output head (if untied)
```

**Why separate**: Different parameter types have different optimal learning rates and update dynamics

---

## Advanced Techniques

### 27. FlashAttention 3 (Hopper-Optimized)

Essential for long sequences on H100. ~10% speedup over FA2.

```python
from flash_attn import flash_attn_func

attn_out = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
```

---

### 28. Logit Softcapping

Prevents logit explosion, stabilizes training.

```python
LOGIT_SOFTCAP=30.0

logits = torch.tanh(logits / softcap) * softcap
```

**Variants**:
- Standard: tanh capping (cap=30)
- Polynomial: degree 5 polynomial (cap=10)

---

### 29. NTK-Aware RoPE Scaling

Extends effective context length for sequences longer than RoPE base.

```python
ROPE_BASE=10000.0  # Standard
TRAIN_SEQ_LEN=2048

# NTK-aware scaling
if seq_len > base_seq_len:
    scale = seq_len / base_seq_len
    rope_base = rope_base * scale
```

**Variant**: YaRN (Yet Another RoPE extensioN) for even longer contexts

---

## Compression Techniques

### 30. Compression Algorithm Choice

**zstd-22** (standard):
- Fast decompression
- ~1.5-1.9x compression on quantized weights
- `zstd.compress(bytes, level=22)`

**lzma** (tighter, 2-5% better):
- Slower but within eval budget
- `lzma.compress(bytes, preset=6)`

**Base-3 + LZMA** (for ternary):
- Pack 5 trits per byte
- 39% reduction vs int8+zlib

---

## System Optimizations

### 31. torch.compile

Essential for performance. Use `torch.compile(model, mode='default')`.

**Gotchas**:
- Class attributes get constant-folded at first trace
- Use instance attributes for dynamic flags
- Recompilation mid-training can cause OOM

### 32. Data Loading Optimizations

- Pinned memory for H2D transfers
- Async prefetch with background thread + separate CUDA stream
- Coprime-stride shard sampling for diversity

---

## Testing & Verification Workflow

1. **Smoke test** (200-1000 steps): Verify training runs without errors
2. **1xH100 ablation** (~900-1100 steps, 10 min): Quick iteration
3. **8xH100 full run** (~7000 steps, 10 min): Competition-scale validation
4. **3-seed statistical test**: Prove significance with p < 0.01

---

## Common Variable Naming Conventions

```bash
# Environment variables used in train_gpt.py
NUM_LAYERS=11                    # Transformer depth
MODEL_DIM=512                    # Hidden dimension
NUM_HEADS=8                      # Attention heads
NUM_KV_HEADS=4                   # GQA KV heads
MLP_MULT=3                       # MLP expansion (hidden = dim × mult)
VOCAB_SIZE=1024                  # Tokenizer vocabulary size
TRAIN_SEQ_LEN=2048               # Sequence length
TRAIN_BATCH_TOKENS=786432        # Batch size in tokens
ITERATIONS=9000                  # Total training steps
WARMDOWN_ITERS=3500              # Warmdown duration
EVAL_STRIDE=64                   # Sliding window stride
XSA_LAST_N=4                     # Apply XSA to last N layers
BIGRAM_VOCAB_SIZE=2048           # BigramHash bucket count
ROPE_DIMS=16                     # Partial RoPE dimensions
EMA_DECAY=0.997                  # EMA decay rate
```

---

## Technique Compatibility Matrix

| Technique | Compatible With | Incompatible With |
|-----------|----------------|-------------------|
| LeakyReLU² | Everything | None |
| XSA | Most techniques | Very deep models (slowdown) |
| TTT | Most eval methods | None (but expensive) |
| Ternary/BitNet | Most arch changes | EMA, some Muon configs, VRL |
| EMA | Most techniques | Ternary (sometimes) |
| Parameter Banking | Most techniques | Non-standard parameter layouts |
| Partial RoPE | Everything | None |
| BigramHash | Everything | None |

---

## Recent Frontier Innovations (March 2026)

1. **Podracing** (PR #674): 5-gram eval interpolation → 1.0461 BPB
2. **30-Epoch TTT** (PR #672): Longer TTT adaptation → 1.0781 BPB
3. **Enhanced Attention** (PR #684): Learned k/v shift mixing, adaptive RoPE → 1.0574 BPB
4. **Sidecar Architectures**: Shared sparse side-car parameters
5. **Async Data Pipeline**: Memory-mapped with prefetch
6. **Manifold-Guided Architecture**: Novel geometric approach

---

## Key Metrics to Track

- **val_bpb**: Primary metric (bits per byte on validation)
- **val_loss**: Cross-entropy loss (natural log)
- **Quantization gap**: BPB increase from quantization (0.01-0.03 typical)
- **Steps completed**: More steps in 10 min = better (7000-7200 typical for SOTA)
- **ms/step**: Training speed (82-92ms typical on 8xH100)
- **Artifact size**: Must be ≤16,000,000 bytes
- **Peak memory**: ~20GB per H100 for SOTA configs

---

## Research Strategy

1. **Start with baseline**: Understand 1.2244 BPB baseline first
2. **Add techniques incrementally**: One at a time, measure impact
3. **Sliding window first**: Largest eval gain for lowest cost
4. **Architecture then quantization**: Get model working, then compress
5. **TTT last**: Most expensive, add after other optimizations
6. **3-seed validation**: Always test statistical significance

---

## Related Sub-Skills

- Use `/parameter-golf-submit` for submission process details
- Use `/parameter-golf-implement` for code implementation guidance
- Use `/parameter-golf-verify` for testing and reproducibility
- Use `/parameter-golf-debug` for troubleshooting common issues
