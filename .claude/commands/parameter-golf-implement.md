---
name: parameter-golf-implement
description: Practical guide for implementing Parameter Golf techniques in train_gpt.py
---

# Parameter Golf Implementation Guide

This skill provides step-by-step guidance for implementing specific techniques in your train_gpt.py script.

## Understanding train_gpt.py Structure

The baseline `train_gpt.py` has this structure (~1500 lines):

```python
# 1. HYPERPARAMETERS (lines 1-90)
class Hyperparameters:
    # All config via environment variables
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    # ... etc

# 2. MUON OPTIMIZER (lines 90-170)
class Muon(torch.optim.Optimizer):
    # Orthogonalizing optimizer

# 3. TOKENIZER-AGNOSTIC EVALUATION (lines 170-280)
def eval_val(...):
    # Computes val_loss and val_bpb

# 4. QUANTIZATION (lines 280-425)
def quantize_state_dict_int8(...):
def dequantize_state_dict_int8(...):

# 5. DATA LOADING (lines 425-495)
class TokenStream:
class DistributedTokenLoader:

# 6. TRANSFORMER MODULES (lines 495-800)
class RMSNorm:
class CausalSelfAttention:
class TransformerBlock:
class GPT:

# 7. MAIN TRAINING LOOP (lines 800-1500)
def main():
    # Setup, training loop, evaluation, export
```

---

## Implementation Patterns

### Pattern 1: Adding Environment Variables

**When**: New hyperparameter or feature flag

**Where**: In `Hyperparameters` class (top of file)

**Example**:
```python
class Hyperparameters:
    # ... existing params ...

    # Your new parameters
    use_xsa = bool(int(os.environ.get("USE_XSA", "0")))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
```

**Usage**:
```bash
USE_XSA=1 XSA_LAST_N=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

### Pattern 2: Modifying Activation Functions

**Goal**: Change from relu² to leaky_relu²

**Location**: MLP class in TransformerBlock

**Steps**:
1. Add hyperparameter for activation choice
2. Modify MLP forward pass
3. Test with small model

**Implementation**:
```python
# In Hyperparameters
class Hyperparameters:
    activation = os.environ.get("ACTIVATION", "relu2")  # or "leaky_relu2"
    leaky_slope = float(os.environ.get("LEAKY_SLOPE", 0.5))

# In TransformerBlock or MLP class
class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.model_dim, args.model_dim * args.mlp_mult, bias=False)
        self.proj = nn.Linear(args.model_dim * args.mlp_mult, args.model_dim, bias=False)
        self.activation = args.activation
        self.leaky_slope = args.leaky_slope

    def forward(self, x):
        x = self.fc(x)
        if self.activation == "leaky_relu2":
            x = F.leaky_relu(x, negative_slope=self.leaky_slope).square()
        else:  # relu2 (default)
            x = torch.relu(x).square()
        x = self.proj(x)
        return x
```

**Test**:
```bash
# Test both activations produce valid results
ACTIVATION=relu2 ITERATIONS=10 python3 train_gpt.py
ACTIVATION=leaky_relu2 LEAKY_SLOPE=0.5 ITERATIONS=10 python3 train_gpt.py
```

---

### Pattern 3: Adding Architectural Components (e.g., SmearGate)

**Goal**: Add SmearGate to embedding layer

**Location**: After embedding lookup, before transformer

**Steps**:
1. Define SmearGate module
2. Add to GPT.__init__
3. Apply in GPT.forward
4. Add hyperparameters

**Implementation**:
```python
# 1. Define module (near other modules)
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0))  # sigmoid(3) ≈ 0.95

    def forward(self, x):
        # x: [B, T, dim]
        gate = torch.sigmoid(self.gate)
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))  # Shift to get previous tokens
        return gate * x + (1 - gate) * prev

# 2. Add to GPT.__init__
class GPT(nn.Module):
    def __init__(self, args):
        # ... existing init ...
        self.use_smeargate = args.use_smeargate
        if self.use_smeargate:
            self.smeargate = SmearGate(args.model_dim)

# 3. Apply in forward
class GPT(nn.Module):
    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)  # [B, T, dim]

        if self.use_smeargate:
            x = self.smeargate(x)

        # Continue with transformer blocks...
        for block in self.blocks:
            x = block(x)
        # ...

# 4. Add hyperparameter
class Hyperparameters:
    use_smeargate = bool(int(os.environ.get("USE_SMEARGATE", "0")))
```

---

### Pattern 4: Implementing Sliding Window Evaluation

**Goal**: Replace standard eval with sliding window

**Location**: Add new eval function, modify main()

**Steps**:
1. Create `eval_val_sliding()` function
2. Add stride parameter
3. Call instead of `eval_val()`

**Implementation**:
```python
# Add new function after eval_val
def eval_val_sliding(
    args, model, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64
):
    """
    Sliding window evaluation: score tokens with full context.
    Only tokens in the rightmost 'stride' positions are scored.
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Each rank handles subset of windows
    num_windows = (total_tokens - seq_len) // stride + 1
    windows_per_rank = (num_windows + world_size - 1) // world_size
    start_window = rank * windows_per_rank
    end_window = min(start_window + windows_per_rank, num_windows)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = 0
    val_byte_count = 0

    model.eval()
    with torch.inference_mode():
        for window_idx in range(start_window, end_window):
            # Window spans [start : start + seq_len + 1]
            start = window_idx * stride
            window_tokens = val_tokens[start : start + seq_len + 1]

            # Move to device
            x = window_tokens[:-1].unsqueeze(0).to(device, dtype=torch.int64)
            y = window_tokens[1:].unsqueeze(0).to(device, dtype=torch.int64)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)  # [1, seq_len, vocab_size]

            # Score only the rightmost 'stride' tokens
            logits_scored = logits[:, -stride:]
            y_scored = y[:, -stride:]

            loss = F.cross_entropy(
                logits_scored.reshape(-1, logits.size(-1)),
                y_scored.reshape(-1),
                reduction='sum'
            )
            val_loss_sum += loss.detach().to(torch.float64)

            # Count bytes (for BPB)
            prev_ids = x[:, -stride-1:-1].reshape(-1)
            tgt_ids = y_scored.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids]
            token_bytes += (has_leading_space_lut[tgt_ids] &
                           ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += token_bytes.sum().item()
            val_token_count += stride

    # All-reduce across ranks
    if dist.is_available() and dist.is_initialized():
        val_loss_sum_tensor = val_loss_sum.clone()
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        val_loss_sum = val_loss_sum_tensor

        count_tensor = torch.tensor([val_token_count, val_byte_count],
                                    device=device, dtype=torch.float64)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        val_token_count = count_tensor[0].item()
        val_byte_count = count_tensor[1].item()

    val_loss = val_loss_sum.item() / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    val_bpb = bits_per_token * tokens_per_byte

    model.train()
    return val_loss, val_bpb

# In main(), replace eval call
if args.eval_stride > 0:
    val_loss, val_bpb = eval_val_sliding(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride
    )
else:
    val_loss, val_bpb = eval_val(...)  # Standard eval
```

**Hyperparameters**:
```python
class Hyperparameters:
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))  # 0 = standard, 64 = sliding
```

---

### Pattern 5: Adding EMA

**Goal**: Maintain exponential moving average of parameters

**Location**: In main training loop

**Steps**:
1. Create EMA shadow model after model initialization
2. Update after each training step
3. Use EMA weights for evaluation

**Implementation**:
```python
# In Hyperparameters
class Hyperparameters:
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "0")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

# In main(), after model creation
def main():
    # ... model creation ...
    model = GPT(args)
    model = torch.compile(model)

    # Create EMA shadow model
    if args.ema_enabled:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        # Don't compile EMA (only used for eval)

    # Training loop
    for step in range(args.iterations):
        # ... training step ...
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update EMA every step
        if args.ema_enabled:
            with torch.no_grad():
                decay = args.ema_decay
                for ema_p, model_p in zip(ema_model.parameters(),
                                          model.parameters()):
                    ema_p.mul_(decay).add_(model_p.data, alpha=1 - decay)

    # Use EMA for evaluation
    eval_model = ema_model if args.ema_enabled else model
    val_loss, val_bpb = eval_val(args, eval_model, ...)
```

**Memory consideration**: EMA doubles model memory. On 8xH100 with 80GB, this is fine for 512d models.

---

### Pattern 6: Implementing XSA (Exclusive Self-Attention)

**Goal**: Add XSA to last N layers

**Location**: CausalSelfAttention module

**Steps**:
1. Add layer_idx tracking
2. Modify attention forward pass
3. Add XSA computation for eligible layers

**Implementation**:
```python
# In Hyperparameters
class Hyperparameters:
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))  # 0 = disabled

# In CausalSelfAttention.__init__
class CausalSelfAttention(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = args.num_layers
        self.use_xsa = args.xsa_last_n > 0 and \
                       layer_idx >= args.num_layers - args.xsa_last_n
        # ... rest of init ...

    def forward(self, x):
        B, T, C = x.shape

        # Get Q, K, V
        q, k, v = self.qkv(x).split([
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim
        ], dim=-1)

        # Reshape for attention
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Standard FlashAttention
        from flash_attn import flash_attn_func
        attn_out = flash_attn_func(q, k, v, causal=True)

        # Apply XSA if enabled for this layer
        if self.use_xsa:
            # Expand V to match attention heads (for GQA)
            v_expanded = v.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=2
            )  # [B, T, num_heads, head_dim]

            # Compute projection of attn_out onto own value
            # attn_out: [B, T, num_heads, head_dim]
            dot_product = (attn_out * v_expanded).sum(dim=-1, keepdim=True)
            own_value_component = dot_product * v_expanded

            # Subtract own-value component
            attn_out = attn_out - own_value_component

        # Reshape back and project
        attn_out = attn_out.reshape(B, T, C)
        return self.out_proj(attn_out)
```

**Testing**:
```bash
# Without XSA (baseline)
USE_XSA=0 ITERATIONS=100 python3 train_gpt.py

# With XSA on last 4 layers
USE_XSA=1 XSA_LAST_N=4 ITERATIONS=100 python3 train_gpt.py

# Verify training doesn't diverge, loss improves
```

---

### Pattern 7: Adding U-Net Skip Connections

**Goal**: Connect encoder and decoder layers

**Location**: TransformerBlock and GPT classes

**Steps**:
1. Identify encoder/decoder split
2. Add skip_weight parameters to decoder blocks
3. Pass encoder states through forward pass
4. Apply skip connections in decoder

**Implementation**:
```python
# In TransformerBlock.__init__
class TransformerBlock(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = args.num_layers

        # Determine if this is encoder or decoder
        self.is_encoder = layer_idx < num_layers // 2

        # Decoder blocks get skip weights
        if not self.is_encoder:
            self.skip_weight = nn.Parameter(torch.ones(1))
        else:
            self.skip_weight = None

        # ... rest of init ...

    def forward(self, x, encoder_state=None):
        # Attention and MLP
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        # Handle skip connections
        if self.is_encoder:
            # Return both output and state for skip
            return x, x
        else:
            # Decoder: apply skip if available
            if self.skip_weight is not None and encoder_state is not None:
                x = x + self.skip_weight * encoder_state
            return x, None

# In GPT.forward
class GPT(nn.Module):
    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        encoder_states = []

        # Forward pass with U-Net structure
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) // 2:
                # Encoder: save states
                x, state = block(x, encoder_state=None)
                encoder_states.append(state)
            else:
                # Decoder: use skip from corresponding encoder
                encoder_idx = len(self.blocks) - 1 - i
                skip = encoder_states[encoder_idx] if encoder_idx >= 0 else None
                x, _ = block(x, encoder_state=skip)

        # ... rest of forward ...
```

---

### Pattern 8: Implementing GPTQ-lite Quantization

**Goal**: Post-training quantization with optimal clipping

**Location**: Add quantization function, modify export code

**Implementation**:
```python
def gptq_lite_quantize_int6(tensor, num_candidates=5):
    """
    Try multiple clip percentiles, pick best MSE.
    """
    if tensor.ndim != 2:
        # Fall back to standard for non-matrices
        return standard_quantize_int6(tensor)

    candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0][:num_candidates]

    best_q = None
    best_scale = None
    best_mse = float('inf')

    for percentile in candidates:
        # Per-row clip
        clip_vals = torch.quantile(tensor.abs(), percentile, dim=1)
        clip_vals = clip_vals.clamp_min(1e-6)

        # Quantize
        scale = clip_vals / 31.0
        clipped = torch.clamp(tensor, -clip_vals[:, None], clip_vals[:, None])
        q = torch.round(clipped / scale[:, None]).clamp(-31, 31)

        # Measure reconstruction error
        reconstructed = q * scale[:, None]
        mse = ((tensor - reconstructed) ** 2).mean()

        if mse < best_mse:
            best_mse = mse
            best_q = q.to(torch.int8)
            best_scale = scale

    return best_q, best_scale

# In export/quantization section
def quantize_model_gptq_lite(state_dict):
    quantized = {}
    scales = {}

    for name, tensor in state_dict.items():
        if should_quantize(name, tensor):
            q, s = gptq_lite_quantize_int6(tensor)
            quantized[name] = q
            scales[name] = s.to(torch.float16)
        else:
            quantized[name] = tensor  # Passthrough

    return {"quantized": quantized, "scales": scales}
```

---

### Pattern 9: Adding Late QAT (Quantization-Aware Training)

**Goal**: Enable fake quantization in final training phase

**Location**: CastedLinear class, main training loop

**Important**: Avoid torch.compile constant-folding bug!

**Implementation**:
```python
# In CastedLinear class
class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # IMPORTANT: Use register_buffer for compile-safe flag
        self.register_buffer('qat_enabled', torch.tensor(False))

    def enable_qat(self):
        self.qat_enabled.fill_(True)

    def forward(self, x):
        w = self.weight

        # QAT with STE (if enabled and training)
        if self.qat_enabled.item() and self.training:
            # Per-row scaling for int6
            scale = w.abs().max(dim=1).values.clamp_min(1e-6) / 31.0
            w_scaled = w / scale[:, None]
            w_q = torch.round(w_scaled).clamp(-31, 31)

            # Forward uses quantized, backward flows through (STE)
            w = w_q * scale[:, None]

        return F.linear(x, w)

# In main training loop
def main():
    # ... training setup ...
    late_qat_threshold = args.late_qat_threshold  # e.g., 0.15

    for step in range(args.iterations):
        # ... training step ...

        # Enable QAT when LR scale drops below threshold
        if not qat_enabled and lr_scale < late_qat_threshold:
            if rank == 0:
                print(f"Enabling QAT at step {step}, lr_scale={lr_scale:.4f}")
            for module in model.modules():
                if hasattr(module, 'enable_qat'):
                    module.enable_qat()
            qat_enabled = True
```

---

### Pattern 10: Implementing Test-Time Training (Legal Score-First)

**Goal**: Adapt model during evaluation on scored chunks

**Location**: Add TTT eval function

**Steps**:
1. Split validation into chunks
2. Score each chunk under inference_mode
3. Train on scored chunks (except last)
4. Return final metrics

**Implementation**:
```python
def eval_val_ttt_legal(args, model, val_tokens, device, rank, world_size, ...):
    """
    Legal score-first TTT:
    1. Score chunk under inference_mode (no grad tracking possible)
    2. Train on that chunk (except last)
    3. Repeat for all chunks
    """
    chunk_size = args.ttt_chunk_tokens  # e.g., 32768
    chunks = split_into_chunks(val_tokens, chunk_size)

    # Deep copy model for adaptation
    adapted_model = copy.deepcopy(model)
    adapted_model.train()

    # TTT optimizer (SGD typical)
    ttt_optimizer = torch.optim.SGD(
        adapted_model.parameters(),
        lr=args.ttt_lr,  # e.g., 0.002
        momentum=args.ttt_momentum  # e.g., 0.9
    )

    # Cosine LR schedule across chunks
    num_adapt_chunks = len(chunks) - 1  # Last chunk not trained on

    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0

    for chunk_idx, chunk_tokens in enumerate(chunks):
        # === SCORE FIRST ===
        adapted_model.eval()
        with torch.inference_mode():  # Hard guarantee no weight mutation
            loss, bytes_count = score_chunk(
                adapted_model, chunk_tokens, device, ...
            )
            total_loss += loss
            total_bytes += bytes_count
            total_tokens += len(chunk_tokens) - 1

        # === TRAIN (except last chunk) ===
        if chunk_idx < len(chunks) - 1:
            adapted_model.train()

            # Cosine LR decay across chunks
            progress = chunk_idx / num_adapt_chunks
            lr = args.ttt_lr * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in ttt_optimizer.param_groups:
                param_group['lr'] = lr

            # Train for N epochs
            for epoch in range(args.ttt_epochs):  # e.g., 3
                # Batch the chunk for GPU efficiency
                for batch_start in range(0, len(chunk_tokens) - args.ttt_seq_len,
                                        args.ttt_batch_tokens):
                    x, y = prepare_batch(chunk_tokens, batch_start, ...)

                    # Forward + backward
                    train_loss = adapted_model(x, y)
                    train_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        adapted_model.parameters(), args.ttt_grad_clip
                    )

                    ttt_optimizer.step()
                    ttt_optimizer.zero_grad()

    # Compute final metrics
    val_loss = total_loss / total_tokens
    val_bpb = compute_bpb(total_loss, total_bytes, total_tokens)

    return val_loss, val_bpb

# Hyperparameters for TTT
class Hyperparameters:
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))  # 0 = all blocks
```

**Testing considerations**:
- TTT adds 400-500s to eval time
- Must stay under 10-min eval budget
- Test with 1 epoch first, then scale up

---

### Pattern 11: Implementing BigramHash Embedding

**Goal**: Add hash-based bigram embeddings

**Location**: After token embeddings, before transformer

**Implementation**:
```python
# In Hyperparameters
class Hyperparameters:
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))  # 0 = disabled
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

# BigramHash module
class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets, dim, model_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.embeddings = nn.Embedding(num_buckets, dim)
        self.proj = nn.Linear(dim, model_dim, bias=False)

        # Hash constants (large prime)
        self.hash_multiplier = 92821

        # Initialize
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, token_ids):
        # token_ids: [B, T]
        B, T = token_ids.shape

        # Get previous tokens (zero-pad first position)
        prev_ids = F.pad(token_ids[:, :-1], (1, 0), value=0)

        # Hash: (prev * multiplier + cur) % buckets
        hash_idx = (prev_ids * self.hash_multiplier + token_ids) % self.num_buckets

        # Lookup and project
        bigram_emb = self.embeddings(hash_idx)
        return self.proj(bigram_emb)

# In GPT.__init__
class GPT(nn.Module):
    def __init__(self, args):
        # ... standard init ...
        self.token_embedding = nn.Embedding(args.vocab_size, args.model_dim)

        # Add BigramHash if enabled
        if args.bigram_vocab_size > 0:
            self.bigram_hash = BigramHashEmbedding(
                args.bigram_vocab_size,
                args.bigram_dim,
                args.model_dim
            )

    def forward(self, idx, targets=None):
        # Token embeddings
        x = self.token_embedding(idx)

        # Add bigram hash embeddings
        if hasattr(self, 'bigram_hash'):
            x = x + self.bigram_hash(idx)

        # ... continue with transformer ...
```

---

## Common Implementation Challenges

### Challenge 1: torch.compile Compatibility

**Problem**: Dynamic control flow or mutable class attributes

**Solution**:
```python
# BAD: Class attribute (gets constant-folded)
class MyModule(nn.Module):
    qat_enabled = False  # torch.compile sees this as constant!

    def forward(self, x):
        if self.qat_enabled:  # This branch gets eliminated
            x = quantize(x)
        return x

# GOOD: Buffer or instance attribute
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('qat_enabled', torch.tensor(False))

    def forward(self, x):
        if self.qat_enabled.item():  # Runtime check
            x = quantize(x)
        return x
```

---

### Challenge 2: Memory Management

**Problem**: OOM with EMA + large models

**Solution**:
```python
# Option 1: Selective EMA (only large parameters)
if args.ema_enabled:
    ema_params = {
        name: p for name, p in model.named_parameters()
        if p.numel() > 1000  # Only large parameters
    }

# Option 2: CPU offloading (slow)
if args.ema_enabled:
    ema_model = copy.deepcopy(model).cpu()
    # Move to GPU only for eval

# Option 3: Gradient checkpointing
model = torch.compile(model, mode='reduce-overhead')
```

---

### Challenge 3: Distributed Training Correctness

**Problem**: All-reduce not applied correctly

**Solution**:
```python
# Always check if distributed
if dist.is_available() and dist.is_initialized():
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # For averages, divide by world_size
    tensor /= dist.get_world_size()

# For losses, sum then divide by total count
dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
dist.all_reduce(count, op=dist.ReduceOp.SUM)
final_loss = loss_sum / count
```

---

### Challenge 4: FlashAttention Integration

**Problem**: GQA with FlashAttention

**Solution**:
```python
from flash_attn import flash_attn_func

# For GQA: repeat K, V to match Q heads
q = q.view(B, T, num_heads, head_dim)
k = k.view(B, T, num_kv_heads, head_dim)
v = v.view(B, T, num_kv_heads, head_dim)

# Expand K, V (FlashAttention handles this internally)
# If using flash_attn_func, it auto-expands if shapes differ
attn_out = flash_attn_func(q, k, v, causal=True)

# Returns [B, T, num_heads, head_dim]
attn_out = attn_out.reshape(B, T, num_heads * head_dim)
```

---

## Code Organization Best Practices

### 1. Keep train_gpt.py Under 1500 Lines

**Strategies**:
- Combine related functionality
- Remove unused experimental code
- Use compact implementations
- Inline small helper functions

### 2. Use Environment Variables for All Config

**Why**: Easy to sweep hyperparameters without editing code

```python
class Hyperparameters:
    # Good pattern
    technique_enabled = bool(int(os.environ.get("TECHNIQUE_ENABLED", "0")))
    technique_param = float(os.environ.get("TECHNIQUE_PARAM", 0.5))

# Then sweep:
for param in [0.3, 0.5, 0.7]:
    run_command(f"TECHNIQUE_PARAM={param} torchrun ...")
```

### 3. Self-Contained Modules

Everything needed must be in train_gpt.py or importable from standard libraries:

```python
# Good
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

# Bad
from my_custom_module import special_function  # Must inline this!
```

### 4. Proper Random Seeding

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Call at start of main()
set_seed(args.seed)
```

---

## Debugging Your Implementation

### Issue: Training Diverges

**Symptoms**: Loss becomes NaN or infinity

**Debug steps**:
1. Check learning rates (might be too high)
2. Verify gradient clipping applied
3. Check for division by zero in custom code
4. Inspect weight initialization (all finite?)
5. Disable torch.compile temporarily

```python
# Add gradient norm logging
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if rank == 0:
    print(f"Step {step}, grad_norm: {grad_norm:.4f}, loss: {loss:.4f}")
```

---

### Issue: Slower Than Expected

**Symptoms**: <6000 steps in 10 min (expect 7000+)

**Debug steps**:
1. Profile with PyTorch profiler
2. Check torch.compile enabled
3. Verify FlashAttention being used
4. Check for CPU-GPU synchronization points
5. Measure step time components

```python
import time

# Profile sections
t0 = time.perf_counter()
loss = model(x, y)
t1 = time.perf_counter()
loss.backward()
t2 = time.perf_counter()
optimizer.step()
t3 = time.perf_counter()

if step % 100 == 0 and rank == 0:
    print(f"Forward: {(t1-t0)*1000:.1f}ms, "
          f"Backward: {(t2-t1)*1000:.1f}ms, "
          f"Optimizer: {(t3-t2)*1000:.1f}ms")
```

---

### Issue: Quantization Gap Too Large

**Symptoms**: >0.03 BPB gap between fp32 and quantized

**Solutions**:
1. Enable Late QAT (reduces gap to ~0.01 BPB)
2. Try GPTQ-lite with more candidates
3. Use mixed precision (int5/int6/fp16)
4. Keep control tensors in FP16/FP32
5. Check quantization implementation for bugs

---

## Testing Workflow

### Step 1: Syntax and Smoke Test (Local)
```bash
# Syntax check
python3 -c "import ast; ast.parse(open('train_gpt.py').read())"

# 10-step smoke test
ITERATIONS=10 VAL_LOSS_EVERY=0 python3 train_gpt.py
```

### Step 2: Short Run Test (1xH100, ~900 steps)
```bash
# Quick 10-min run to verify training works
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
# Expect ~900-1100 steps on 1xH100
```

### Step 3: Full Run (8xH100, ~7000 steps)
```bash
# Competition-scale test
SEED=1337 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
# Expect ~6800-7200 steps on 8xH100
```

### Step 4: Multi-Seed Validation (3 seeds)
```bash
for seed in 1337 42 2025; do
  SEED=$seed torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${seed}.log
done

# Analyze variance
python3 -c "
import re
logs = ['train_seed1337.log', 'train_seed42.log', 'train_seed2025.log']
bpbs = []
for log in logs:
    with open(log) as f:
        for line in f:
            if 'final' in line and 'val_bpb' in line:
                bpb = float(re.search(r'val_bpb[:\s]+([\d.]+)', line).group(1))
                bpbs.append(bpb)
                break
import numpy as np
print(f'Mean: {np.mean(bpbs):.4f}')
print(f'Std: {np.std(bpbs, ddof=1):.4f}')
"
```

---

## Quick Reference: Common Hyperparameter Ranges

| Parameter | Baseline | SOTA Range | Notes |
|-----------|----------|------------|-------|
| NUM_LAYERS | 9 | 10-11 | More = better, but slower |
| MODEL_DIM | 512 | 512-768 | 768 needs int5/ternary to fit |
| MLP_MULT | 2 | 3-4 | Higher = more capacity |
| TRAIN_SEQ_LEN | 1024 | 1024-4096 | Longer = better eval, slower |
| BATCH_TOKENS | 524k | 524k-786k | Larger = fewer steps but better gradients |
| WARMDOWN_ITERS | 1200 | 3000-3500 | Longer = better convergence |
| MUON_MOMENTUM | 0.95 | 0.99 | Higher = smoother updates |
| MUON_WD | 0.0 | 0.04 | Weight decay improves quant+compress |
| MATRIX_LR | 0.04 | 0.02-0.025 | Tune with momentum |
| EMA_DECAY | N/A | 0.997-0.9985 | Higher = more smoothing |
| EVAL_STRIDE | 0 | 32-64 | Lower = better but slower |
| BIGRAM_VOCAB_SIZE | 0 | 1536-10240 | More buckets = fewer collisions |

---

## Parameter Sizing for 16MB Budget

**Rule of thumb**: ~20-27M parameters fit in 16MB with int6+zstd

**Budget breakdown**:
```
Tied embeddings (1024 vocab, 512 dim):    ~2.1M params
11 layers × (attn + MLP):                 ~18M params
BigramHash (2048×128 + proj):             ~0.5M params
Control tensors (scales, gates):          ~0.01M params
Total:                                    ~20.6M params

After int6 quantization:                  ~10.3MB
After zstd-22 compression:                ~6.8MB
Code size:                                ~60KB
Total artifact:                           ~6.9MB (margin for tuning)
```

**To fit more parameters**:
- Use int5 for MLPs: +30% more params
- Use ternary: +200% more params (but harder to train)
- Smaller vocabulary: 1024 vs 8192 saves ~4MB
- Remove embeddings: Use factored/shared embeddings

---

## Working Example: Adding a New Technique

**Goal**: Add learned per-layer LR multipliers

**Step 1 - Define the component**:
```python
class LayerwiseLRMultipliers(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        # Initialize to 1.0 (identity)
        self.multipliers = nn.Parameter(torch.ones(num_layers))

    def get_multiplier(self, layer_idx):
        return self.multipliers[layer_idx]
```

**Step 2 - Integrate into model**:
```python
class GPT(nn.Module):
    def __init__(self, args):
        # ... existing init ...
        if args.use_layerwise_lr:
            self.lr_mults = LayerwiseLRMultipliers(args.num_layers)
```

**Step 3 - Use in training loop**:
```python
# In main(), when setting up optimizer
param_groups = []
for layer_idx, block in enumerate(model.blocks):
    lr_mult = model.lr_mults.get_multiplier(layer_idx).item() \
              if args.use_layerwise_lr else 1.0
    param_groups.append({
        'params': block.parameters(),
        'lr': args.matrix_lr * lr_mult
    })
```

**Step 4 - Add config**:
```python
class Hyperparameters:
    use_layerwise_lr = bool(int(os.environ.get("USE_LAYERWISE_LR", "0")))
```

**Step 5 - Test**:
```bash
# Without feature
ITERATIONS=100 python3 train_gpt.py

# With feature
USE_LAYERWISE_LR=1 ITERATIONS=100 python3 train_gpt.py

# Compare losses to verify both work
```

---

## Optimization Tips

### Speed Optimizations

1. **torch.compile**: Essential, use `mode='default'`
2. **FlashAttention**: Required for seq_len > 512
3. **Fused operations**: torch.compile handles most automatically
4. **Reduce print statements**: Only log every 100-200 steps
5. **Pinned memory**: For data transfers

### Memory Optimizations

1. **Gradient accumulation**: If OOM
2. **Clear caches**: `torch.cuda.empty_cache()` after warmup
3. **BFloat16**: Use for activations
4. **Checkpoint gradients**: For very deep models

### Convergence Optimizations

1. **Longer warmdown**: 3500+ iterations
2. **EMA**: decay=0.997 for smoothing
3. **SWA**: Collect checkpoints during warmdown
4. **Gradient clipping**: 0.3 prevents instability
5. **Weight decay**: 0.04 improves generalization

---

## Related Skills

- Use `/parameter-golf` for challenge overview
- Use `/parameter-golf-techniques` for technique details and theory
- Use `/parameter-golf-verify` for testing and validation
- Use `/parameter-golf-submit` for submission process
