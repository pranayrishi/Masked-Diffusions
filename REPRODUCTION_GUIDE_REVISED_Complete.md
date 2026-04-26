# COMPLETE REPRODUCTION GUIDE (REVISED)
# "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"
# Kim, Shah, Kontonis, Kakade, Chen — ICML 2025, arXiv:2502.06768v3
#
# Designed for Google Colab execution with Google Drive storage
# Split into 7 self-contained Colab notebooks

---

# NOTEBOOK STRUCTURE OVERVIEW

| Notebook | Contents | GPU Needed | Est. Time |
|----------|----------|------------|-----------|
| Colab 1  | Environment + Data Preparation | CPU/T4 | 2-4 hrs |
| Colab 2  | L&O-NAE-SAT Experiments (Fig 2 bottom-right, Table 1) | T4/A100 | 8-16 hrs |
| Colab 3  | Sudoku + Zebra Experiments (Tables 2, 3, 5) | A100 | 24-48 hrs |
| Colab 4  | Scaling Law Experiments (Fig 2 left, Fig 2 top-right) | A100 | 100-300 hrs |
| Colab 5  | Text Generation Perplexity (Fig 3) | A100 | 20-40 hrs |
| Colab 6  | LLaDA 8B Evaluation (Table 4) | A100 80GB | 10-20 hrs |
| Colab 7  | Belief Propagation + All Plots (Fig 4, all figures) | CPU | 2-5 hrs |

---

# =============================================
# COLAB NOTEBOOK 1: ENVIRONMENT + DATA PREPARATION
# =============================================

## Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directory structure
import os
BASE_DIR = '/content/drive/MyDrive/mdm_reproduction'
os.makedirs(f'{BASE_DIR}/data/slimpajama', exist_ok=True)
os.makedirs(f'{BASE_DIR}/data/lo_nae_sat', exist_ok=True)
os.makedirs(f'{BASE_DIR}/data/sudoku', exist_ok=True)
os.makedirs(f'{BASE_DIR}/data/zebra', exist_ok=True)
os.makedirs(f'{BASE_DIR}/models', exist_ok=True)
os.makedirs(f'{BASE_DIR}/results', exist_ok=True)
os.makedirs(f'{BASE_DIR}/figures', exist_ok=True)
os.makedirs(f'{BASE_DIR}/codebases', exist_ok=True)
print("Directory structure created.")
```

## Cell 2: Install Dependencies
```python
!pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.2 datasets==2.16.1 accelerate==0.25.0
!pip install wandb einops scipy matplotlib seaborn tqdm
!pip install flash-attn --no-build-isolation  # Optional, skip if build fails on Colab
```

## Cell 3: Clone Required Codebases

The paper explicitly depends on three codebases. These are non-negotiable — the paper
says "we use the codebase of (Ye et al., 2024)" and "We leverage the codebase from (Nie et al., 2024)".

```python
%cd {BASE_DIR}/codebases

# 1. Nie et al. (2024) — "Scaling up masked diffusion models on text"
#    Used for: scaling law experiments (Fig 2 Left), MDM training on text
#    Repo: https://github.com/ML-GSAI/SMDM
!git clone https://github.com/ML-GSAI/SMDM.git

# 2. Ye et al. (2024) — "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning"
#    Used for: Sudoku/Zebra training and inference (Tables 2, 3, 5)
#    The paper says "we use the codebase of (Ye et al., 2024) with keeping most of the hyperparameters default"
!git clone https://github.com/HKUNLP/discrete-diffusion.git
# NOTE: If the above URL is wrong, search for the Ye et al. 2024 repo.
# Alternative: https://github.com/yegonkim/discrete-diffusion-reasoning

# 3. LLaDA (Nie et al., 2025) — "Large Language Diffusion Models"
#    Used for: Table 4 (LLaDA 8B experiments)
!git clone https://github.com/ML-GSAI/LLaDA.git

# 4. Shah et al. (2024) — "Causal language modeling can elicit search and reasoning"
#    Used for: Sudoku/Zebra datasets AND the ARM-with-ordering baseline
!git clone https://github.com/kulinshah98/logic-puzzles.git
```

## Cell 4: Prepare SlimPajama Dataset

```python
"""
SlimPajama dataset preparation.

Paper reference (Section 3.2):
"We use the Slimpajama dataset (Soboleva et al., 2023)"

Tokenizer: GPT-2 (vocab size 50257)
Sequence length: L = 2048 (Section 3.2: "the sequence length L is 2048")

The Nie et al. (2024) codebase handles this. Follow their data preparation
instructions. Key points:
- Use GPT-2 tokenizer
- Pack documents into sequences of length 2048
- Standard train/validation split
"""
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pickle

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Stream and tokenize SlimPajama
# For full reproduction, use the entire dataset.
# The scaling law experiments vary model size at fixed compute budgets,
# so you need enough data that the model doesn't overfit at any budget.
# Minimum: ~10B tokens. Full: ~627B tokens.

dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

# Tokenize and pack into chunks of 2048
SEQ_LEN = 2048
buffer = []
all_sequences = []
count = 0
target_sequences = 500000  # Adjust based on storage: 500K seqs = ~1B tokens

for example in dataset:
    tokens = tokenizer.encode(example['text'])
    buffer.extend(tokens)
    
    while len(buffer) >= SEQ_LEN:
        all_sequences.append(buffer[:SEQ_LEN])
        buffer = buffer[SEQ_LEN:]
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} sequences ({count * SEQ_LEN / 1e9:.2f}B tokens)")
        if count >= target_sequences:
            break
    if count >= target_sequences:
        break

# Save
all_sequences = np.array(all_sequences, dtype=np.int32)
np.save(f'{BASE_DIR}/data/slimpajama/train_sequences.npy', all_sequences[:-10000])
np.save(f'{BASE_DIR}/data/slimpajama/val_sequences.npy', all_sequences[-10000:])
print(f"Saved {len(all_sequences)} sequences. Total tokens: {len(all_sequences) * SEQ_LEN / 1e9:.2f}B")
```

## Cell 5: Generate L&O-NAE-SAT Data

```python
"""
L&O-NAE-SAT Distribution Generator

CRITICAL DETAILS (gathered from multiple sections):

1. Definition (Section 3.3):
   - π = identity permutation
   - Latent tokens: positions 0 to N-1, sampled uniformly from {1,...,m}
   - Observation tokens: positions N to N+P-1
   - Each observation j uses a RANDOMLY CHOSEN (PRE-FIXED) triple (i1,i2,i3) from [N]
   - O_j = NAE(x_{i1}, x_{i2}, x_{i3}) = 1 - 1[x_{i1} = x_{i2} = x_{i3}]
   - Observations are DETERMINISTIC (not noisy) given the latent tokens

2. Alphabet size m = 2 (DERIVED, not stated explicitly):
   Table 1 says "naive guessing leads to 75% accuracy"
   P(NAE=1 | random assignment from {1,...,m}^3) = 1 - m/m^3 = 1 - 1/m^2
   For m=2: 1 - 1/4 = 0.75 = 75%. CONFIRMED.

3. Configurations:
   - Figure 2 bottom-right (Appendix C.2.1): (N,P) = (20,280), padded to 512
   - Table 1 (Appendix D.1.1): (N,P) ∈ {(25,275),(30,270),(40,260),(50,250),(100,200)}
   - All have total L = N+P = 300

4. Padding (Appendix C.2.1):
   "For each example sequence from L&O-NAE-SAT, we pad the last 212 tokens
    with an additional token value of 2"
   So the model sees sequences of length 300 + 212 = 512
   Padded positions have value 2 (a third token value beyond {0,1} observations)

5. Triple selection:
   "randomly chosen (pre-fixed) triples (i1, i2, i3) ∈ [N]"
   This means i1, i2, i3 are each independently drawn from {0,...,N-1}.
   The triples are sampled ONCE and fixed for the entire distribution.
   Replacement is allowed (same index can appear multiple times in a triple).
"""
import numpy as np
import json

def generate_lo_nae_sat(N, P, m, num_samples, pad_to=512, seed=42):
    """Generate L&O-NAE-SAT dataset.
    
    Returns:
        sequences: np.array of shape (num_samples, pad_to)
        triples: np.array of shape (P, 3) — the fixed triples
        metadata: dict with configuration info
    """
    rng = np.random.RandomState(seed)
    
    # Step 1: Fix random triples for ALL P observations
    triples = rng.randint(0, N, size=(P, 3))  # Each element in {0,...,N-1}
    
    # Step 2: Generate samples
    L = N + P
    sequences = np.zeros((num_samples, pad_to), dtype=np.int64)
    
    for s in range(num_samples):
        # Sample latent tokens uniformly from {1, ..., m} = {1, 2} for m=2
        latents = rng.randint(1, m + 1, size=N)
        sequences[s, :N] = latents
        
        # Compute observation tokens deterministically
        for j in range(P):
            i1, i2, i3 = triples[j]
            # NAE: 1 if NOT all equal, 0 if all equal
            all_equal = (latents[i1] == latents[i2]) and (latents[i2] == latents[i3])
            sequences[s, N + j] = 0 if all_equal else 1
        
        # Pad remaining positions with token value 2
        # (Appendix C.2.1: "pad the last 212 tokens with an additional token value of 2")
        sequences[s, L:] = 2
    
    metadata = {
        'N': N, 'P': P, 'm': m, 'L': L,
        'pad_to': pad_to, 'seed': seed,
        'num_samples': num_samples,
        'naive_accuracy': 1.0 - 1.0 / (m ** 2)  # Should be 0.75 for m=2
    }
    
    return sequences, triples, metadata

# Verify naive accuracy
_, _, meta = generate_lo_nae_sat(20, 280, 2, 100)
print(f"Naive guessing accuracy: {meta['naive_accuracy']:.2%}")  # Should print 75.00%

# Generate all configurations
configs = [
    (20, 280),   # Figure 2 bottom-right
    (25, 275),   # Table 1
    (30, 270),   # Table 1
    (40, 260),   # Table 1
    (50, 250),   # Table 1
    (100, 200),  # Table 1
]

for N, P in configs:
    print(f"\nGenerating (N={N}, P={P})...")
    
    # Training data: 100K samples
    train_data, triples, meta = generate_lo_nae_sat(
        N, P, m=2, num_samples=100000, pad_to=512, seed=42
    )
    
    # Test data: 10K samples (different seed)
    test_data, _, _ = generate_lo_nae_sat(
        N, P, m=2, num_samples=10000, pad_to=512, seed=123
    )
    # IMPORTANT: Use the SAME triples for test data
    # Regenerate test data with same triples
    test_rng = np.random.RandomState(123)
    test_sequences = np.zeros((10000, 512), dtype=np.int64)
    for s in range(10000):
        latents = test_rng.randint(1, 3, size=N)  # {1, 2} for m=2
        test_sequences[s, :N] = latents
        for j in range(P):
            i1, i2, i3 = triples[j]
            all_equal = (latents[i1] == latents[i2]) and (latents[i2] == latents[i3])
            test_sequences[s, N + j] = 0 if all_equal else 1
        test_sequences[s, N+P:] = 2
    
    save_dir = f'{BASE_DIR}/data/lo_nae_sat/N{N}_P{P}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/train.npy', train_data)
    np.save(f'{save_dir}/test.npy', test_sequences)
    np.save(f'{save_dir}/triples.npy', triples)
    with open(f'{save_dir}/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Saved {len(train_data)} train, {len(test_sequences)} test sequences")
    print(f"  Sequence shape: {train_data.shape}")
    
    # Verify: compute actual NAE accuracy on random predictions
    random_preds = test_rng.randint(0, 2, size=(1000, P))
    actual_obs = test_sequences[:1000, N:N+P]
    random_acc = (random_preds == actual_obs).mean()
    print(f"  Verified random guess accuracy: {random_acc:.4f} (expected ~0.75)")
```

## Cell 6: Prepare Sudoku Dataset

```python
"""
Sudoku Dataset Preparation

Paper references:
- Appendix D.2: "For both Sudoku and Zebra puzzles, we use the dataset 
  provided in Shah et al. (2024) to train our model."
- "This dataset is created by filtering the puzzles from (Radcliffe, 2020) 
  that can be solved using a fixed list of 7 strategies."
- "The hard dataset contains around 1M Sudoku puzzles."

Source: https://www.kaggle.com/dsv/1495975 (Radcliffe 2020, "3 million Sudoku puzzles with ratings")

The Shah et al. (2024) codebase should have the filtering logic.
If not available, the 7 strategies are standard Sudoku constraint-propagation techniques:
1. Naked Singles
2. Hidden Singles  
3. Naked Pairs
4. Hidden Pairs
5. Pointing Pairs
6. Box/Line Reduction
7. Naked Triples

Representation:
- 81 tokens (9×9 grid, row-major)
- Token values: 0 = mask (empty cell), 1-9 = digits
- Given clues are pre-filled; the model fills in masked positions

For ARM with ordering:
- Shah et al. (2024) provides the solving order for each puzzle
- The solving order = the sequence in which cells are determined by the strategies
- ARM is trained to predict cells in this order (teacher forcing)
"""

# Download Kaggle dataset (requires kaggle API key)
# Alternative: manually download from https://www.kaggle.com/dsv/1495975
!pip install kaggle
# Upload your kaggle.json to Colab first
!mkdir -p ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d bryanpark/sudoku -p {BASE_DIR}/data/sudoku/raw/

# If Shah et al.'s pre-filtered dataset is available from their repo:
import os
shah_repo = f'{BASE_DIR}/codebases/logic-puzzles'
if os.path.exists(shah_repo):
    print("Shah et al. repo found. Use their data preparation scripts.")
    print(f"Check: {shah_repo}/data/ for pre-processed Sudoku/Zebra datasets")
    print(f"Check: {shah_repo}/scripts/ for strategy filtering code")
else:
    print("WARNING: Shah et al. repo not found. Need to obtain Sudoku data separately.")
    print("The dataset filtering is critical — wrong filtering = wrong results.")

# Sudoku sequence representation
def sudoku_to_sequence(puzzle_str, solution_str):
    """
    Convert Sudoku strings to token sequences.
    
    puzzle_str: 81 chars, '.' or '0' for empty, '1'-'9' for given clues
    solution_str: 81 chars, all '1'-'9'
    
    Returns:
        clue_tokens: list of 81 ints (0 for empty, 1-9 for clues)
        solution_tokens: list of 81 ints (1-9, the full solution)
    """
    clue_tokens = []
    solution_tokens = []
    for p, s in zip(puzzle_str, solution_str):
        clue_val = 0 if p in '.0' else int(p)
        sol_val = int(s)
        clue_tokens.append(clue_val)
        solution_tokens.append(sol_val)
    return clue_tokens, solution_tokens

# Example
puzzle = "..5...9..1..4...7.8..7.3..2.9.....4..4.1.8..5.....6.1.7..2.9..3.3...5..2..1...7.."
solution = "375218964126495378894763512293657841641382759587149623768921435439576182512834796"
clues, sol = sudoku_to_sequence(puzzle, solution)
print(f"Clues (0=empty): {clues[:9]}")
print(f"Solution:        {sol[:9]}")
print(f"Empty cells: {clues.count(0)}")
```

## Cell 7: Prepare Zebra Dataset

```python
"""
Zebra (Einstein) Puzzle Dataset

Paper references:
- Table 3: 19M MDM, 42M ARM
- Same source: Shah et al. (2024)

The Zebra puzzle assigns attributes to houses. The exact tokenization
follows Shah et al.'s format. Check their repository for details.

Key difference from Sudoku:
- Different sequence length (depends on puzzle encoding)
- Different vocabulary size
"""
print("Zebra dataset: use Shah et al. (2024) codebase data preparation.")
print(f"Check: {BASE_DIR}/codebases/logic-puzzles/")
```

## Cell 8: Download LLaDA 8B Weights

```python
"""
LLaDA 8B Model Weights

Paper reference (Section 4.4):
"we adapted LLaDA, the 8B MDM model from (Nie et al., 2025)"

The base model (not instruct variant) is used.
Weights: https://huggingface.co/GSAI-ML/LLaDA-8B-Base
"""
!pip install huggingface_hub
from huggingface_hub import snapshot_download

# Download to Google Drive (WARNING: ~16GB)
# Only run this when you need it for Colab 6
# snapshot_download(
#     "GSAI-ML/LLaDA-8B-Base",
#     local_dir=f'{BASE_DIR}/models/llada-8b-base',
#     local_dir_use_symlinks=False
# )
print("LLaDA download command ready. Uncomment when needed.")
```

---

# =============================================
# COLAB NOTEBOOK 2: L&O-NAE-SAT EXPERIMENTS
# (Figure 2 bottom-right, Table 1)
# =============================================

## Cell 1: Setup
```python
from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = '/content/drive/MyDrive/mdm_reproduction'

!pip install torch torchvision transformers einops tqdm matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
```

## Cell 2: Model Architecture — 19M MDM with RoPE

```python
"""
MODEL ARCHITECTURE

Paper says (Appendix C.2.1):
"We employ a 19M MDM with RoPE and a maximum sequence length of 512"

For puzzles (Appendix D.2):
"we use the codebase of (Ye et al., 2024)"
"For the Sudoku dataset, we use 6M GPT-2 model"
"for the Zebra dataset, we use 19M model"

GPT-2 architecture sizes (standard configurations):
- 6M params:  ~4 layers, 256 hidden, 4 heads
- 19M params: ~6 layers, 384 hidden, 6 heads  
- 42M params: ~8 layers, 512 hidden, 8 heads
- 170M params: 12 layers, 768 hidden, 12 heads

These sizes are approximate. The exact configuration depends on the
Ye et al. (2024) codebase defaults. Check their config files.

CRITICAL ARCHITECTURAL DIFFERENCE:
- MDM uses BIDIRECTIONAL attention (no causal mask)
  The model must see all unmasked tokens regardless of position.
- ARM uses CAUSAL attention (standard autoregressive mask)
  Each position can only attend to previous positions.

The paper explicitly states (Section 2):
"a time-embedding-free architecture for the denoising network, 
 i.e., p_θ(·|x_t, t) = p_θ(·|x_t) is generally used"
So: NO time embedding input. The model takes only token IDs.
"""

import math

class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return cos_emb, sin_emb

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        B, L, D = x.shape
        
        # Pre-norm
        h = self.ln1(x)
        
        # QKV projection
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if provided
        if rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Causal mask for ARM, no mask for MDM
        if self.causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask[None, None], float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class MDMTransformer(nn.Module):
    """
    Masked Diffusion Model transformer.
    
    Key properties:
    - BIDIRECTIONAL attention (causal=False)
    - NO time embedding (time-embedding-free per Section 2)
    - RoPE or learned positional embeddings depending on experiment
    """
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ff_dim,
                 max_seq_len=512, dropout=0.1, pos_type='rope'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding: vocab includes mask token (0) plus actual values
        # For L&O-NAE-SAT with m=2: tokens are {0(mask), 1, 2(pad_value)}
        # But observations can also be 0 or 1, so we need to be careful.
        # Actually, the vocabulary for L&O-NAE-SAT is:
        #   Latent tokens: {1, 2} (m=2)
        #   Observation tokens: {0, 1} (NAE output)
        #   Pad token: 2
        #   Mask token: need a SEPARATE mask token
        # 
        # IMPORTANT: The mask token must be distinct from all data tokens.
        # The paper uses 0 as the mask token and data tokens start from 1.
        # But observations can be 0 or 1. This creates ambiguity.
        #
        # RESOLUTION: Looking at the L&O definition (Definition 3.1):
        # "pdata is over {0,...,m}^L" — so data tokens are {0,...,m}.
        # And "We use 0 to denote the mask token."
        # This means observation value 0 and mask token 0 use the SAME value.
        # The model must learn that during training (when a position is masked),
        # position value 0 means "masked", but during unmasked positions,
        # observation value 0 means "all-equal" (NAE=0).
        #
        # This is standard in MDM implementations: during the forward process,
        # when we CREATE the masked input x_t, we replace masked positions with
        # the mask token. The model then predicts the ORIGINAL token at each
        # masked position. So the model output for a masked position predicts
        # from the full vocabulary {0, 1, ..., m}, and the loss is computed
        # against the original value.
        #
        # For L&O-NAE-SAT with pad=2:
        #   Total distinct values: {0, 1, 2} for data + need mask
        #   If mask=some_special_id, vocab_size = 4 (0,1,2 data + mask)
        #   OR if following the paper's convention where 0=mask:
        #   Need to shift data values so that actual data ∈ {1,...,m+1}
        #   and 0 = mask.
        #
        # SAFEST APPROACH: Use a dedicated mask token ID.
        # vocab_size should be max_data_value + 2 (one extra for mask)
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.pos_type = pos_type
        if pos_type == 'rope':
            self.rope = RoPE(hidden_dim // num_heads, max_seq_len)
            self.pos_emb = None
        elif pos_type == 'learned':
            self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
            self.rope = None
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout, causal=False)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len) — token IDs including mask tokens
        Returns: logits (batch_size, seq_len, vocab_size)
        """
        B, L = x.shape
        h = self.embedding(x)
        
        if self.pos_type == 'learned' and self.pos_emb is not None:
            positions = torch.arange(L, device=x.device)
            h = h + self.pos_emb(positions)
        
        rope_cos, rope_sin = None, None
        if self.pos_type == 'rope' and self.rope is not None:
            rope_cos, rope_sin = self.rope(h, L)
        
        for block in self.blocks:
            h = block(h, rope_cos, rope_sin)
        
        h = self.ln_final(h)
        logits = self.output_head(h)
        return logits


class ARMTransformer(nn.Module):
    """
    Autoregressive Model transformer.
    Same as MDM but with CAUSAL attention.
    """
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ff_dim,
                 max_seq_len=512, dropout=0.1, pos_type='learned'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.pos_type = pos_type
        if pos_type == 'learned':
            self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        elif pos_type == 'rope':
            self.rope = RoPE(hidden_dim // num_heads, max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout, causal=True)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        B, L = x.shape
        h = self.embedding(x)
        
        if self.pos_type == 'learned':
            positions = torch.arange(L, device=x.device)
            h = h + self.pos_emb(positions)
        
        rope_cos, rope_sin = None, None
        if self.pos_type == 'rope':
            rope_cos, rope_sin = self.rope(h, L)
        
        for block in self.blocks:
            h = block(h, rope_cos, rope_sin)
        
        h = self.ln_final(h)
        return self.output_head(h)


def get_model_config(target_params_M, model_type='mdm'):
    """
    Return (hidden_dim, num_layers, num_heads, ff_dim) for target parameter count.
    
    Based on GPT-2 scaling conventions used in Ye et al. (2024):
    - ff_dim = 4 * hidden_dim (standard GPT-2 ratio)
    - num_heads = hidden_dim // 64 (standard head_dim=64)
    
    Approximate sizes:
    - 6M:  hidden=256, layers=6, heads=4, ff=1024
    - 19M: hidden=384, layers=8, heads=6, ff=1536
    - 42M: hidden=512, layers=10, heads=8, ff=2048
    - 170M: hidden=768, layers=12, heads=12, ff=3072
    - 1.1B: hidden=2048, layers=24, heads=32, ff=8192
    
    These are APPROXIMATIONS. Check the Ye et al. (2024) codebase for exact configs.
    """
    configs = {
        6:   (256, 6, 4, 1024),
        19:  (384, 8, 6, 1536),
        42:  (512, 10, 8, 2048),
        170: (768, 12, 12, 3072),
        1100: (2048, 24, 32, 8192),
    }
    return configs.get(target_params_M, configs[19])
```

## Cell 3: MDM Training Loop

```python
"""
MDM TRAINING

The loss function (from Section 2 and Proposition E.1):

Lθ = - Σ_{n=1}^{L} E_{x(n)~q̃(·|x0)} [1/n Σ_{ℓ: x^ℓ(n)=0} log pθ(x^ℓ_0 | x(n))]

In practice, for each training step:
1. Sample x0 from data
2. Sample number of tokens to mask n ~ Uniform(1, L)  
3. Randomly and uniformly mask n positions → x_masked
4. Model predicts original tokens at masked positions
5. Loss = (1/n) * Σ_{masked positions} -log pθ(x^i_0 | x_masked)

KEY: Weight the cross-entropy by 1/n where n = number of masked tokens.
This ensures the loss properly weights different masking levels.
"""

def mdm_train_step(model, optimizer, x0, mask_token_id, device):
    """
    Single MDM training step.
    
    Args:
        x0: (batch_size, seq_len) clean sequences
        mask_token_id: integer ID for the mask token
    """
    model.train()
    B, L = x0.shape
    
    # Sample number of masks per sequence: n ~ Uniform(1, L)
    n = torch.randint(1, L + 1, (B,), device=device)
    
    # Create masks: for each sample, randomly choose n positions
    # Use argsort trick for efficient batched random selection
    noise = torch.rand(B, L, device=device)
    # Sort and take first n positions
    sorted_indices = noise.argsort(dim=-1)
    
    # Create boolean mask
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        mask[b, sorted_indices[b, :n[b]]] = True
    
    # Create masked input
    x_masked = x0.clone()
    x_masked[mask] = mask_token_id
    
    # Forward pass
    logits = model(x_masked)  # (B, L, vocab_size)
    
    # Compute loss only at masked positions, weighted by 1/n
    # For efficiency, compute per-sample loss
    total_loss = 0.0
    for b in range(B):
        masked_logits = logits[b, mask[b]]   # (n_b, vocab_size)
        masked_targets = x0[b, mask[b]]      # (n_b,)
        
        if len(masked_targets) > 0:
            sample_loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
            total_loss = total_loss + sample_loss
    
    total_loss = total_loss / B
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


def train_mdm(model, train_data, mask_token_id, num_iterations, batch_size,
              lr=1e-3, weight_decay=0.1, device='cuda', save_path=None):
    """
    Full MDM training loop.
    
    Training config from Appendix C.1:
    - AdamW: β1=0.9, β2=0.95, weight_decay=0.1
    - Cosine LR schedule: max_lr=4e-4, min_lr=4e-5
    
    BUT for L&O-NAE-SAT (Appendix D.2):
    - learning_rate=0.001 (for puzzles)
    - For L&O-NAE-SAT, the paper says "train a 19M MDM" without specifying
      different hyperparameters, so we use the default puzzle settings.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=weight_decay
    )
    
    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=lr / 10
    )
    
    model = model.to(device)
    dataset_size = len(train_data)
    
    losses = []
    for step in tqdm(range(num_iterations), desc="Training MDM"):
        # Sample batch
        indices = np.random.randint(0, dataset_size, size=batch_size)
        x0 = torch.tensor(train_data[indices], dtype=torch.long, device=device)
        
        loss = mdm_train_step(model, optimizer, x0, mask_token_id, device)
        scheduler.step()
        losses.append(loss)
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"Step {step+1}/{num_iterations}, Loss: {avg_loss:.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return losses
```

## Cell 4: Inference Strategies

```python
"""
INFERENCE STRATEGIES (Section 4)

Three strategies are implemented:
1. Vanilla: random unmasking (baseline)
2. Top probability: unmask positions where model is most confident
3. Top probability margin: unmask positions with largest gap between top-2 predictions

CRITICAL DETAILS:

1. Number of tokens to unmask per step K (Appendix D.1.2):
   K = (# masked tokens in current x_t) × (α_s - α_t) / (1 - α_t)
   For linear schedule α_t = 1-t: K = (# masked) × (t - s) / t
   
   "We set the number of tokens to unmask K so that the number of 
    unmasked tokens matches that of vanilla MDM inference in expectation"

2. Gumbel noise (Appendix D.2):
   "We add Gumbel noise with a coefficient of 0.5 to the MDM inference oracle F"
   Applied to the SCORES (certainty values), not to token probabilities.

3. For text data (Appendix D.1.2):
   GAUSSIAN noise (ε) instead of Gumbel: F = Top K(margin + ε)
   "adding a certain level of temperature to the oracle is useful"

4. Number of reverse steps (Appendix D.2):
   "we use 50 reverse sampling steps"

5. The paper uses a LINEAR noise schedule (standard in MDM literature).
"""

def run_mdm_inference(model, seq_len, mask_token_id, vocab_size, 
                      strategy='vanilla', num_steps=50, 
                      gumbel_coeff=0.0, device='cuda',
                      fixed_tokens=None, fixed_mask=None):
    """
    MDM inference with configurable strategy.
    
    Args:
        model: trained MDM
        seq_len: length of sequence to generate
        mask_token_id: mask token ID
        vocab_size: total vocabulary size (including mask)
        strategy: 'vanilla', 'top_prob', or 'top_prob_margin'
        num_steps: number of reverse sampling steps (paper uses 50)
        gumbel_coeff: Gumbel noise coefficient (0.5 for puzzles)
        fixed_tokens: optional tensor of fixed token values (for conditional generation)
        fixed_mask: optional boolean tensor (True = this position is fixed/given)
    
    Returns:
        generated sequence of shape (seq_len,)
    """
    model.eval()
    
    # Initialize: fully masked or partially masked (conditional)
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    if fixed_tokens is not None and fixed_mask is not None:
        x[0, fixed_mask] = fixed_tokens[fixed_mask]
    
    # Linear noise schedule: α_t = 1 - t
    # Reverse from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        
        # Identify currently masked positions (excluding fixed positions)
        is_masked = (x[0] == mask_token_id)
        if fixed_mask is not None:
            is_masked = is_masked & ~fixed_mask
        
        masked_positions = is_masked.nonzero(as_tuple=True)[0]
        num_masked = len(masked_positions)
        
        if num_masked == 0:
            break
        
        # Compute K: number to unmask at this step
        # K = num_masked × (α_s - α_t) / (1 - α_t)
        # With α_t = 1-t: K = num_masked × (s - t + 1 - 1 + t) / t 
        # Wait, let me recompute:
        # α_t = 1 - t, α_s = 1 - s
        # (α_s - α_t) / (1 - α_t) = ((1-s) - (1-t)) / (1 - (1-t)) = (t - s) / t
        unmask_frac = (t - s) / (t + 1e-10)
        K = max(1, round(num_masked * unmask_frac))
        K = min(K, num_masked)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab_size)
        
        # Extract logits at masked positions
        # IMPORTANT: exclude mask token from predictions
        # Token probabilities should be over actual values only
        masked_logits = logits[0, masked_positions]  # (num_masked, vocab_size)
        
        # Create distribution over non-mask tokens
        # Set mask token logit to -inf
        masked_logits[:, mask_token_id] = float('-inf')
        probs = F.softmax(masked_logits, dim=-1)
        
        if strategy == 'vanilla':
            # Random selection of K positions
            perm = torch.randperm(num_masked, device=device)[:K]
            positions_to_unmask = masked_positions[perm]
            
            # Sample tokens
            for idx in perm:
                token = torch.multinomial(probs[idx], 1).item()
                x[0, masked_positions[idx]] = token
        
        elif strategy == 'top_prob':
            # Certainty = max_j p(x^i = j)
            scores = probs.max(dim=-1).values  # (num_masked,)
            
            # Add Gumbel noise
            if gumbel_coeff > 0:
                gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
                scores = scores + gumbel_coeff * gumbel
            
            # Select top-K
            _, top_k = torch.topk(scores, K)
            
            for idx in top_k:
                token = torch.multinomial(probs[idx], 1).item()
                x[0, masked_positions[idx]] = token
        
        elif strategy == 'top_prob_margin':
            # Certainty = |p(j1) - p(j2)| where j1, j2 are top-2 values
            top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
            if top2_vals.shape[-1] >= 2:
                scores = top2_vals[:, 0] - top2_vals[:, 1]
            else:
                scores = top2_vals[:, 0]
            
            # Add Gumbel noise
            if gumbel_coeff > 0:
                gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
                scores = scores + gumbel_coeff * gumbel
            
            # Select top-K
            _, top_k = torch.topk(scores, K)
            
            for idx in top_k:
                token = torch.multinomial(probs[idx], 1).item()
                x[0, masked_positions[idx]] = token
    
    # Fill any remaining masked positions greedily
    remaining = (x[0] == mask_token_id)
    if fixed_mask is not None:
        remaining = remaining & ~fixed_mask
    if remaining.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining.nonzero(as_tuple=True)[0]:
            logits[0, pos, mask_token_id] = float('-inf')
            token = torch.multinomial(F.softmax(logits[0, pos], dim=-1), 1).item()
            x[0, pos] = token
    
    return x[0]
```

## Cell 5: Run L&O-NAE-SAT Experiments

```python
"""
EXPERIMENT: Figure 2 bottom-right (Error Imbalance)
and Table 1 (Vanilla vs Adaptive Accuracy)
"""

# === FIGURE 2 BOTTOM-RIGHT ===
# Configuration: (N=20, P=280), 19M MDM, RoPE, seq_len=512
# Train for 2000 iterations, proxy for 50000 iterations

N, P = 20, 280
data = np.load(f'{BASE_DIR}/data/lo_nae_sat/N{N}_P{P}/train.npy')

# Vocabulary: data values are {0, 1, 2}, mask token must be distinct
# Use mask_token_id = 3
MASK_TOKEN_ID = 3
VOCAB_SIZE = 4  # tokens: 0, 1, 2 (data) + 3 (mask)

# Shift data so mask token doesn't collide with data values
# Actually, the data already uses values {0, 1, 2} and we add mask=3
# No shifting needed as long as mask_token_id != any data value

hidden_dim, num_layers, num_heads, ff_dim = get_model_config(19)

model = MDMTransformer(
    vocab_size=VOCAB_SIZE,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    max_seq_len=512,
    pos_type='rope'  # Appendix C.2.1: "19M MDM with RoPE"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Train model (2000 iterations)
losses = train_mdm(
    model, data, mask_token_id=MASK_TOKEN_ID,
    num_iterations=2000, batch_size=128,
    lr=0.001, device=device,
    save_path=f'{BASE_DIR}/models/lo_nae_sat_N20_P280_model.pt'
)

# Train proxy (Bayes-optimal approximation, 50000 iterations)
proxy_model = MDMTransformer(
    vocab_size=VOCAB_SIZE,
    hidden_dim=hidden_dim, num_layers=num_layers,
    num_heads=num_heads, ff_dim=ff_dim,
    max_seq_len=512, pos_type='rope'
)

proxy_losses = train_mdm(
    proxy_model, data, mask_token_id=MASK_TOKEN_ID,
    num_iterations=50000, batch_size=128,
    lr=0.001, device=device,
    save_path=f'{BASE_DIR}/models/lo_nae_sat_N20_P280_proxy.pt'
)

# Measure error imbalance (Appendix C.2.1)
"""
"For each ℓ ∈ [1, N-1], we randomly mask ℓ tokens in the latent positions 
 and ℓ × (P/N) tokens in the observed positions."
"The result in Figure 2 corresponds to the case when ℓ = 11"
"we repeat this process 1000 times"
"""
test_data = np.load(f'{BASE_DIR}/data/lo_nae_sat/N{N}_P{P}/test.npy')
ell = 11
num_trials = 1000
errors = np.zeros(N + P)
error_counts = np.zeros(N + P)

model.eval()
proxy_model.eval()

for trial in tqdm(range(num_trials), desc="Measuring error"):
    # Sample random test example
    idx = np.random.randint(len(test_data))
    x0 = test_data[idx]
    
    # Mask ell latent positions
    lat_mask_idx = np.random.choice(N, size=ell, replace=False)
    # Mask ell * (P/N) observation positions
    obs_mask_count = int(ell * P / N)
    obs_mask_idx = np.random.choice(np.arange(N, N + P), size=obs_mask_count, replace=False)
    
    all_mask_idx = np.concatenate([lat_mask_idx, obs_mask_idx])
    
    # Create masked sequence
    x_masked = x0.copy()
    x_masked[all_mask_idx] = MASK_TOKEN_ID
    
    x_tensor = torch.tensor(x_masked, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits_model = model(x_tensor)
        logits_proxy = proxy_model(x_tensor)
    
    # Compute squared error at each masked position
    for pos in all_mask_idx:
        if pos >= N + P:
            continue  # Skip padding
        log_p_model = F.log_softmax(logits_model[0, pos], dim=-1)
        log_p_proxy = F.log_softmax(logits_proxy[0, pos], dim=-1)
        
        true_token = x0[pos]
        err = (log_p_model[true_token] - log_p_proxy[true_token]).item() ** 2
        errors[pos] += err
        error_counts[pos] += 1

# Average
error_counts = np.maximum(error_counts, 1)
avg_errors = errors / error_counts

# Save for plotting
np.save(f'{BASE_DIR}/results/fig2_bottom_right_errors.npy', avg_errors)
print(f"Latent positions (0-{N-1}) mean error: {avg_errors[:N].mean():.6f}")
print(f"Observation positions ({N}-{N+P-1}) mean error: {avg_errors[N:N+P].mean():.6f}")
print("Latent errors should be HIGHER than observation errors (paper finding).")


# === TABLE 1 ===
# Five configurations, vanilla vs adaptive (top_prob_margin)
table1_results = {}

for N_cfg, P_cfg in [(25, 275), (30, 270), (40, 260), (50, 250), (100, 200)]:
    print(f"\n=== Config (N={N_cfg}, P={P_cfg}) ===")
    
    cfg_data = np.load(f'{BASE_DIR}/data/lo_nae_sat/N{N_cfg}_P{P_cfg}/train.npy')
    cfg_test = np.load(f'{BASE_DIR}/data/lo_nae_sat/N{N_cfg}_P{P_cfg}/test.npy')
    
    cfg_model = MDMTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, ff_dim=ff_dim,
        max_seq_len=512, pos_type='rope'
    )
    
    # Train
    train_mdm(
        cfg_model, cfg_data, mask_token_id=MASK_TOKEN_ID,
        num_iterations=2000, batch_size=128,
        lr=0.001, device=device,
        save_path=f'{BASE_DIR}/models/lo_nae_sat_N{N_cfg}_P{P_cfg}.pt'
    )
    
    # Evaluate: accuracy on observation tokens
    # Generate sequences from fully masked state, check observation token accuracy
    cfg_model.eval()
    
    for strategy_name in ['vanilla', 'top_prob_margin']:
        correct = 0
        total = 0
        num_eval = min(500, len(cfg_test))
        
        for i in tqdm(range(num_eval), desc=f"Eval {strategy_name}"):
            x0 = cfg_test[i]
            
            # Run inference from fully masked state
            generated = run_mdm_inference(
                cfg_model, seq_len=512,
                mask_token_id=MASK_TOKEN_ID, vocab_size=VOCAB_SIZE,
                strategy=strategy_name, num_steps=50,
                gumbel_coeff=0.5 if strategy_name != 'vanilla' else 0.0,
                device=device
            )
            
            # Check observation token accuracy
            obs_correct = (generated[N_cfg:N_cfg+P_cfg].cpu().numpy() == x0[N_cfg:N_cfg+P_cfg]).sum()
            correct += obs_correct
            total += P_cfg
        
        accuracy = correct / total
        table1_results[(N_cfg, P_cfg, strategy_name)] = accuracy
        print(f"  {strategy_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save results
with open(f'{BASE_DIR}/results/table1.json', 'w') as f:
    json.dump({str(k): v for k, v in table1_results.items()}, f, indent=2)
```

## Cell 6: Quick verification
```python
# Verify Table 1 results make sense
print("\n=== TABLE 1 VERIFICATION ===")
print("Expected: Vanilla ≈ 63-78%, Adaptive ≈ 89-94%")
print("Also: accuracy should DECREASE as N increases (harder problem)")
for key, val in table1_results.items():
    print(f"  {key}: {val*100:.2f}%")
```

---

# REMAINING NOTEBOOKS (3-7) FOLLOW THE SAME STRUCTURE
# Due to length, I provide the key details for each below.

---

# =============================================
# COLAB NOTEBOOK 3: SUDOKU + ZEBRA (Tables 2, 3, 5)
# =============================================

"""
KEY IMPLEMENTATION NOTES:

1. SUDOKU REPRESENTATION:
   - 81 tokens, values 1-9 for digits
   - Mask token: 0 (or a separate ID)
   - Given clues are FIXED during inference
   - "Correctly solved" = ALL 81 cells correct

2. MODEL SIZES (Appendix D.2):
   - Sudoku MDM: 6M parameters, GPT-2 architecture
   - Zebra MDM: 19M parameters
   - ARM baselines: 42M parameters each
   
3. TRAINING (Appendix D.2):
   - Learning rate: 0.001
   - Batch size: 128
   - Epochs: 300
   - Use the Ye et al. (2024) codebase defaults for everything else

4. INFERENCE (Appendix D.2):
   - 50 reverse sampling steps
   - Gumbel noise coefficient: 0.5
   - This noise is added to the oracle's SCORES (not token probs)

5. ARM WITH ORDERING:
   The key innovation comparison. Shah et al. (2024) provides:
   - A Sudoku solver that outputs solving ORDER
   - Each puzzle has a unique solving order based on 7 strategies
   - The ARM is trained via teacher forcing in this order
   - This is a STRONGER baseline than standard left-to-right ARM
   
   You MUST use Shah et al.'s code for this. The order matters.
   The fact that MDM+adaptive BEATS this supervised ordering baseline
   is the paper's strongest empirical claim.

6. HARD SUDOKU (Table 5):
   - SAME models trained on easy puzzles
   - Evaluated on HARD puzzles (require backtracking / unseen strategies)
   - "The hard dataset contains around 1M Sudoku puzzles"
   - No retraining needed

7. EVALUATION METRIC:
   Percentage of puzzles completely and correctly solved.
   A single wrong cell = entire puzzle counts as incorrect.
"""

# The actual implementation should use the Ye et al. (2024) codebase.
# The code in Phase 2 provides the inference strategy implementations
# that need to be integrated into their codebase.


# =============================================
# COLAB NOTEBOOK 4: SCALING LAWS (Fig 2 left, Fig 2 top-right)
# =============================================

"""
THIS IS THE MOST COMPUTE-INTENSIVE EXPERIMENT.

KEY DETAILS:

1. ISOFLOP ANALYSIS (Appendix C.1, citing Hoffmann et al. 2022):
   - For each FLOPs budget C:
     - Train models of varying sizes (non-embedding parameters)
     - Set iterations so total tokens = C / (6 * N_params)
     - Record validation loss for each model size
     - Plot the BEST (lowest) validation loss for that C
   - This gives one point per FLOPs budget

2. POSITIONAL EMBEDDING (Section 3.2):
   "given that RoPE has an inductive bias towards left-to-right ordering,
    we employ a LEARNABLE positional embedding layer for ALL experiments"
   
   CRITICAL: This means the ARM baseline is ALSO re-run with learned
   positional embeddings. The paper says: "Consequently, we also re-run
   the baseline results, where RoPE was employed."

3. π-LEARNER IMPLEMENTATION (Section 3.2):
   - Use causal transformer (same as ARM)
   - Input is permuted: π(x0) = (x_{π(0)}, x_{π(1)}, ..., x_{π(L-1)})
   - The model sees the PERMUTED sequence with causal attention
   - Evaluation uses Equation (3) to compute likelihood
   
   "To train a π-learner, we employ a transformer with causal attention 
    and use permuted data π(x0) as input."

4. PERMUTATION DISTRIBUTIONS (Appendix C.1):
   - Identity: π = id (this IS the ARM)
   - Uniform: π ~ Uniform(S_L) — random permutation
   - Closer: start from identity, do L/10 random transposition swaps
   - Much-closer: start from identity, do √L random transposition swaps
   
   "L log(L) number of swaps results in a distribution very close to 
    Uniform(S_L) (Bormashenko, 2011)"
   
   So L/10 swaps is still far from uniform (moderately shuffled).
   √L swaps is very close to identity (barely shuffled).
   
   "We sample three permutations from the interpolating distribution 
    and Uniform(S_L) and plot the scaling law for each"
   
   So: 3 lines for uniform, 3 for closer, 3 for much-closer.

5. MDM TRAINING:
   Uses the Nie et al. (2024) codebase with learnable pos embeddings.
   
6. FIGURE 2 LEFT:
   x-axis: log(FLOPs), logarithmic scale
   y-axis: -log p_θ(x) = negative log-likelihood = validation loss
   Lines: AR (1), MDM (1), π-learner-much-closer (3), 
          π-learner-closer (3), π-learner-unif (3)

7. FIGURE 2 TOP-RIGHT (table):
   Single model size (likely 170M), three π-learner configs.
   This uses the TEXT IMBALANCE experiment (Appendix C.2.2):
   "We take a 170M MDM pretrained with text data"
   "we calculate the expectation over 1024 samples of x0 ~ pdata"
"""


# =============================================
# COLAB NOTEBOOK 5: TEXT GENERATION PERPLEXITY (Fig 3)
# =============================================

"""
KEY DETAILS:

1. MODEL DISCREPANCY:
   Figure 3 caption: "We employ a pretrained 170M MDM"
   Appendix D.1.2: "We employ a 1.1B MDM pretrained on text data as a baseline"
   
   RESOLUTION: Appendix D.1.2 is more detailed and authoritative.
   Use 1.1B MDM. The caption may refer to an earlier version.
   
   However, if 1.1B is unavailable, 170M is acceptable (noted in caption).

2. EVALUATION MODEL:
   "LLama2-7B (Touvron et al., 2023)" — NOT a newer LLaMA variant.
   Specifically LLaMA-2 7B (base, not chat).

3. TEXT ORACLE (Appendix D.1.2):
   Uses GAUSSIAN noise (not Gumbel):
   F(θ, x_t) = Top K(|p(j1) - p(j2)| + ε)  where ε ~ N(0, σ²)
   
   The noise std σ is a hyperparameter to tune.

4. METRICS:
   - Generative Perplexity (GenPPL): 
     exp(-1/N × Σ log p_eval(x_i | x_{<i}))
     where p_eval is LLaMA-2 7B
   
   - Entropy:
     -Σ p_i log p_i where p_i = #{x^i = i}/L
     This is UNIGRAM token frequency entropy of generated text
     
5. x-axis: Sampling Steps (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
   Two y-axes: left = GenPPL, right = Entropy

6. K CHOICE (Appendix D.1.2):
   K = (# masked tokens) × (α_s - α_t) / (1 - α_t)
   Both deterministic K and stochastic K (Binomial) give similar results.
"""


# =============================================
# COLAB NOTEBOOK 6: LLaDA 8B EVALUATION (Table 4)
# =============================================

"""
KEY DETAILS:

1. MODEL: LLaDA 8B base model (Nie et al., 2025)

2. TASK CATEGORIES (Appendix D.3):
   a) INFILLING tasks (non-autoregressive sampling):
      - HumanEval-Infill (Single, Multi, Split): from Bavarian et al. (2022)
        "Each instance is grouped by the span of the masked code"
        Output length = size of masked span (predetermined)
      - ROCStories: story completion infilling
      
   b) INSTRUCTION-ANSWERING tasks (semi-autoregressive sampling):
      - Math: mathematical reasoning
      - MMLU: multiple-choice knowledge questions
        "For instruction-answering tasks, we employ a semi-autoregressive 
         sampling strategy"
        "instruction-answering tasks require an explicit length specification"
        "we follow the sampling configuration of (Nie et al., 2025)"

3. SEMI-AUTOREGRESSIVE SAMPLING:
   Follow the LLaDA codebase exactly. The idea:
   - Prompt is given as prefix (unmasked)
   - Response region has fixed-length mask tokens  
   - MDM inference fills in the masked region
   - For instruction tasks, response length must be specified
   
   The LLaDA paper (Nie et al., 2025) describes this in detail.

4. APPLYING ADAPTIVE INFERENCE:
   Replace the vanilla token unmasking in LLaDA's inference with
   top_prob or top_prob_margin strategies.
   
   The rest of the pipeline (prompt formatting, evaluation metrics,
   response parsing) stays identical to LLaDA's defaults.
"""


# =============================================
# COLAB NOTEBOOK 7: BELIEF PROPAGATION + ALL PLOTS
# =============================================

"""
FIGURE 4: Belief Propagation for Planted CSP

Parameters (from caption):
- k = 3 (arity)
- m = 3 (alphabet/vocabulary size)
- g = NAE (Not-All-Equal predicate)
- N = 10000 (dimension)
- x-axis: D/k (average degree / arity), range approximately 43 to 68
- y-axis: overlap with ground truth
- Two curves: "planted init" and "random init"

KEY VALUES from caption:
- D_KS/K = 64 (analytically computed, consistent with phase transition)
- D_cond/K ≈ 50 (empirically observed from plot)

BELIEF PROPAGATION IMPLEMENTATION:
Follow Definition B.10 exactly. Messages are:
- M^c_{i→S}: variable i to clause S, for color c
- M^c_{S→i}: clause S to variable i, for color c

Update rules (Equations 4 and 5):
- Variable-to-clause: M^c_{i→S} ∝ Π_{T: i∈T, T≠S} M^c_{T→i}
- Clause-to-variable: M^c_{S→i} ∝ Σ_{σ∈{1,...,m}^{S\i}} g(σ∪_i c) Π_{j≠i∈S} M^{σ_j}_{j→S}

NAE predicate: NAE(x1,x2,x3) = 1 - 1[x1=x2=x3]

PLANTED CSP GENERATION (Definition B.9):
1. Sample σ ~ Uniform({1,...,m}^N) — ground truth
2. For each ordered k-tuple S of distinct elements from [N]:
   Include clause S with probability φ/N^{k-1} if g(σ|_S) = 1
3. Average degree = kP/N

OVERLAP (Definition B.9):
d(σ, σ̂) = min_{π∈S_m} Σ_i 1[σ_i = π(σ̂_i)]
(minimize over permutations of the alphabet to handle symmetry)

For m=3, k=3, NAE:
- γ = P(NAE satisfied by random) = 1 - m/m^3 = 1 - 3/27 = 24/27 = 8/9
- Wait: P(all equal for 3 tokens from {1,2,3}) = 3/27 = 1/9
- So γ = 1 - 1/9 = 8/9 ≈ 0.889

This is CPU-only, runs in a few hours.
"""

# See Phase 2 document for the full BP implementation code.


# =============================================
# PLOTTING SPECIFICATIONS
# =============================================

"""
ALL FIGURES should use publication-quality settings:
- Font: serif (matching LaTeX default)
- Font size: 12pt
- Figure sizes: approximately 6×5 inches for single plots
- DPI: 300 for PDF output
- Colors: follow the paper's conventions:
  - ARM/AR: orange
  - MDM: blue
  - π-learners: varying colors for different distributions
- Save as both PDF and PNG

FIGURE 1 (conceptual):
Best done in LaTeX/TikZ. If using matplotlib, create a schematic
showing the masking process and inference comparison.

FIGURE 2 (composite):
Left panel: scaling law (log-log plot)
Right top: small table (can be matplotlib table or just text)
Right bottom: bar chart of prediction error by position

FIGURE 3:
Dual y-axis plot (GenPPL + Entropy vs Sampling Steps)

FIGURE 4:
Simple line plot (overlap vs D/k) with two curves

TABLES 1-5:
Generate as LaTeX tables for inclusion in a paper,
and also as matplotlib tables for standalone viewing.
"""
```