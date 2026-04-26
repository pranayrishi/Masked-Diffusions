# Reproduction Guide: "Train for the Worst, Plan for the Best"
## Phase 1: Infrastructure, Data, and MDM/ARM Training

**Paper**: Kim, Shah, Kontonis, Kakade, Chen. "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions." ICML 2025. arXiv:2502.06768v3.

**Target**: Full reproduction of all experiments, figures, and tables.

---

## 1. HIGH-LEVEL PAPER SUMMARY (For the AI Agent's Context)

This paper studies **Masked Diffusion Models (MDMs)** for discrete (token-based) data and compares them to **Autoregressive Models (ARMs)**. The paper has two core theses:

1. **Training Complexity (Sections 3.1–3.3)**: MDMs are forced to learn exponentially many infilling subproblems (one for every possible mask pattern M ⊆ [L]), many of which are computationally intractable. ARMs only need to solve L subproblems (left-to-right). This is demonstrated both theoretically (via connections to planted CSPs) and empirically (via scaling law experiments on text data and synthetic L&O distributions).

2. **Inference Flexibility (Section 4)**: Despite training on hard problems, MDMs contain enough information in their logits to *sidestep* hard subproblems at inference time. By adaptively choosing which tokens to unmask (instead of random order), MDMs can dramatically improve. On Sudoku, accuracy goes from <7% (vanilla) to ~90% (adaptive). MDMs with adaptive inference can even outperform ARMs that were trained with knowledge of the correct generation order.

### Key Experiments to Reproduce:
- **Figure 2 (Left)**: Scaling law comparison — ARM vs MDM vs π-learners on SlimPajama text data
- **Figure 2 (Right, Top)**: Validation loss table for π = id vs π ~ Closer(S_L) vs π ~ Unif(S_L)
- **Figure 2 (Right, Bottom)**: Prediction error imbalance across positions for L&O-NAE-SAT
- **Figure 3**: Generative perplexity comparison (vanilla vs adaptive MDM inference) on text
- **Table 1**: L&O-NAE-SAT accuracy (vanilla vs adaptive) for various (N, P)
- **Table 2**: Sudoku puzzle accuracy (ARM with/without ordering, MDM vanilla/top-prob/top-prob-margin)
- **Table 3**: Zebra puzzle accuracy (same comparison)
- **Table 4**: LLaDA 8B results on coding/math/infill tasks
- **Table 5**: Hard Sudoku generalization results
- **Figure 4** (Appendix): Belief propagation overlap vs degree for planted CSP

---

## 2. ENVIRONMENT SETUP

### 2.1 Hardware Requirements
- **GPU**: Minimum 1× NVIDIA A100 80GB for training small models (6M, 19M, 42M, 170M parameters). For LLaDA 8B inference experiments (Table 4), you need at least 1× A100 80GB (or 2× A6000 48GB with model parallelism).
- **Storage**: ~500GB for SlimPajama dataset, ~10GB for puzzle datasets, ~16GB for LLaDA 8B weights.
- **RAM**: 64GB+ recommended.

### 2.2 Software Stack
```bash
# Create conda environment
conda create -n mdm_reproduce python=3.10 -y
conda activate mdm_reproduce

# PyTorch (use CUDA 11.8 or 12.1 depending on your system)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers==4.36.2
pip install datasets==2.16.1
pip install accelerate==0.25.0
pip install wandb
pip install numpy==1.26.3
pip install scipy==1.12.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
pip install tqdm
pip install einops
pip install flash-attn --no-build-isolation  # Optional but recommended for speed

# For Sudoku/Zebra experiments — clone the codebase from Ye et al. (2024)
# Paper states: "For the training and inference, we use the codebase of (Ye et al., 2024)"
# This is: https://github.com/HKUNLP/diffusion-forcing (or the MDM codebase referenced)
# The exact repo may be: https://github.com/HKUNLP/DiffuSeq or similar discrete diffusion repos

# For scaling law experiments — paper states: "We leverage the codebase from (Nie et al., 2024)"
# This is the MDLM/scaling codebase: https://github.com/ML-GSAI/SMDM or the scaling laws repo
# Specifically referenced paper: "Scaling up masked diffusion models on text" (Nie et al., 2024)
```

### 2.3 Critical Note on Codebases
The paper explicitly references two codebases:
1. **Nie et al. (2024)** codebase for scaling law experiments (Section 3.2, Figure 2 Left). This is from "Scaling up masked diffusion models on text" — likely https://github.com/ML-GSAI/SMDM
2. **Ye et al. (2024)** codebase for Sudoku/Zebra experiments (Section 4.2, Tables 2-3). This is from "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning" — likely https://github.com/HKUNLP/discrete-diffusion or a related repo.
3. **LLaDA** (Nie et al., 2025) for the 8B model experiments (Table 4). This is from "Large Language Diffusion Models" — https://github.com/ML-GSAI/LLaDA

**Action for the AI agent**: Clone all three repositories. Inspect their READMEs and configs to understand the training pipeline before modifying anything.

---

## 3. DATA PREPARATION

### 3.1 SlimPajama Dataset (for Sections 3.2, 3.3, and Figure 2/3)

**Source**: Soboleva et al. (2023). "SlimPajama: A 627B token cleaned and deduplicated version of RedPajama."

```python
from datasets import load_dataset

# Load SlimPajama — this is a large dataset (~627B tokens)
# The paper uses a subset for training. Based on scaling law experiments,
# they train models at various compute budgets.
dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
```

**Tokenizer**: The paper uses the codebase from Nie et al. (2024), which uses the GPT-2 tokenizer (vocab size 50257) as that paper's codebase is based on TinyLlama configurations.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Sequence length L = 2048 (stated in Section 3.2: "the sequence length L is 2048")
```

**Preprocessing**:
- Tokenize all text with GPT-2 tokenizer
- Pack into sequences of length L = 2048 (standard packing, concatenate documents with EOS separator and chunk into fixed length)
- Create train/validation splits

### 3.2 L&O-NAE-SAT Distribution (for Sections 3.3 and 4.2, Figure 2 Right Bottom, Table 1)

This is a **synthetic** distribution you must generate. Definition from Section 3.3:

**L&O-NAE-SAT Definition**:
- Permutation π = identity (stated: "π given by the identity permutation")
- N latent tokens, P observation tokens, N + P = L
- Latent tokens: sampled independently from a prior (uniform over alphabet {1, ..., m})
- Each observation O_j is deterministically given by:
  `NAE(x_{i1}, x_{i2}, x_{i3}) = 1 - 1[x_{i1} = x_{i2} = x_{i3}]`
  for some randomly chosen (pre-fixed) triples (i1, i2, i3) ∈ [N]
- The triples are pre-fixed (sampled once and held fixed for the distribution)

**Concrete parameters from Appendix C.2.1**:
- For the error imbalance experiment (Figure 2 bottom right): **(N, P) = (20, 280)**, total sequence length L = 300
- Pad last 212 tokens with additional token value 2 (to reach sequence length 512 for the model)
- Model: 19M MDM with RoPE, max sequence length 512
- Training: 2×10³ iterations for the model, then 5×10⁴ iterations for the Bayes-optimal proxy

**For Table 1 experiments (Appendix D.1.1)**:
- Five configurations: (N,P) = (25,275), (30,270), (40,260), (50,250), (100,200)
- Each has total L = 300
- Train a 19M MDM for each configuration

**Data generation code**:
```python
import numpy as np
import torch

def generate_lo_nae_sat_dataset(N, P, m, num_samples, seed=42):
    """
    Generate L&O-NAE-SAT distribution samples.
    
    Args:
        N: number of latent tokens
        P: number of observation tokens  
        m: alphabet size (vocabulary size for latent tokens)
        num_samples: number of samples to generate
        seed: random seed
    
    Returns:
        data: np.array of shape (num_samples, N+P), values in {1,...,m} for latents, {0,1} for observations
        triples: the fixed triples used for observations
    """
    rng = np.random.RandomState(seed)
    
    # Step 1: Fix the random triples for observations
    # Each observation corresponds to a randomly chosen triple of latent indices
    # P triples, each is 3 indices from {0, 1, ..., N-1}
    triples = []
    for _ in range(P):
        triple = rng.choice(N, size=3, replace=True)  # with replacement as per definition
        triples.append(triple)
    triples = np.array(triples)  # shape (P, 3)
    
    # Step 2: Generate samples
    data = np.zeros((num_samples, N + P), dtype=np.int64)
    
    for s in range(num_samples):
        # Sample latent tokens uniformly from {1, ..., m}
        latents = rng.randint(1, m + 1, size=N)  # values in {1, ..., m}
        data[s, :N] = latents
        
        # Compute observation tokens
        for j in range(P):
            i1, i2, i3 = triples[j]
            # NAE(x_i1, x_i2, x_i3) = 1 - 1[x_i1 = x_i2 = x_i3]
            nae_value = 1 - int(latents[i1] == latents[i2] == latents[i3])
            data[s, N + j] = nae_value  # observation is 0 or 1
        
    return data, triples

# For Figure 2 (bottom right) experiment:
# m = 3 (alphabet size, as suggested by Figure 4 caption: k=3, m=3 for NAE)
# Actually, the paper does not explicitly state m for Section 3.3 L&O-NAE-SAT.
# From the context of Proposition 3.3 and Figure 4 (m=3, k=3, g=NAE), use m=3.
# Also, naive guessing accuracy of 75% in Table 1 is consistent with NAE on m=3:
# P(NAE=1 | random) = 1 - (1/m)^2 = 1 - 1/9 = 8/9 ≈ 0.889 for m=3 ... 
# Wait: P(all equal) for 3 tokens from {1,...,m} = m * (1/m)^3 = 1/m^2 = 1/9
# So P(NAE=1) = 8/9 ≈ 0.889, but Table 1 says naive guessing = 75%.
# For m=2: P(all equal) = 2*(1/2)^3 = 1/4, so P(NAE=1) = 3/4 = 75%. This matches!
# Therefore m = 2 for the L&O-NAE-SAT experiments.

data, triples = generate_lo_nae_sat_dataset(N=20, P=280, m=2, num_samples=100000)
```

**CRITICAL INSIGHT**: The naive guessing accuracy of 75% mentioned in Table 1 implies m=2 (binary alphabet for latents). This is because with m=2, the probability that three uniformly random binary tokens are all equal is (1/2)^2 = 1/4, so P(NAE=1) = 3/4 = 75%.

### 3.3 Sudoku Dataset (for Tables 2 and 5)

**Source**: Shah et al. (2024), who created it from Radcliffe (2020) — "3 million Sudoku puzzles with ratings" from Kaggle: https://www.kaggle.com/dsv/1495975

**Train/Test Split** (from Appendix D.2):
- **Training set**: Puzzles from Radcliffe (2020) that can be solved using 7 fixed strategies and do NOT require backtracking-based search. These are the "easy" puzzles.
- **Test set (easy)**: Held-out portion from the same filtered set.
- **Test set (hard, Table 5)**: The REMAINING puzzles from Radcliffe (2020) — those that require strategies NOT in the 7 fixed strategies and/or require backtracking. Contains ~1M puzzles.

**Representation**: A 9×9 Sudoku grid is represented as a sequence of 81 tokens. Each token is a digit 1-9. Given clues are unmasked; empty cells are masked (token = 0 = mask token).

```python
# Download from Kaggle
# File: sudoku-3m.csv with columns: puzzle, solution, clues, difficulty
# puzzle: 81-char string with '.' for empty cells
# solution: 81-char string with all digits filled

import pandas as pd

df = pd.read_csv("sudoku-3m.csv")

def puzzle_to_tokens(puzzle_str, solution_str):
    """Convert puzzle string to token sequence."""
    puzzle_tokens = []
    solution_tokens = []
    for p, s in zip(puzzle_str, solution_str):
        if p == '.':
            puzzle_tokens.append(0)  # mask token
        else:
            puzzle_tokens.append(int(p))
        solution_tokens.append(int(s))
    return puzzle_tokens, solution_tokens
```

**The Shah et al. (2024) filtering**: You need to implement or obtain the 7 Sudoku strategies filter. The strategies typically include: naked singles, hidden singles, naked pairs, hidden pairs, pointing pairs, box/line reduction, and naked triples. These are standard constraint-propagation strategies. Shah et al.'s code/dataset should be available at their paper's repo.

### 3.4 Zebra (Einstein) Puzzle Dataset (for Table 3)

**Source**: Also from Shah et al. (2024). The Zebra puzzle is a classic logic puzzle where you must determine assignments of attributes to houses given a set of clues.

**Representation**: The puzzle state is a sequence of tokens representing attribute-house assignments. The exact tokenization follows Shah et al. (2024)'s format.

### 3.5 LLaDA 8B Weights (for Table 4)

**Source**: Nie et al. (2025). "Large Language Diffusion Models."
- Weights available at: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct (or base model)
- This is a pretrained 8B parameter MDM.

```bash
# Download LLaDA 8B
# The paper uses the base model (not instruct) for the experiments
pip install huggingface_hub
huggingface-cli download GSAI-ML/LLaDA-8B --local-dir ./llada-8b/
```

---

## 4. MODEL ARCHITECTURES

### 4.1 Transformer Architecture Details

The paper uses the **TinyLlama/GPT-style transformer** architecture, following Nie et al. (2024) configurations.

**Key architectural detail (Section 3.2)**: "given that RoPE has an inductive bias towards left-to-right ordering, we employ a **learnable positional embedding layer** for all experiments to correct this."

This is critical: **Do NOT use RoPE for the scaling law experiments**. Use learned absolute positional embeddings instead. The paper re-runs baselines with this change.

**Exception**: For the L&O-NAE-SAT error imbalance experiment (Appendix C.2.1), they DO use "a 19M MDM with RoPE and a maximum sequence length of 512."

**Model sizes used in the paper**:

| Experiment | Model Size | Architecture Notes |
|-----------|-----------|-------------------|
| Scaling laws (Fig 2 Left) | Various (IsoFLOP analysis) | Transformer with causal attention (for π-learners), learnable positional embeddings |
| L&O-NAE-SAT error (Fig 2 Right) | 19M | MDM with RoPE, seq len 512 |
| Text gen perplexity (Fig 3) | 170M (inference) / 1.1B (Appendix D.1.2) | MDM pretrained on text |
| Sudoku (Table 2) | 6M (MDM), 42M (ARM) | GPT-2 architecture |
| Zebra (Table 3) | 19M (MDM), 42M (ARM) | GPT-2 architecture |
| LLaDA (Table 4) | 8B | LLaDA architecture |
| Hard Sudoku (Table 5) | 6M (MDM), 42M (ARM) | Same as Table 2 |

### 4.2 MDM Training Implementation

The MDM training objective (from Proposition 2.1 and the loss in Section 2):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDMTrainer:
    """
    Masked Diffusion Model training.
    
    The loss is:
    L_θ = ∫₀¹ (α'_t / (1 - α_t)) * E_{x0~pdata, xt~q_{t|0}(·|x0)} 
          [Σ_{i: x^i_t = 0} -log p_θ(x^i_0 | x_t)] dt
    
    In practice (following Zheng et al. 2024, Proposition E.1), this simplifies to:
    For each training step:
    1. Sample x0 from data
    2. Sample number of tokens to mask n ~ some distribution over {1,...,L}
    3. Randomly mask n positions to get x_masked
    4. Predict original tokens at masked positions
    5. Loss = cross-entropy at masked positions, weighted by 1/n
    """
    
    def __init__(self, model, vocab_size, seq_len, mask_token_id=0):
        self.model = model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
    
    def compute_loss(self, x0):
        """
        x0: (batch_size, seq_len) — clean token sequences, values in {1, ..., vocab_size}
        
        Following the paper's framework:
        - Sample noise level t uniformly from [0, 1]
        - Each token is masked independently with probability (1 - α_t)
        - The network predicts the original token at each masked position
        - Loss is cross-entropy weighted appropriately
        """
        batch_size, L = x0.shape
        device = x0.device
        
        # Sample noise level t ~ Uniform(0, 1)
        # In practice, discretize: sample number of masks n ~ Uniform(1, L)
        # This is equivalent per Proposition E.1 (Zheng et al. 2024)
        n = torch.randint(1, L + 1, (batch_size,), device=device)  # (batch_size,)
        
        # Create mask: for each sample, randomly select n positions to mask
        # Efficient implementation: for each sample, generate random permutation 
        # and take first n indices
        mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        for b in range(batch_size):
            indices = torch.randperm(L, device=device)[:n[b]]
            mask[b, indices] = True
        
        # Create masked input
        x_masked = x0.clone()
        x_masked[mask] = self.mask_token_id
        
        # Forward pass — model predicts distribution over vocab at each position
        # The model is time-embedding-free (Section 2): p_θ(·|x_t) = p_θ(·|x_t)
        logits = self.model(x_masked)  # (batch_size, seq_len, vocab_size)
        
        # Compute cross-entropy loss only at masked positions
        # Weight by 1/n per the loss formulation
        loss = F.cross_entropy(
            logits[mask],      # (total_masked_tokens, vocab_size)
            x0[mask],          # (total_masked_tokens,)
            reduction='none'
        )
        
        # Group by sample and weight by 1/n
        # In practice, simple mean over masked tokens works (equivalent in expectation)
        loss = loss.mean()
        
        return loss
```

**IMPORTANT**: The paper notes (Section 2): "In practice, a time-embedding-free architecture for the denoising network, i.e., p_θ(·|x_t, t) = p_θ(·|x_t) is generally used as x_t implicitly contains information about t via the number of masked tokens." So **do NOT** pass the noise level t as input to the network.

### 4.3 ARM Training Implementation

For autoregressive models, use standard causal language modeling:

```python
class ARMTrainer:
    """
    Standard left-to-right autoregressive training.
    
    Loss = Σ_{i=0}^{L-1} -log p_θ(x^i | x^0, ..., x^{i-1})
    """
    
    def __init__(self, model, vocab_size, seq_len):
        self.model = model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def compute_loss(self, x0):
        """Standard causal LM loss with teacher forcing."""
        # Shift: input is x0[:-1], target is x0[1:]
        logits = self.model(x0[:, :-1])  # (batch, L-1, vocab)
        targets = x0[:, 1:]              # (batch, L-1)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        return loss
```

### 4.4 π-Learner Training (for Section 3.2, Figure 2 Left)

This is the KEY experiment for understanding training complexity. A π-learner trains an autoregressive model on permuted data.

```python
class PiLearnerTrainer:
    """
    π-learner: Train causal model on permuted sequences.
    
    Given permutation π, input is π(x0) = (x_{π(0)}, x_{π(1)}, ..., x_{π(L-1)})
    The model is a standard causal transformer on the permuted sequence.
    
    Likelihood is computed via:
    log p_θ(x0) = Σ_{i=0}^{L-1} log p_θ(x^{π(i)} | x0[π{i,...,L-1}])
    
    which equals the standard causal loss on the permuted input.
    """
    
    def __init__(self, model, vocab_size, seq_len, permutation):
        """
        permutation: np.array of shape (seq_len,) — the fixed permutation π
        """
        self.model = model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.perm = permutation  # π
    
    def permute_batch(self, x0):
        """Apply permutation π to the sequence."""
        # x0: (batch, L)
        return x0[:, self.perm]  # Reorder columns by permutation
    
    def compute_loss(self, x0):
        """Causal LM loss on permuted input."""
        x_perm = self.permute_batch(x0)
        logits = self.model(x_perm[:, :-1])
        targets = x_perm[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        return loss
```

**Permutation distributions** (from Appendix C.1):

```python
import numpy as np

def sample_permutation(L, distribution="uniform"):
    """
    Sample a permutation from one of the distributions described in the paper.
    
    Distributions (Appendix C.1):
    - "identity": π = identity (= ARM)
    - "uniform": π ~ Uniform(S_L) (≈ MDM training)  
    - "closer": Start from identity, apply L/10 random swaps
    - "much_closer": Start from identity, apply sqrt(L) random swaps
    
    Background: L*log(L) swaps gives distribution close to Uniform(S_L)
    (Bormashenko, 2011: "A coupling argument for the random transposition walk").
    So L/10 swaps is moderately close to identity, sqrt(L) is very close to identity.
    """
    perm = np.arange(L)
    
    if distribution == "identity":
        return perm
    
    if distribution == "uniform":
        np.random.shuffle(perm)
        return perm
    
    if distribution == "closer":
        num_swaps = L // 10  # L/10 swaps
    elif distribution == "much_closer":
        num_swaps = int(np.sqrt(L))  # sqrt(L) swaps
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    for _ in range(num_swaps):
        i, j = np.random.choice(L, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm
```

**Experimental protocol for Figure 2 Left** (from Appendix C.1):

1. Sample 3 permutations from each of: Uniform(S_L), Closer, Much-Closer distributions
2. For each permutation, train a π-learner at multiple compute budgets (IsoFLOP analysis)
3. Also train standard ARM (π = identity) and MDM
4. Plot validation loss vs log(FLOPs)

**IsoFLOP analysis** (from Appendix C.1, citing Hoffmann et al. 2022):
- For a given FLOPs budget C, vary model size (non-embedding parameters)
- Set number of training iterations so total tokens = C / (6 × N_params)
- For each compute budget, pick the model size that achieves lowest validation loss
- This gives one data point per compute budget

---

## 5. TRAINING CONFIGURATIONS

### 5.1 Optimizer and Hyperparameters (from Appendix C.1)

All models use the same training configuration:

```python
# AdamW optimizer (Loshchilov & Hutter, 2017)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-4,          # max learning rate
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Cosine learning rate schedule
# Max LR: 4e-4
# Min LR: 4e-5
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=total_steps,
    eta_min=4e-5
)

# Sequence length
L = 2048

# Positional embedding: LEARNABLE (not RoPE) for scaling law experiments
# Exception: L&O-NAE-SAT uses RoPE with max_seq_len=512
```

### 5.2 Sudoku/Zebra Training (from Appendix D.2)

```python
# Model sizes:
# Sudoku MDM: 6M parameters, GPT-2 architecture
# Zebra MDM: 19M parameters, GPT-2 architecture
# ARM baselines: 42M parameters

# Training config:
learning_rate = 0.001
batch_size = 128
num_epochs = 300

# Inference config:
num_reverse_steps = 50  # "50 reverse sampling steps"
gumbel_noise_coefficient = 0.5  # "add Gumbel noise with a coefficient of 0.5 to MDM inference oracle F"
```

### 5.3 ARM with Ordering Information (for Tables 2, 3, 5)

The "ARM (with ordering)" baseline is trained using **supervised teacher forcing** where the training data includes the correct generation order for each puzzle. From Shah et al. (2024):

- For each puzzle, compute a valid solving order (using constraint propagation strategies)
- Train the ARM to generate tokens in this order (i.e., permute the sequence so the generation order becomes left-to-right)
- This is a much stronger baseline than standard left-to-right ARM

The "ARM (without ordering)" baseline uses standard left-to-right generation.

---

## 6. SPECIFIC EXPERIMENT INSTRUCTIONS

### 6.1 Experiment: Figure 2 Left — Scaling Laws

**What to plot**: x-axis = log(FLOPs), y-axis = -log p_θ(x) (negative log-likelihood / validation loss)

**Lines to plot**:
1. **AR** (orange): Standard ARM with identity permutation
2. **MDM** (blue): Full MDM training
3. **π-learner-much-closer**: 3 permutations from Much-Closer distribution
4. **π-learner-closer**: 3 permutations from Closer distribution  
5. **π-learner-unit** (should be "π-learner-unif"): 3 permutations from Uniform(S_L)

**Steps**:
1. Define compute budgets: Use ~5-7 FLOPs values spanning roughly 10^18 to 10^21 (adjust based on your available compute). The paper shows roughly 3 orders of magnitude on x-axis.
2. For each FLOPs budget, train models of varying sizes and pick best validation loss.
3. For ARM: standard causal transformer with learnable pos embeddings on SlimPajama.
4. For MDM: Use the Nie et al. (2024) codebase with learnable pos embeddings (replacing RoPE).
5. For π-learners: Train causal transformers on permuted data.

**Evaluation**: Compute validation loss on a held-out portion of SlimPajama.

### 6.2 Experiment: Figure 2 Right Top — Validation Loss Table

This is the table in the top-right of Figure 2:

| Task | π = id | π ~ Closer(S_L) | π ~ Unif(S_L) |
|------|--------|-----------------|---------------|
| Val. Loss | 3.171 | 3.212 | 3.245 |

**What this shows**: Validation loss for a single model size (likely the 170M model) trained as π-learners with different permutations. The identity permutation (ARM) achieves the best loss on text data.

### 6.3 Experiment: Figure 2 Right Bottom — L&O-NAE-SAT Error Imbalance

**What to plot**: x-axis = Position (0 to ~280), y-axis = Prediction Error

**Setup** (from Appendix C.2.1):
- Distribution: L&O-NAE-SAT with (N=20, P=280), m=2
- Model: 19M MDM with RoPE, max sequence length 512
- Pad sequences to 512 with token value 2
- Train MDM for 2×10³ iterations
- Train proxy Bayes-optimal MDM for 5×10⁴ iterations

**Error measurement**:
```python
def measure_error_imbalance(model, proxy_model, data, N, P, ell=11):
    """
    For each ℓ ∈ [1, N-1], randomly mask ℓ latent positions 
    and ℓ*(P/N) observation positions.
    Measure error at each position.
    Repeat 1000 times for certainty.
    
    Error at position i:
    E_{x0}[|log p_θ(x0|x0[M]) - log p_data(x0|x0[M])|²]
    where p_data is approximated by the proxy model.
    """
    errors = np.zeros(N + P)
    num_trials = 1000
    
    for trial in range(num_trials):
        # Sample x0 from data
        x0 = data[np.random.randint(len(data))]
        
        # Randomly mask ell latent positions
        latent_mask_indices = np.random.choice(N, size=ell, replace=False)
        # Randomly mask ell * (P/N) observation positions
        obs_mask_count = int(ell * P / N)
        obs_mask_indices = np.random.choice(range(N, N+P), size=obs_mask_count, replace=False)
        
        mask_indices = np.concatenate([latent_mask_indices, obs_mask_indices])
        
        # Create masked sequence
        x_masked = x0.copy()
        x_masked[mask_indices] = 0  # mask token
        
        # Get predictions from both models
        with torch.no_grad():
            logits_model = model(torch.tensor(x_masked).unsqueeze(0))
            logits_proxy = proxy_model(torch.tensor(x_masked).unsqueeze(0))
        
        # Compute error at each masked position
        for idx in mask_indices:
            log_p_model = F.log_softmax(logits_model[0, idx], dim=-1)
            log_p_proxy = F.log_softmax(logits_proxy[0, idx], dim=-1)
            # Use the true token's probability
            err = (log_p_model[x0[idx]] - log_p_proxy[x0[idx]]).item() ** 2
            errors[idx] += err
    
    errors /= num_trials
    return errors
```

**Expected result**: Latent positions (first N=20 positions) show HIGHER error (darker in the figure, shown as light region in text description). Observation positions (positions 20-299) show LOWER error.

---

## 7. CHECKLIST FOR PHASE 1

Before moving to Phase 2, verify:

- [ ] All three codebases cloned and functional (Nie et al. 2024, Ye et al. 2024, LLaDA)
- [ ] SlimPajama dataset downloaded and tokenized
- [ ] L&O-NAE-SAT data generator implemented and verified (check that naive guessing = 75%)
- [ ] Sudoku dataset downloaded and split (easy train, easy test, hard test)
- [ ] Zebra dataset obtained from Shah et al. (2024)
- [ ] LLaDA 8B weights downloaded
- [ ] Transformer model with learnable positional embeddings implemented
- [ ] MDM training loop implemented (mask-predict objective, no time embedding)
- [ ] ARM training loop implemented (standard causal LM)
- [ ] π-learner training loop implemented (causal LM on permuted data)
- [ ] Permutation sampling functions implemented (identity, uniform, closer, much-closer)
- [ ] Optimizer, scheduler, and hyperparameters configured per Appendix C.1
- [ ] Scaling law IsoFLOP analysis pipeline ready

---

*Continue to Phase 2 for inference strategies and main results reproduction.*