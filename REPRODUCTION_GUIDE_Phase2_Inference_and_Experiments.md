# Reproduction Guide: "Train for the Worst, Plan for the Best"
## Phase 2: Adaptive Inference, Main Experiments, and Figure/Table Reproduction

---

## 8. ADAPTIVE INFERENCE STRATEGIES (Section 4)

This is the core algorithmic contribution of the paper. The key idea: instead of randomly choosing which masked tokens to unmask (vanilla MDM inference), use the model's own logits to decide which tokens the model is most "certain" about, and unmask those first.

### 8.1 Vanilla MDM Inference (Algorithm 1 / Baseline)

```python
import torch
import torch.nn.functional as F
import numpy as np

def vanilla_mdm_inference(model, seq_len, vocab_size, num_steps=50, 
                          mask_token_id=0, device='cuda'):
    """
    Vanilla MDM inference: unmask tokens in random order.
    
    Algorithm (from Section 2.1.2):
    Start from fully masked sequence x_1 = (0, 0, ..., 0).
    For each step from t to s (t > s):
        (a) Sample a set S of masked tokens to unmask.
            Each masked position i is included in S with probability (α_s - α_t)/(1 - α_t).
        (b) For each i in S, sample x^i_s ~ p_θ(x^i | x_t).
    
    The noise schedule α_t goes from α_0 ≈ 1 (no masking) to α_1 ≈ 0 (full masking).
    We reverse from t=1 to t=0.
    
    Args:
        model: the trained MDM (takes masked sequence, outputs logits)
        seq_len: L
        vocab_size: number of token values (excluding mask)
        num_steps: number of reverse steps
        mask_token_id: the mask token ID (0 in paper)
        device: torch device
    
    Returns:
        generated_sequence: (seq_len,) tensor of token IDs
    """
    # Initialize fully masked sequence
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # Define noise schedule: linear schedule α_t = 1 - t
    # So α_0 = 1 (fully clean), α_1 = 0 (fully masked)
    # We go from t=1 to t=0 in num_steps steps
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        
        alpha_t = 1.0 - t  # α_t = 1 - t for linear schedule
        alpha_s = 1.0 - s
        
        # Find currently masked positions
        masked_positions = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(masked_positions) == 0:
            break
        
        # (a) Sample which positions to unmask
        # Probability of unmasking each masked position: (α_s - α_t) / (1 - α_t)
        unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
        unmask_prob = min(unmask_prob, 1.0)
        
        # Sample S: each masked position independently with probability unmask_prob
        unmask_mask = torch.bernoulli(
            torch.full((len(masked_positions),), unmask_prob, device=device)
        ).bool()
        positions_to_unmask = masked_positions[unmask_mask]
        
        if len(positions_to_unmask) == 0:
            continue
        
        # (b) For each position in S, sample token from model's prediction
        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab_size)
        
        for pos in positions_to_unmask:
            # Sample from p_θ(x^i | x_t)
            probs = F.softmax(logits[0, pos, 1:], dim=-1)  # Exclude mask token (index 0)
            token = torch.multinomial(probs, 1).item() + 1  # +1 because we excluded mask
            x[0, pos] = token
    
    # Fill any remaining masked positions (shouldn't happen with enough steps)
    remaining_mask = (x[0] == mask_token_id)
    if remaining_mask.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining_mask.nonzero(as_tuple=True)[0]:
            probs = F.softmax(logits[0, pos, 1:], dim=-1)
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    return x[0]
```

### 8.2 Adaptive MDM Inference — Top Probability

```python
def adaptive_mdm_inference_top_prob(model, seq_len, vocab_size, num_steps=50,
                                     mask_token_id=0, device='cuda',
                                     gumbel_noise_coeff=0.0,
                                     temperature=1.0):
    """
    Adaptive MDM inference with Top Probability oracle.
    
    From Section 4.1:
    "The certainty at position i is max_{j} p_θ(x^i = j | x_t)
     and F(θ, x_t) = Top K(max p_θ(x^i | x_t))"
    
    Instead of randomly selecting which positions to unmask, we select the K positions
    where the model is most confident (highest max probability).
    
    K is chosen to match the expected number of unmaskings in vanilla inference
    (Appendix D.1.2): K = (# masked tokens) × (α_s - α_t) / (1 - α_t)
    
    Args:
        gumbel_noise_coeff: Coefficient for Gumbel noise added to the oracle
            (Appendix D.2: "add Gumbel noise with a coefficient of 0.5")
            This prevents greedy/deterministic selection.
    """
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        
        alpha_t = 1.0 - t
        alpha_s = 1.0 - s
        
        masked_positions = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            break
        
        # Compute K: number of positions to unmask at this step
        # From Appendix D.1.2: K = (# mask tokens) × (α_s - α_t) / (1 - α_t)
        num_masked = len(masked_positions)
        unmask_frac = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
        K = max(1, int(round(num_masked * unmask_frac)))
        K = min(K, num_masked)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab_size)
        
        # Compute certainty scores at masked positions
        # Certainty = max_j p_θ(x^i = j | x_t) for each masked position i
        probs_at_masked = F.softmax(logits[0, masked_positions, 1:] / temperature, dim=-1)
        max_probs = probs_at_masked.max(dim=-1).values  # (num_masked,)
        
        # Add Gumbel noise for stochasticity (Appendix D.2)
        if gumbel_noise_coeff > 0:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(max_probs) + 1e-8) + 1e-8)
            max_probs = max_probs + gumbel_noise_coeff * gumbel_noise
        
        # Select top K positions by certainty
        _, top_k_indices = torch.topk(max_probs, K)
        positions_to_unmask = masked_positions[top_k_indices]
        
        # Sample tokens at selected positions
        for idx, pos in zip(top_k_indices, positions_to_unmask):
            probs = probs_at_masked[idx]
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    # Handle remaining masked positions
    remaining_mask = (x[0] == mask_token_id)
    if remaining_mask.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining_mask.nonzero(as_tuple=True)[0]:
            probs = F.softmax(logits[0, pos, 1:], dim=-1)
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    return x[0]
```

### 8.3 Adaptive MDM Inference — Top Probability Margin (KEY STRATEGY)

This is the paper's main proposed strategy and consistently outperforms Top Probability.

```python
def adaptive_mdm_inference_top_prob_margin(model, seq_len, vocab_size, num_steps=50,
                                            mask_token_id=0, device='cuda',
                                            gumbel_noise_coeff=0.0,
                                            temperature=1.0):
    """
    Adaptive MDM inference with Top Probability Margin oracle.
    
    From Section 4.1:
    "The uncertainty of a position is estimated using the absolute difference 
    between the two most probable values at position i."
    
    Certainty at position i = |p_θ(x^i = j1 | x_t) - p_θ(x^i = j2 | x_t)|
    where j1 and j2 are the two most probable values.
    
    F(θ, x_t) = Top K(|p_θ(x^i = j1 | x_t) - p_θ(x^i = j2 | x_t)|)
    
    Key insight (Section 4.1): "When multiple values have similar probabilities 
    at a position, top probability margin strategy will provide a better estimate 
    of the uncertainty of a position."
    
    Example: If position has probs [0.45, 0.44, 0.11], top-prob gives 0.45 (seems certain)
    but margin gives |0.45 - 0.44| = 0.01 (correctly identifies uncertainty).
    """
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        
        alpha_t = 1.0 - t
        alpha_s = 1.0 - s
        
        masked_positions = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            break
        
        num_masked = len(masked_positions)
        unmask_frac = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
        K = max(1, int(round(num_masked * unmask_frac)))
        K = min(K, num_masked)
        
        with torch.no_grad():
            logits = model(x)
        
        # Compute probability margins at masked positions
        probs_at_masked = F.softmax(logits[0, masked_positions, 1:] / temperature, dim=-1)
        
        # Get top-2 probabilities at each position
        top2_probs, _ = torch.topk(probs_at_masked, k=2, dim=-1)  # (num_masked, 2)
        
        # Margin = |p(j1) - p(j2)| = top1 - top2 (since top1 >= top2)
        margins = top2_probs[:, 0] - top2_probs[:, 1]  # (num_masked,)
        
        # Add Gumbel noise
        if gumbel_noise_coeff > 0:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(margins) + 1e-8) + 1e-8)
            margins = margins + gumbel_noise_coeff * gumbel_noise
        
        # Select top K positions by margin (highest margin = most certain)
        _, top_k_indices = torch.topk(margins, K)
        positions_to_unmask = masked_positions[top_k_indices]
        
        # Sample tokens
        for idx, pos in zip(top_k_indices, positions_to_unmask):
            probs = probs_at_masked[idx]
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    # Handle remaining
    remaining_mask = (x[0] == mask_token_id)
    if remaining_mask.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining_mask.nonzero(as_tuple=True)[0]:
            probs = F.softmax(logits[0, pos, 1:], dim=-1)
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    return x[0]
```

### 8.4 Temperature-augmented Oracle for Text Data (Appendix D.1.2)

For text generation (Figure 3), the oracle includes Gaussian noise to prevent greedy sampling:

```python
def adaptive_inference_with_temperature(model, seq_len, vocab_size, num_steps, 
                                        mask_token_id=0, device='cuda',
                                        noise_std=1.0):
    """
    From Appendix D.1.2:
    F(θ, x_t) = Top K(|p_θ(x^i = j1|x_t) - p_θ(x^i = j2|x_t)| + ε)
    where ε is Gaussian noise.
    
    "adding a certain level of temperature to the oracle is useful... 
    the top probability margin or the top probability often leads to 
    greedy sampling, which harms the diversity (entropy)"
    """
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        alpha_t = 1.0 - t
        alpha_s = 1.0 - s
        
        masked_positions = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            break
        
        num_masked = len(masked_positions)
        unmask_frac = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
        K = max(1, int(round(num_masked * unmask_frac)))
        K = min(K, num_masked)
        
        with torch.no_grad():
            logits = model(x)
        
        probs_at_masked = F.softmax(logits[0, masked_positions, 1:], dim=-1)
        top2_probs, _ = torch.topk(probs_at_masked, k=2, dim=-1)
        margins = top2_probs[:, 0] - top2_probs[:, 1]
        
        # Add GAUSSIAN noise (not Gumbel — this is the text data variant)
        epsilon = torch.randn_like(margins) * noise_std
        margins = margins + epsilon
        
        _, top_k_indices = torch.topk(margins, K)
        positions_to_unmask = masked_positions[top_k_indices]
        
        for idx, pos in zip(top_k_indices, positions_to_unmask):
            probs = probs_at_masked[idx]
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    remaining_mask = (x[0] == mask_token_id)
    if remaining_mask.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining_mask.nonzero(as_tuple=True)[0]:
            probs = F.softmax(logits[0, pos, 1:], dim=-1)
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    return x[0]
```

---

## 9. REPRODUCING EACH TABLE AND FIGURE

### 9.1 Table 1: L&O-NAE-SAT Vanilla vs Adaptive Accuracy

**Target values**:
| (N, P) | Vanilla inference | Adaptive inference |
|--------|------------------|--------------------|
| (25, 275) | 78.06% | 93.76% |
| (30, 270) | 75.70% | 93.54% |
| (40, 260) | 74.60% | 92.21% |
| (50, 250) | 67.94% | 90.01% |
| (100, 200) | 62.84% | 88.91% |

**Metric**: Accuracy in predicting observation tokens.

**Protocol**:
1. For each (N, P) configuration with total L=300 and m=2:
   a. Generate training data (sufficient samples — at least 100K)
   b. Train 19M MDM
   c. Run vanilla inference on test puzzles
   d. Run adaptive inference (top probability margin) on test puzzles
   e. Measure: fraction of observation tokens correctly predicted

```python
def evaluate_lo_nae_sat(model, test_data, N, P, num_steps=50, 
                         strategy='vanilla', mask_token_id=0):
    """
    Evaluate MDM on L&O-NAE-SAT distribution.
    
    For each test sample:
    1. Mask ALL tokens (both latent and observation)
    2. Run inference (vanilla or adaptive)
    3. Check if observation tokens match ground truth
    
    Returns: accuracy (fraction of observation tokens correctly predicted)
    """
    correct = 0
    total = 0
    
    for x0 in test_data:
        if strategy == 'vanilla':
            x_pred = vanilla_mdm_inference(model, len(x0), vocab_size=3, 
                                           num_steps=num_steps, mask_token_id=mask_token_id)
        elif strategy == 'top_prob_margin':
            x_pred = adaptive_mdm_inference_top_prob_margin(
                model, len(x0), vocab_size=3, num_steps=num_steps, 
                mask_token_id=mask_token_id, gumbel_noise_coeff=0.5)
        
        # Check observation tokens (positions N to N+P-1)
        obs_correct = (x_pred[N:N+P] == torch.tensor(x0[N:N+P])).sum().item()
        correct += obs_correct
        total += P
    
    return correct / total
```

### 9.2 Table 2: Sudoku Accuracy

**Target values**:
| Method | # Param | Accuracy |
|--------|---------|----------|
| ARM (w/o ordering) | 42M | 9.73% |
| ARM (with ordering) | | 87.18% |
| MDM (vanilla) | 6M | 6.88% |
| MDM (Top probability) | | 18.51% |
| MDM (Top prob. margin) | | 89.49% |

**Critical implementation detail for Sudoku**:
- A Sudoku puzzle has 81 cells. Given clues are FIXED (never masked).
- Only the empty cells are masked.
- "Correctly solved" means ALL 81 cells have the correct value.
- A single wrong cell = puzzle not solved.

```python
def evaluate_sudoku(model, test_puzzles, strategy='vanilla', num_steps=50):
    """
    Evaluate MDM on Sudoku puzzles.
    
    Each puzzle: (clues, solution) where clues has 0 for empty cells.
    
    Returns: fraction of puzzles completely and correctly solved.
    """
    solved = 0
    
    for clues, solution in test_puzzles:
        # Start from the clue-given state (partially masked)
        x = torch.tensor(clues, dtype=torch.long, device='cuda').unsqueeze(0)  # (1, 81)
        
        # Run inference only on masked positions
        if strategy == 'vanilla':
            x_final = conditional_vanilla_inference(model, x, mask_token_id=0, 
                                                     num_steps=num_steps)
        elif strategy == 'top_prob':
            x_final = conditional_adaptive_inference(model, x, mask_token_id=0,
                                                      num_steps=num_steps,
                                                      strategy='top_prob',
                                                      gumbel_coeff=0.5)
        elif strategy == 'top_prob_margin':
            x_final = conditional_adaptive_inference(model, x, mask_token_id=0,
                                                      num_steps=num_steps,
                                                      strategy='top_prob_margin',
                                                      gumbel_coeff=0.5)
        
        # Check if completely correct
        if torch.all(x_final == torch.tensor(solution, device='cuda')):
            solved += 1
    
    return solved / len(test_puzzles)

def conditional_adaptive_inference(model, x_init, mask_token_id, num_steps, 
                                    strategy, gumbel_coeff=0.0):
    """
    Adaptive inference for conditional generation (given fixed clues).
    
    IMPORTANT: Only unmask positions that were originally masked.
    Never modify the given clue positions.
    """
    x = x_init.clone()
    fixed_positions = (x_init[0] != mask_token_id)  # Positions with given clues
    
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    for step in range(num_steps):
        t = timesteps[step].item()
        s = timesteps[step + 1].item()
        alpha_t = 1.0 - t
        alpha_s = 1.0 - s
        
        # Only consider currently masked positions (that were originally empty)
        masked_positions = ((x[0] == mask_token_id) & ~fixed_positions).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            break
        
        num_masked = len(masked_positions)
        unmask_frac = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
        K = max(1, int(round(num_masked * unmask_frac)))
        K = min(K, num_masked)
        
        with torch.no_grad():
            logits = model(x)
        
        probs_at_masked = F.softmax(logits[0, masked_positions, 1:], dim=-1)
        # For Sudoku, vocab is {1,...,9}, so indices 1-9 in the model
        
        if strategy == 'top_prob':
            scores = probs_at_masked.max(dim=-1).values
        elif strategy == 'top_prob_margin':
            top2, _ = torch.topk(probs_at_masked, k=2, dim=-1)
            scores = top2[:, 0] - top2[:, 1]
        
        if gumbel_coeff > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            scores = scores + gumbel_coeff * gumbel
        
        _, top_k_idx = torch.topk(scores, K)
        
        for idx in top_k_idx:
            pos = masked_positions[idx]
            probs = probs_at_masked[idx]
            token = torch.multinomial(probs, 1).item() + 1  # +1 for 1-indexed vocab
            x[0, pos] = token
    
    # Fill remaining
    remaining = ((x[0] == mask_token_id) & ~fixed_positions)
    if remaining.any():
        with torch.no_grad():
            logits = model(x)
        for pos in remaining.nonzero(as_tuple=True)[0]:
            probs = F.softmax(logits[0, pos, 1:], dim=-1)
            token = torch.multinomial(probs, 1).item() + 1
            x[0, pos] = token
    
    return x[0]
```

### 9.3 Table 3: Zebra Puzzle Accuracy

**Target values**:
| Method | # Param | Accuracy |
|--------|---------|----------|
| ARM (w/o ordering) | 42M | 80.31% |
| ARM (with ordering) | | 91.17% |
| MDM (vanilla) | 19M | 76.9% |
| MDM (Top probability) | | 98.5% |
| MDM (Top prob. margin) | | 98.3% |

Same protocol as Sudoku but with 19M model and Zebra puzzle data from Shah et al. (2024).

### 9.4 Table 4: LLaDA 8B Results

**Target values**:
| Method | HumanEval-Single | HumanEval-Multi | HumanEval-Split | Math | MMLU | ROCStories |
|--------|-----------------|----------------|----------------|------|------|------------|
| Vanilla | 31.8% | 16.5% | 14.2% | 28.5% | 33.2% | 21.23% |
| Top probability | 32.9% | 20.8% | 18.4% | 31.3% | 36.5% | 21.10% |
| Top prob. margin | 33.5% | 25.4% | 22.3% | 34.3% | 35.4% | 21.41% |

**Key details from Appendix D.3**:

- **Two task categories**:
  1. **Infilling** (HumanEval-Infill, ROCStories): Non-autoregressive sampling. Output length = size of masked span.
  2. **Instruction-answering** (Math, MMLU): Semi-autoregressive sampling strategy. Output length must be specified explicitly.

- **HumanEval-Infill**: Uses the problem set from Bavarian et al. (2022). Three categories by masked span:
  - Single-line: one line masked
  - Multi-line: multiple lines masked  
  - Split: non-contiguous regions masked

- For semi-autoregressive tasks, follow the sampling configuration of Nie et al. (2025) — the LLaDA paper's inference protocol.

```python
# Load LLaDA 8B
from transformers import AutoModel, AutoTokenizer

# Exact loading depends on the LLaDA codebase
# Follow instructions at https://github.com/ML-GSAI/LLaDA

def evaluate_llada_humaneval_infill(model, tokenizer, problems, strategy='vanilla'):
    """
    Evaluate on HumanEval-Infill benchmark.
    
    For each problem:
    1. Tokenize prefix + [MASK]*span_length + suffix
    2. Run MDM inference on masked span
    3. Detokenize and evaluate with test cases
    """
    pass_count = 0
    
    for problem in problems:
        prefix_tokens = tokenizer.encode(problem['prefix'])
        suffix_tokens = tokenizer.encode(problem['suffix'])
        span_length = len(tokenizer.encode(problem['canonical_solution']))
        
        # Create masked input
        mask_token_id = tokenizer.mask_token_id  # or whatever LLaDA uses
        input_ids = prefix_tokens + [mask_token_id] * span_length + suffix_tokens
        input_tensor = torch.tensor([input_ids], device='cuda')
        
        # Mark which positions are fixed (prefix and suffix)
        fixed = torch.ones(len(input_ids), dtype=torch.bool)
        fixed[len(prefix_tokens):len(prefix_tokens) + span_length] = False
        
        # Run inference
        if strategy == 'vanilla':
            output = vanilla_mdm_inference_conditional(model, input_tensor, fixed, ...)
        elif strategy == 'top_prob':
            output = adaptive_inference_conditional(model, input_tensor, fixed, 
                                                     strategy='top_prob', ...)
        elif strategy == 'top_prob_margin':
            output = adaptive_inference_conditional(model, input_tensor, fixed,
                                                     strategy='top_prob_margin', ...)
        
        # Decode and evaluate
        generated_text = tokenizer.decode(output[len(prefix_tokens):len(prefix_tokens) + span_length])
        
        if run_test_cases(problem, generated_text):
            pass_count += 1
    
    return pass_count / len(problems)
```

### 9.5 Table 5: Hard Sudoku Generalization

**Target values**:
| Method | # Param | Accuracy |
|--------|---------|----------|
| ARM (with ordering) | 42M | 32.57% |
| MDM (random) | 6M | 3.62% |
| MDM (Top probability) | | 9.44% |
| MDM (Top prob. margin) | | 49.88% |

**Key point**: Same models as Table 2 (trained on easy puzzles), but evaluated on hard puzzles. No retraining.

### 9.6 Figure 3: Generative Perplexity

**What to plot**: 
- x-axis: Sampling Steps (250 to 2000)
- Left y-axis: Generative Perplexity (GenPPL)
- Right y-axis: Entropy
- Lines: Gen PPL (Vanilla), Gen PPL (Adaptive), Entropy (Vanilla), Entropy (Adaptive)

**From Appendix D.1.2**:
- Model: **1.1B MDM** pretrained on text data (Note: caption says 170M but Appendix D.1.2 says 1.1B — use 1.1B as the appendix is more precise)
- Evaluator: **LLaMA-2 7B** (Touvron et al., 2023)
- Protocol:
  1. Generate unconditional text samples using vanilla and adaptive inference
  2. Compute likelihood of generated text using LLaMA-2 7B
  3. Generative Perplexity = exp(-mean(log-likelihood per token)) as computed by LLaMA-2 7B
  4. Entropy = Σ p_i log p_i where p_i = #{x^i = i} / L (unigram token frequency in generated text)
  5. Vary number of sampling steps: [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

```python
def compute_generative_perplexity(generated_texts, eval_model, eval_tokenizer):
    """
    Compute generative perplexity using LLaMA-2 7B as evaluator.
    
    GenPPL = exp(-1/N * Σ log p_eval(x_i | x_{<i}))
    where p_eval is the evaluation LM (LLaMA-2 7B).
    """
    total_nll = 0.0
    total_tokens = 0
    
    for text in generated_texts:
        inputs = eval_tokenizer(text, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = eval_model(**inputs, labels=inputs.input_ids)
        total_nll += outputs.loss.item() * inputs.input_ids.shape[1]
        total_tokens += inputs.input_ids.shape[1]
    
    avg_nll = total_nll / total_tokens
    gen_ppl = np.exp(avg_nll)
    return gen_ppl

def compute_entropy(generated_text_tokens, vocab_size):
    """
    Compute unigram entropy of generated text.
    
    Entropy = -Σ p_i log p_i
    where p_i = count(token_i) / total_tokens
    """
    counts = np.bincount(generated_text_tokens, minlength=vocab_size)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)
    entropy = -np.sum(probs * np.log(probs))
    return entropy
```

### 9.7 Figure 4 (Appendix): Belief Propagation for Planted CSP

**What to plot**: x-axis = D/k (average degree / arity), y-axis = overlap
- Two lines: "planted init" (BP initialized at ground truth), "random init" (BP initialized randomly)
- Parameters: k=3, m=3, g=NAE, N=10000

This is a numerical experiment demonstrating the phase transition in planted CSP recovery.

```python
def belief_propagation_planted_csp(N, k, m, D_over_k_values, g_func, num_trials=10):
    """
    Run belief propagation for planted CSP with NAE predicate.
    
    For each average degree D:
    1. Sample planted assignment σ ~ Uniform({1,...,m}^N)
    2. Generate random k-tuples and include each with probability φ/N^{k-1}
       if g(σ|tuple) = 1
    3. Run BP from planted initialization and random initialization
    4. Compute overlap of BP output with ground truth
    
    NAE(x1, x2, x3) = 1 - 1[x1 = x2 = x3]
    """
    results = {'planted_init': [], 'random_init': []}
    
    for D_over_k in D_over_k_values:
        D = D_over_k * k
        phi = D / k  # clause density parameter
        
        overlaps_planted = []
        overlaps_random = []
        
        for trial in range(num_trials):
            # Sample ground truth assignment
            sigma = np.random.randint(1, m + 1, size=N)
            
            # Generate clauses
            # Expected number of clauses: P ≈ φ * N^{k-1} * γ
            # where γ = P(NAE satisfied by random) = 1 - m/m^k = 1 - 1/m^{k-1}
            # For m=3, k=3: γ = 1 - 1/9 = 8/9
            clauses = generate_planted_csp_clauses(N, k, m, phi, sigma, g_func)
            
            # Run BP from planted init
            overlap_p = run_bp(N, m, clauses, sigma, init='planted', max_iter=100)
            overlaps_planted.append(overlap_p)
            
            # Run BP from random init
            overlap_r = run_bp(N, m, clauses, sigma, init='random', max_iter=100)
            overlaps_random.append(overlap_r)
        
        results['planted_init'].append(np.mean(overlaps_planted))
        results['random_init'].append(np.mean(overlaps_random))
    
    return results

def nae_predicate(x):
    """NAE(x1, x2, x3) = 1 - 1[x1 = x2 = x3]"""
    return 0 if (x[0] == x[1] == x[2]) else 1
```

The BP update rules are given in Definition B.10. Implement these exactly:

```python
def run_bp(N, m, clauses, sigma, init='random', max_iter=100, damping=0.5):
    """
    Belief propagation for planted CSP (Definition B.10).
    
    Messages:
    - M^c_{i→S}[t]: variable-to-clause message (variable i to clause S, color c)
    - M^c_{S→i}[t]: clause-to-variable message (clause S to variable i, color c)
    
    Update rules:
    M^c_{i→S}[t+1] ∝ Π_{T: i∈T, T≠S} M^c_{T→i}[t]          (Eq. 4)
    M^c_{S→i}[t+1] ∝ Σ_{σ ∈ {1,...,m}^{S\i}} g(σ ∪_i c) Π_{j: j∈S, j≠i} M^{σ_j}_{j→S}[t]   (Eq. 5)
    
    Initialization:
    - planted: messages biased toward ground truth
    - random: messages = 1/m (paramagnetic fixed point) + small noise
    
    Overlap: d(σ, σ_hat) = min_π Σ_i 1[σ_i = π(σ_hat_i)]
    """
    # Initialize messages
    # M_var_to_clause[(i, S_idx)][c] for c in {1,...,m}
    # M_clause_to_var[(S_idx, i)][c]
    
    var_to_clause = {}
    clause_to_var = {}
    
    # Build adjacency: for each variable, which clauses contain it
    var_clauses = {i: [] for i in range(N)}
    for s_idx, clause in enumerate(clauses):
        for i in clause:
            var_clauses[i].append(s_idx)
    
    # Initialize
    for s_idx, clause in enumerate(clauses):
        for i in clause:
            if init == 'random':
                msg = np.ones(m) / m + np.random.randn(m) * 0.01
            elif init == 'planted':
                msg = np.zeros(m)
                msg[sigma[i] - 1] = 0.9
                msg += 0.1 / m
            msg = msg / msg.sum()
            var_to_clause[(i, s_idx)] = msg.copy()
            clause_to_var[(s_idx, i)] = np.ones(m) / m
    
    # Iterate BP
    for iteration in range(max_iter):
        # Update clause-to-variable messages (Eq. 5)
        new_clause_to_var = {}
        for s_idx, clause in enumerate(clauses):
            for i in clause:
                msg = np.zeros(m)
                other_vars = [j for j in clause if j != i]
                
                # Sum over assignments to other variables
                # For k=3, this is a sum over m^2 assignments
                for c in range(m):
                    total = 0.0
                    if len(other_vars) == 2:
                        for c1 in range(m):
                            for c2 in range(m):
                                assignment = {}
                                assignment[i] = c + 1
                                assignment[other_vars[0]] = c1 + 1
                                assignment[other_vars[1]] = c2 + 1
                                values = [assignment[v] for v in clause]
                                if nae_predicate(values):
                                    total += (var_to_clause[(other_vars[0], s_idx)][c1] * 
                                             var_to_clause[(other_vars[1], s_idx)][c2])
                    msg[c] = total
                
                msg = msg / (msg.sum() + 1e-10)
                new_clause_to_var[(s_idx, i)] = msg
        
        # Apply damping
        for key in clause_to_var:
            clause_to_var[key] = (damping * clause_to_var[key] + 
                                  (1 - damping) * new_clause_to_var[key])
        
        # Update variable-to-clause messages (Eq. 4)
        for i in range(N):
            for s_idx in var_clauses[i]:
                msg = np.ones(m)
                for t_idx in var_clauses[i]:
                    if t_idx != s_idx:
                        msg *= clause_to_var[(t_idx, i)]
                msg = msg / (msg.sum() + 1e-10)
                var_to_clause[(i, s_idx)] = msg
    
    # Compute marginals and round to assignment
    sigma_hat = np.zeros(N, dtype=int)
    for i in range(N):
        marginal = np.ones(m)
        for s_idx in var_clauses[i]:
            marginal *= clause_to_var[(s_idx, i)]
        sigma_hat[i] = np.argmax(marginal) + 1
    
    # Compute overlap (considering color permutation symmetry)
    from itertools import permutations
    best_overlap = 0
    for perm in permutations(range(1, m + 1)):
        mapped = np.array([perm[s - 1] for s in sigma_hat])
        overlap = np.mean(mapped == sigma)
        best_overlap = max(best_overlap, overlap)
    
    return best_overlap
```

---

## 10. PLOTTING SPECIFICATIONS

### 10.1 Figure 2 Left: Scaling Laws
```python
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(6, 5))

# Plot each line
ax.plot(log_flops_ar, neg_loglik_ar, 'o-', color='tab:orange', label='AR', linewidth=2)
ax.plot(log_flops_mdm, neg_loglik_mdm, 's-', color='tab:blue', label='MDM', linewidth=2)
# Plot π-learners (3 lines each from Closer, Much-Closer, Uniform)
for i, (flops, nll) in enumerate(pi_learner_much_closer_data):
    label = 'π-learner-much-closer' if i == 0 else None
    ax.plot(flops, nll, '^-', color='tab:green', label=label, alpha=0.7)
for i, (flops, nll) in enumerate(pi_learner_closer_data):
    label = 'π-learner-closer' if i == 0 else None
    ax.plot(flops, nll, 'v-', color='tab:red', label=label, alpha=0.7)
for i, (flops, nll) in enumerate(pi_learner_unif_data):
    label = 'π-learner-unif' if i == 0 else None
    ax.plot(flops, nll, 'D-', color='tab:purple', label=label, alpha=0.7)

ax.set_xlabel('log(FLOPs)')
ax.set_ylabel('-log $p_\\theta(x)$')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_left.pdf', dpi=300, bbox_inches='tight')
```

### 10.2 Figure 2 Right Bottom: Error Imbalance
```python
fig, ax = plt.subplots(figsize=(6, 3))

# x-axis: position 0 to N+P-1 = 299
# y-axis: prediction error
# Color: light for latent positions (0 to N-1), darker for observation positions (N to N+P-1)

# Use a colorbar to indicate error magnitude
positions = np.arange(N + P)
ax.bar(positions[:N], errors[:N], color='lightcoral', label='Latent positions')
ax.bar(positions[N:], errors[N:], color='steelblue', label='Observation positions')

ax.set_xlabel('Position')
ax.set_ylabel('Prediction Error')
ax.legend()
plt.tight_layout()
plt.savefig('figure2_right_bottom.pdf', dpi=300, bbox_inches='tight')
```

### 10.3 Figure 3: Generative Perplexity
```python
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

steps = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

ax1.plot(steps, gen_ppl_vanilla, 'o-', color='tab:orange', label='Gen PPL (Vanilla Inference)')
ax1.plot(steps, gen_ppl_adaptive, 's-', color='tab:blue', label='Gen PPL (Adaptive Inference)')
ax2.plot(steps, entropy_vanilla, 'o--', color='tab:orange', label='Entropy (Vanilla Inference)', alpha=0.6)
ax2.plot(steps, entropy_adaptive, 's--', color='tab:blue', label='Entropy (Adaptive Inference)', alpha=0.6)

ax1.set_xlabel('Sampling Steps')
ax1.set_ylabel('Generative Perplexity')
ax2.set_ylabel('Entropy')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('figure3.pdf', dpi=300, bbox_inches='tight')
```

### 10.4 Figure 4: Belief Propagation Overlap
```python
fig, ax = plt.subplots(figsize=(6, 5))

D_over_k_values = np.linspace(43, 68, 50)

ax.plot(D_over_k_values, overlaps_planted, 'o-', label='planted init', markersize=3)
ax.plot(D_over_k_values, overlaps_random, 's-', label='random init', markersize=3)

ax.set_xlabel('D / k')
ax.set_ylabel('overlap')
ax.legend()
ax.grid(True, alpha=0.3)

# Mark the phase transition region
# DKS/K = 64 (analytically, from caption)
# Dcond/K ≈ 50 (from the plot)
ax.axvline(x=64, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('figure4.pdf', dpi=300, bbox_inches='tight')
```

---

## 11. VALIDATION CHECKPOINTS

Use these to verify your implementation is correct before running full experiments:

### Quick Sanity Checks:

1. **L&O-NAE-SAT naive accuracy = 75%**: Generate 10K samples with m=2, predict observation tokens randomly. Accuracy should be ~75%.

2. **MDM loss converges**: On a small dataset, MDM training loss should decrease steadily.

3. **Vanilla MDM < Adaptive MDM**: Even on a toy dataset, adaptive inference should outperform vanilla.

4. **Top prob margin ≥ Top prob**: On Sudoku/Zebra, margin strategy should match or beat top probability.

5. **ARM with identity permutation is best on text**: For π-learners on SlimPajama, identity should give lowest loss.

---

## 12. PHASE 2 CHECKLIST

- [ ] Vanilla MDM inference implemented and tested
- [ ] Top Probability adaptive inference implemented
- [ ] Top Probability Margin adaptive inference implemented
- [ ] Gumbel noise addition for puzzle inference working
- [ ] Gaussian noise oracle for text inference working
- [ ] K (number to unmask per step) calculation matches paper formula
- [ ] Conditional inference (fixed clues) for Sudoku/Zebra working
- [ ] Generative perplexity computation with LLaMA-2 7B working
- [ ] Entropy computation working
- [ ] Belief propagation for planted CSP implemented
- [ ] All plotting code ready
- [ ] Evaluation scripts for all tables ready

---

*Continue to Phase 3 for the Figure 1 conceptual diagram, additional appendix experiments, and final integration.*