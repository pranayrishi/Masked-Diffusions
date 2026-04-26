# Reproduction Guide: "Train for the Worst, Plan for the Best"
## Phase 3: Noise Schedules, Figure 1, Full Integration Pipeline, and Troubleshooting

---

## 13. NOISE SCHEDULE DETAILS

The noise schedule α_t is a critical component that the paper does not fully specify for every experiment. Here we compile what is known and provide justified defaults.

### 13.1 General Framework

The forward process masks token i with probability (1 - α_t), where:
- α_0 ≈ 1 (clean data, almost no masking)
- α_1 ≈ 0 (fully masked)

The paper uses Proposition 2.1's assumption: α_0 = 1, α_1 = 0.

### 13.2 Noise Schedule Options

**Linear schedule** (most commonly used in MDM literature, e.g., Sahoo et al. 2025):
```python
def linear_noise_schedule(t):
    """α_t = 1 - t"""
    return 1.0 - t
```

**Cosine schedule** (sometimes used, from Austin et al. 2021):
```python
import math
def cosine_noise_schedule(t, s=0.008):
    """α_t = cos((t + s)/(1 + s) * π/2)²"""
    f_t = math.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    f_0 = math.cos((s / (1 + s)) * math.pi / 2) ** 2
    return f_t / f_0
```

**What the paper likely uses**: The Nie et al. (2024) codebase (SMDM) uses a **linear schedule** by default. The Ye et al. (2024) codebase (for puzzles) also uses linear. Use linear schedule unless the codebase you're building on specifies otherwise.

### 13.3 Discrete Reverse Steps

For inference with T reverse steps, divide [0, 1] into T equal intervals:

```python
def get_timesteps(num_steps):
    """
    Get the sequence of noise levels for reverse sampling.
    
    We go from t=1 (fully masked) to t=0 (clean) in num_steps steps.
    Returns: list of (t, s) pairs where t > s.
    """
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    pairs = [(ts[i].item(), ts[i+1].item()) for i in range(num_steps)]
    return pairs
```

---

## 14. FIGURE 1 REPRODUCTION (Conceptual Diagram)

Figure 1 is a **conceptual/schematic figure** (not data-driven). It has two parts:

### 14.1 Top Part: "MDM Training"

Shows the masking forward process from t=0 to t=1.

**Content**:
- x-axis: time from t=0 (left, clean) to t=1 (right, fully masked)
- Show a 3-token sequence (e.g., tokens "a", "b", "c")
- At different noise levels, show various masking patterns:
  - t near 0: few tokens masked (e.g., "a d M", "a M f", "M b c")
  - t near 0.5: more masks (e.g., "M b M", "M M c")
  - t near 1: almost all masked (e.g., "a M M", "M M M")
- Arrows between patterns showing the forward process
- Annotations showing some patterns are "harder" (e.g., predicting latent tokens from observations)

**Implementation**: This should be created as a vector graphic (SVG or TikZ/LaTeX). Use matplotlib with annotations:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])

# --- TOP: MDM Training ---
ax_top.set_title('MDM training', fontsize=14, fontweight='bold')
ax_top.set_xlim(-0.1, 1.1)
ax_top.set_ylim(-0.5, 4.5)
ax_top.set_xlabel('$t = 0$ (clean)' + ' ' * 30 + '$t = 1$ (masked)')

# Draw sequences at various noise levels
def draw_token_box(ax, x, y, text, is_masked=False, width=0.08, height=0.3):
    color = 'lightgray' if is_masked else 'white'
    rect = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                     boxstyle="round,pad=0.02", 
                                     facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    display_text = 'M' if is_masked else text
    ax.text(x, y, display_text, ha='center', va='center', fontsize=9, 
            fontweight='bold' if is_masked else 'normal')

# Example sequences at different noise levels
# Near t=0 (clean side)
sequences_t0 = [
    (0.1, 3.5, [('a', False), ('d', False), ('M', True)]),
    (0.1, 2.8, [('a', False), ('M', True), ('f', False)]),
    (0.1, 2.1, [('M', True), ('b', False), ('c', False)]),
    (0.1, 1.4, [('M', True), ('b', False), ('c', False)]),
    (0.1, 0.7, [('a', False), ('M', True), ('c', False)]),
]

# Near t=0.5 
sequences_t05 = [
    (0.5, 3.0, [('M', True), ('b', False), ('M', True)]),
    (0.5, 2.0, [('a', False), ('M', True), ('M', True)]),
    (0.5, 1.0, [('M', True), ('M', True), ('c', False)]),
]

# Near t=1 (masked side)
sequences_t1 = [
    (0.9, 2.5, [('M', True), ('M', True), ('M', True)]),
]

for x_base, y, tokens in sequences_t0 + sequences_t05 + sequences_t1:
    for i, (text, masked) in enumerate(tokens):
        draw_token_box(ax_top, x_base + i * 0.1, y, text, masked)

ax_top.set_aspect('equal')
ax_top.axis('off')

# --- BOTTOM: MDM Inferences (Vanilla vs. Adaptive) ---
ax_bottom.set_title('MDM inferences (Vanilla vs. Adaptive)', fontsize=14, fontweight='bold')
# Similar drawing showing two paths from fully masked to clean
# Vanilla: random unmasking order
# Adaptive: strategic unmasking (easy tokens first)

plt.tight_layout()
plt.savefig('figure1.pdf', dpi=300, bbox_inches='tight')
```

**Note**: Figure 1 is best reproduced in LaTeX with TikZ for publication quality. The matplotlib version above is a starting point; for the actual paper, use TikZ.

### 14.2 Bottom Part: "MDM Inferences (Vanilla vs. Adaptive)"

Shows two rows:
1. **Vanilla**: Random unmasking leads to different (potentially wrong) results
2. **Adaptive**: Strategic unmasking leads to correct results

Each row shows 4 stages from fully masked (t=1) to fully unmasked (t=0):
- Stage 1: [M M M] / [M M M]
- Stage 2: [M b c] / [M b M] 
- Stage 3: [M b c] / [a b M]
- Stage 4: [g b c] (wrong!) / [a b c] (correct!)

---

## 15. COMPLETE EXPERIMENTAL PIPELINE

### 15.1 Master Execution Script

```bash
#!/bin/bash
# master_run.sh — Execute all experiments in order

# ============================================
# PHASE A: Data Preparation
# ============================================

echo "=== Phase A: Data Preparation ==="

# 1. Download and tokenize SlimPajama
python scripts/prepare_slimpajama.py \
    --output_dir data/slimpajama \
    --tokenizer gpt2 \
    --seq_len 2048 \
    --num_tokens 10B  # Adjust based on compute budget

# 2. Generate L&O-NAE-SAT data
python scripts/generate_lo_nae_sat.py \
    --configs "20,280" "25,275" "30,270" "40,260" "50,250" "100,200" \
    --alphabet_size 2 \
    --num_train 100000 \
    --num_test 10000 \
    --output_dir data/lo_nae_sat

# 3. Download Sudoku data
python scripts/prepare_sudoku.py \
    --kaggle_path data/sudoku-3m.csv \
    --output_dir data/sudoku \
    --strategy_filter_script scripts/sudoku_strategies.py

# 4. Download Zebra data
python scripts/prepare_zebra.py \
    --source_repo "https://github.com/shah-kulin/logic-puzzles" \
    --output_dir data/zebra

# ============================================
# PHASE B: Training
# ============================================

echo "=== Phase B: Training ==="

# 5. Scaling law experiments (Figure 2 Left)
# Train ARM, MDM, and π-learners at various compute budgets
for FLOPS in 1e18 2e18 5e18 1e19 2e19 5e19 1e20; do
    for MODEL_SIZE in 10M 30M 60M 100M 170M 300M; do
        # ARM (identity permutation)
        python train_pi_learner.py \
            --data_dir data/slimpajama \
            --model_size $MODEL_SIZE \
            --flops_budget $FLOPS \
            --permutation identity \
            --pos_embedding learned \
            --output_dir models/scaling_laws/arm_${MODEL_SIZE}_${FLOPS}
        
        # MDM
        python train_mdm.py \
            --data_dir data/slimpajama \
            --model_size $MODEL_SIZE \
            --flops_budget $FLOPS \
            --pos_embedding learned \
            --output_dir models/scaling_laws/mdm_${MODEL_SIZE}_${FLOPS}
        
        # π-learners (3 samples from each distribution)
        for DIST in uniform closer much_closer; do
            for SEED in 1 2 3; do
                python train_pi_learner.py \
                    --data_dir data/slimpajama \
                    --model_size $MODEL_SIZE \
                    --flops_budget $FLOPS \
                    --permutation $DIST \
                    --perm_seed $SEED \
                    --pos_embedding learned \
                    --output_dir models/scaling_laws/pi_${DIST}_s${SEED}_${MODEL_SIZE}_${FLOPS}
            done
        done
    done
done

# 6. L&O-NAE-SAT models
# Error imbalance experiment (Figure 2 Right Bottom)
python train_mdm.py \
    --data_dir data/lo_nae_sat/N20_P280 \
    --model_size 19M \
    --max_seq_len 512 \
    --pos_embedding rope \
    --num_iterations 2000 \
    --output_dir models/lo_nae_sat/N20_P280_model

python train_mdm.py \
    --data_dir data/lo_nae_sat/N20_P280 \
    --model_size 19M \
    --max_seq_len 512 \
    --pos_embedding rope \
    --num_iterations 50000 \
    --output_dir models/lo_nae_sat/N20_P280_proxy

# Table 1 models
for CONFIG in "25,275" "30,270" "40,260" "50,250" "100,200"; do
    N=$(echo $CONFIG | cut -d',' -f1)
    P=$(echo $CONFIG | cut -d',' -f2)
    python train_mdm.py \
        --data_dir data/lo_nae_sat/N${N}_P${P} \
        --model_size 19M \
        --output_dir models/lo_nae_sat/N${N}_P${P}_model
done

# 7. Sudoku models
python train_mdm.py \
    --data_dir data/sudoku/train \
    --model_size 6M \
    --architecture gpt2 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_epochs 300 \
    --output_dir models/sudoku/mdm_6M

python train_arm.py \
    --data_dir data/sudoku/train \
    --model_size 42M \
    --architecture gpt2 \
    --with_ordering false \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_epochs 300 \
    --output_dir models/sudoku/arm_42M_no_order

python train_arm.py \
    --data_dir data/sudoku/train \
    --model_size 42M \
    --architecture gpt2 \
    --with_ordering true \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_epochs 300 \
    --output_dir models/sudoku/arm_42M_with_order

# 8. Zebra models (same structure, 19M MDM)
python train_mdm.py \
    --data_dir data/zebra/train \
    --model_size 19M \
    --architecture gpt2 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_epochs 300 \
    --output_dir models/zebra/mdm_19M

python train_arm.py \
    --data_dir data/zebra/train \
    --model_size 42M \
    --with_ordering false \
    --output_dir models/zebra/arm_42M_no_order

python train_arm.py \
    --data_dir data/zebra/train \
    --model_size 42M \
    --with_ordering true \
    --output_dir models/zebra/arm_42M_with_order

# 9. Text generation model (Figure 3)
# Use a pretrained 1.1B MDM — either train or download
# The paper says "1.1B MDM pretrained on text data"
# This is likely from the Nie et al. (2024) codebase

# ============================================
# PHASE C: Evaluation / Inference
# ============================================

echo "=== Phase C: Evaluation ==="

# 10. Scaling laws evaluation — pick best model per compute budget
python evaluate_scaling_laws.py \
    --models_dir models/scaling_laws \
    --data_dir data/slimpajama \
    --output_file results/figure2_left.json

# 11. L&O-NAE-SAT error imbalance
python evaluate_error_imbalance.py \
    --model_path models/lo_nae_sat/N20_P280_model \
    --proxy_path models/lo_nae_sat/N20_P280_proxy \
    --data_dir data/lo_nae_sat/N20_P280 \
    --N 20 --P 280 --ell 11 --num_trials 1000 \
    --output_file results/figure2_right_bottom.json

# 12. L&O-NAE-SAT vanilla vs adaptive (Table 1)
for CONFIG in "25,275" "30,270" "40,260" "50,250" "100,200"; do
    N=$(echo $CONFIG | cut -d',' -f1)
    P=$(echo $CONFIG | cut -d',' -f2)
    python evaluate_lo_nae_sat.py \
        --model_path models/lo_nae_sat/N${N}_P${P}_model \
        --data_dir data/lo_nae_sat/N${N}_P${P} \
        --N $N --P $P \
        --strategies vanilla top_prob_margin \
        --num_steps 50 \
        --output_file results/table1_N${N}_P${P}.json
done

# 13. Sudoku evaluation (Table 2)
python evaluate_sudoku.py \
    --mdm_model models/sudoku/mdm_6M \
    --arm_model_no_order models/sudoku/arm_42M_no_order \
    --arm_model_with_order models/sudoku/arm_42M_with_order \
    --test_data data/sudoku/test_easy \
    --strategies vanilla top_prob top_prob_margin \
    --num_steps 50 --gumbel_coeff 0.5 \
    --output_file results/table2.json

# 14. Hard Sudoku evaluation (Table 5)
python evaluate_sudoku.py \
    --mdm_model models/sudoku/mdm_6M \
    --arm_model_with_order models/sudoku/arm_42M_with_order \
    --test_data data/sudoku/test_hard \
    --strategies vanilla top_prob top_prob_margin \
    --num_steps 50 --gumbel_coeff 0.5 \
    --output_file results/table5.json

# 15. Zebra evaluation (Table 3)
python evaluate_zebra.py \
    --mdm_model models/zebra/mdm_19M \
    --arm_model_no_order models/zebra/arm_42M_no_order \
    --arm_model_with_order models/zebra/arm_42M_with_order \
    --test_data data/zebra/test \
    --strategies vanilla top_prob top_prob_margin \
    --num_steps 50 --gumbel_coeff 0.5 \
    --output_file results/table3.json

# 16. Text generation perplexity (Figure 3)
for NUM_STEPS in 250 500 750 1000 1250 1500 1750 2000; do
    python evaluate_text_generation.py \
        --mdm_model models/text/mdm_1.1B \
        --eval_model meta-llama/Llama-2-7b \
        --num_steps $NUM_STEPS \
        --num_samples 100 \
        --strategies vanilla adaptive \
        --noise_std 1.0 \
        --output_file results/figure3_steps${NUM_STEPS}.json
done

# 17. LLaDA 8B evaluation (Table 4)
python evaluate_llada.py \
    --model_path llada-8b \
    --benchmarks humaneval_single humaneval_multi humaneval_split math mmlu rocstories \
    --strategies vanilla top_prob top_prob_margin \
    --output_file results/table4.json

# 18. Belief propagation experiment (Figure 4)
python run_belief_propagation.py \
    --N 10000 --k 3 --m 3 --predicate NAE \
    --D_over_k_range 43 68 \
    --num_points 50 --num_trials 10 \
    --output_file results/figure4.json

# ============================================
# PHASE D: Generate Figures and Tables
# ============================================

echo "=== Phase D: Plotting ==="

python plot_all_figures.py \
    --results_dir results \
    --output_dir figures
```

---

## 16. CRITICAL IMPLEMENTATION DETAILS & GOTCHAS

### 16.1 The Mask Token Convention

The paper uses **0** as the mask token. In your vocabulary:
- Token 0 = MASK
- Tokens 1 to m = actual vocabulary values

For text experiments using GPT-2 tokenizer (vocab size 50257), you need to add a mask token:
```python
# Option 1: Reserve token 0 for mask, shift all vocab tokens by 1
# This means model output dimension = 50258
MASK_TOKEN_ID = 0
VOCAB_SIZE = 50258  # 50257 GPT-2 tokens + 1 mask token

# Option 2: Use an existing special token as mask
# Check what the Nie et al. (2024) codebase does and follow that
```

### 16.2 Time-Embedding-Free Architecture

Section 2 explicitly states the network does NOT take t as input: "a time-embedding-free architecture for the denoising network, i.e., p_θ(·|x_t, t) = p_θ(·|x_t) is generally used as x_t implicitly contains information about t via the number of masked tokens."

**This means**: Your transformer model takes only the token sequence as input. No time/noise-level embedding. The model learns to infer the noise level from the number of mask tokens.

### 16.3 Causal vs Bidirectional Attention

- **ARM and π-learners**: Use **causal attention** (standard autoregressive mask). The model can only attend to tokens at earlier positions.
- **MDM**: Use **bidirectional attention** (no causal mask). The model can attend to all positions, since it needs to use information from unmasked tokens at any position.

This is a fundamental architectural difference. The MDM uses a BERT-like bidirectional transformer, while ARM uses a GPT-like causal transformer.

### 16.4 Sudoku Token Representation

For Sudoku:
- Sequence length: 81 (9×9 grid, row-major order)
- Vocabulary: {0, 1, 2, ..., 9} where 0 = mask, 1-9 = Sudoku digits
- Given clues are pre-filled; empty cells start as 0 (mask)
- The model's job during inference: fill in all 0s with correct digits

### 16.5 ARM with Ordering: How to Implement

The "ARM (with ordering)" baseline from Shah et al. (2024):

1. For each training puzzle, compute a valid solving order using constraint propagation
2. The solving order is a permutation of the empty cell positions
3. Reorder the sequence so that cells are in solving order
4. Train a standard left-to-right ARM on this reordered sequence
5. At inference, the model generates cells in the learned order

This is NOT a trivial implementation. You need:
- A Sudoku solver that outputs the solving order (which cell was determined at each step)
- The Shah et al. (2024) codebase likely provides this

### 16.6 Gumbel Noise Implementation Detail

From Appendix D.2: "add Gumbel noise with a coefficient of 0.5 to the MDM inference oracle F"

The Gumbel noise is added to the **certainty scores** (not to the token probabilities). It introduces randomness in which positions get unmasked, preventing the deterministic greedy selection from being stuck.

```python
# Correct: add Gumbel noise to the oracle SCORES
gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
perturbed_scores = scores + 0.5 * gumbel  # coefficient = 0.5

# Then select top-K from perturbed_scores
_, top_k = torch.topk(perturbed_scores, K)
```

### 16.7 Semi-Autoregressive Inference for LLaDA (Appendix D.3)

For instruction-answering tasks (Math, MMLU) with LLaDA 8B:

The paper says: "For instruction–answering tasks, we employ a semi-autoregressive sampling strategy."

This means:
1. The prompt (instruction) is given as unmasked prefix
2. The response area has a fixed number of mask tokens
3. Inference proceeds as normal MDM reverse process on the masked area
4. The "semi-autoregressive" part: likely chunks the response into blocks and generates each block sequentially, or uses the LLaDA paper's specific inference protocol

Follow the LLaDA codebase (Nie et al., 2025) for the exact semi-autoregressive inference implementation.

---

## 17. TROUBLESHOOTING COMMON ISSUES

### 17.1 MDM Training Loss Not Converging
- Check that mask token ID is correctly implemented (all masked positions should predict the ORIGINAL token, not the mask)
- Ensure bidirectional attention is being used (not causal)
- Verify the loss weighting: divide by number of masked tokens per sample

### 17.2 Sudoku Accuracy Much Lower Than Expected
- Check that clue positions are NEVER modified during inference
- Ensure the vocabulary is correct ({0,...,9} with 0=mask)
- Verify num_steps=50 and gumbel_coeff=0.5
- Ensure the training data filter matches Shah et al. (2024)'s 7-strategy filter

### 17.3 Adaptive Inference Not Improving Over Vanilla
- Verify that K (tokens per step) matches the paper's formula
- Check that Gumbel noise coefficient is appropriate (0.5 for puzzles, Gaussian for text)
- Ensure scores are computed correctly (margin = top1 - top2, not absolute values of two arbitrary probs)

### 17.4 Generative Perplexity Too High
- Ensure you're using the correct evaluation model (LLaMA-2 7B, not a smaller model)
- Check that generated text is properly detokenized before evaluation
- Verify entropy is in the expected range (~4.6-5.4 per Figure 3)

### 17.5 Scaling Laws Don't Show Expected Pattern
- Make sure you're using **learnable positional embeddings, not RoPE** for scaling law experiments
- Verify IsoFLOP methodology: for each compute budget, vary model size and pick BEST
- Check that π-learners are using the correct permutation (verify by printing first few permuted positions)

---

## 18. EXPECTED TIMELINE AND COMPUTE BUDGET

| Experiment | GPU Hours (est.) | Priority |
|-----------|-----------------|----------|
| L&O-NAE-SAT (all configs) | 10-20 A100-hrs | High |
| Sudoku MDM 6M | 5-10 A100-hrs | High |
| Sudoku ARM 42M (×2) | 10-20 A100-hrs | High |
| Zebra MDM 19M + ARMs | 15-25 A100-hrs | High |
| Scaling laws (all configs) | 200-500 A100-hrs | Medium |
| Text gen perplexity (Fig 3) | 20-40 A100-hrs | Medium |
| LLaDA 8B evaluation | 10-20 A100-hrs | Medium |
| Belief propagation (Fig 4) | 2-5 CPU-hrs | Low |

**Total estimated**: ~300-650 A100-GPU-hours for full reproduction.

**Recommendation**: Start with L&O-NAE-SAT and Sudoku experiments (fastest, highest impact) to validate the pipeline, then move to scaling laws and LLaDA.

---

## 19. FILE STRUCTURE

```
project/
├── README.md
├── master_run.sh
├── environment.yml
├── configs/
│   ├── scaling_laws.yaml
│   ├── lo_nae_sat.yaml
│   ├── sudoku.yaml
│   ├── zebra.yaml
│   └── llada.yaml
├── data/
│   ├── slimpajama/
│   ├── lo_nae_sat/
│   │   ├── N20_P280/
│   │   ├── N25_P275/
│   │   └── ...
│   ├── sudoku/
│   │   ├── train/
│   │   ├── test_easy/
│   │   └── test_hard/
│   └── zebra/
├── models/
│   ├── architectures/
│   │   ├── transformer.py        # Shared transformer with learnable/RoPE pos embed
│   │   ├── mdm_model.py          # MDM wrapper (bidirectional attention)
│   │   └── arm_model.py          # ARM wrapper (causal attention)
│   ├── training/
│   │   ├── train_mdm.py          # MDM training loop
│   │   ├── train_arm.py          # ARM training loop
│   │   └── train_pi_learner.py   # π-learner training loop
│   └── inference/
│       ├── vanilla_inference.py   # Vanilla MDM inference
│       ├── adaptive_inference.py  # Top-prob and top-prob-margin inference
│       └── arm_inference.py       # ARM generation
├── scripts/
│   ├── prepare_slimpajama.py
│   ├── generate_lo_nae_sat.py
│   ├── prepare_sudoku.py
│   ├── prepare_zebra.py
│   ├── sudoku_strategies.py      # 7-strategy Sudoku solver
│   └── permutation_utils.py      # Permutation sampling (identity, uniform, closer, etc.)
├── evaluation/
│   ├── evaluate_scaling_laws.py
│   ├── evaluate_error_imbalance.py
│   ├── evaluate_lo_nae_sat.py
│   ├── evaluate_sudoku.py
│   ├── evaluate_zebra.py
│   ├── evaluate_text_generation.py
│   ├── evaluate_llada.py
│   └── run_belief_propagation.py
├── plotting/
│   ├── plot_figure1.py           # Conceptual diagram
│   ├── plot_figure2.py           # Scaling laws + error imbalance
│   ├── plot_figure3.py           # Generative perplexity
│   ├── plot_figure4.py           # Belief propagation
│   └── plot_all_tables.py        # LaTeX tables
├── results/
│   └── (JSON files from evaluation)
└── figures/
    └── (PDF/PNG figures)
```

---

## 20. FINAL PHASE 3 CHECKLIST

- [ ] Noise schedule implemented (linear, matching codebases)
- [ ] Figure 1 conceptual diagram created
- [ ] Master execution script ready
- [ ] All data generation scripts ready
- [ ] All training scripts ready
- [ ] All evaluation scripts ready
- [ ] All plotting scripts ready
- [ ] Causal vs bidirectional attention correctly applied per model type
- [ ] Mask token convention consistent across all experiments
- [ ] Gumbel noise for puzzles, Gaussian noise for text — correctly applied
- [ ] Semi-autoregressive inference for LLaDA instruction tasks
- [ ] ARM with ordering baseline correctly implements Shah et al. solving order
- [ ] IsoFLOP methodology for scaling laws correctly implemented
- [ ] File structure organized per Section 19
- [ ] Expected timeline and compute budget estimated

---

## 21. KEY REFERENCES TO HAVE OPEN

1. **This paper**: Kim et al. (2025). arXiv:2502.06768v3
2. **Nie et al. (2024)**: "Scaling up masked diffusion models on text." arXiv:2410.18514 — for the scaling law codebase
3. **Ye et al. (2024)**: "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning." arXiv:2410.14157 — for the Sudoku/Zebra codebase
4. **Shah et al. (2024)**: "Causal language modeling can elicit search and reasoning capabilities on logic puzzles." arXiv:2409.10502 — for the Sudoku/Zebra datasets and ARM-with-ordering baseline
5. **Nie et al. (2025)**: "Large Language Diffusion Models." arXiv:2502.09992 — for LLaDA 8B
6. **Sahoo et al. (2025)**: "Simple and effective masked diffusion language models." NeurIPS — for MDM training details
7. **Zheng et al. (2024)**: "Masked diffusion models are secretly time-agnostic masked models." arXiv:2409.02908 — for the loss equivalence and time-free architecture
8. **Radcliffe (2020)**: "3 million Sudoku puzzles with ratings." Kaggle — for Sudoku data
9. **Hoffmann et al. (2022)**: "Training compute-optimal large language models." (Chinchilla paper) — for IsoFLOP analysis methodology