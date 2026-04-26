# Sudoku scope decision

> **Context.** Per the user's Phase 6 instruction, before writing any Slurm scripts we need to decide whether Sudoku (Tables 2 and 5) is in scope for the first paper or follows up. The decision turns on whether the **ARM-with-ordering** baseline can be vendored from upstream or must be re-implemented from scratch (which the user has explicitly forbidden — "do not reimplement from the paper description alone").

> **Method.** Cloned both candidate repos and read their training scripts, dataset adapters, and tokenization code:
> - `https://github.com/HKUNLP/diffusion-vs-ar` (Ye et al. 2024, the codebase the paper uses for puzzle MDM training).
> - `https://github.com/kulinshah98/llm-reasoning-logic-puzzles` (Shah et al. 2024 — note the original URL `kulinshah98/logic-puzzles` is dead; the actual repo is `llm-reasoning-logic-puzzles`).
>
> Both clones are at `/tmp/mdm_scope/`. URLs verified 2026-04-26 by fetching the arXiv abstract pages and Google search. (`paper_notes.md` §11 and `code_audit.md` previously cited the wrong Shah URL — corrected here.)

---

## TL;DR — recommendation

**Defer Sudoku to a supplementary appendix or to a follow-up paper.** L&O-NAE-SAT alone (Table 1) is a complete reproduction story for the entropy-filtered modification, and the Sudoku ARM-with-ordering baseline requires a substantial port from Shah's JAX/Flax to our PyTorch package — roughly **1–2 weeks of focused work** with non-trivial debugging risk.

If we want Sudoku in v1 of the paper, the realistic path is:
1. Vendor Shah's downloaded `Sudoku-train-data.npy` (1.8M puzzles, with strategy-IDs and solver-order embedded).
2. Vendor Shah's data adapter (`sudoku-code/train/data.py`) but rewrite it in PyTorch.
3. Vendor Shah's GPT model class + training loop (also a JAX→PyTorch port).
4. Cross-validate against Shah's reported 94.21 % Sudoku accuracy before relying on it.
5. Wire it into our existing `baseline/src/train.py` so the same Slurm pipeline drives both L&O-NAE-SAT and Sudoku.

If we **defer** Sudoku, the v1 paper still:
- Reproduces Table 1 (5 (N, P) configs × vanilla + adaptive).
- Tests the entropy filter as a 5-variant ablation across all 5 (N, P) points.
- Demonstrates the paper's hardness theory (Prop 3.3) being borne out in our experiments.
- Shows the inference-fix vs. training-fix axis empirically.

That alone is a complete paper. Sudoku adds the headline number (89.49 % MDM-margin vs. 87.18 % ARM-with-ordering) but is not required for the modification's contribution.

---

## What each repo actually provides

### Ye et al. — `HKUNLP/diffusion-vs-ar`

**What it covers (relevant to Table 2):**

| Row in Table 2 | Ye et al. supports? |
|---|---|
| MDM (vanilla) | ✅ `scripts/sudoku/train-mdm.sh` |
| MDM (Top probability) | ⚠️  partially — `--decoding_strategy stochastic0.5-linear` is their inference oracle |
| MDM (Top prob. margin) | ❌  not implemented; we add this |
| ARM (w/o ordering) | ✅ `scripts/sudoku/train-sft.sh` |
| **ARM (with ordering)** | **❌  not implemented** |

**Format & framework:**
- Built on **LLaMA-Factory**; `src/train_bash.py` is the entry point.
- Sudoku data is **text** ("space-separated tokens"), `cutoff_len = 164` (input + answer ≤ 164 tokens). Format inferred from `metric.py`:
  ```python
  pred = pred.strip().split(' ')   # chess, sudoku, prime branch
  label = label.strip().split(' ')
  ```
- Data download: 1.5 GB Google Drive bundle (https://drive.google.com/file/d/1b0OIlYL76rVVuNYIfIb-L_Ptdg6k_y0c/view). Not in the repo.
- Model sizes: `model_config_tiny` (~6M), `model_config` (~85M), `model_config_medium` (~303M).
- **Three model_config directories** with `config.json`, `tokenizer_config.json`, `special_tokens_map.json` — all GPT-2-style tokenizer.

**Training hyperparameters** (from `train-mdm.sh`):
- batch_size = 128 per device × 8 devices = 1024 effective.
- lr = 1e-3, cosine schedule, 300 epochs (matches paper Appendix D.2).
- `--diffusion_steps 20` for MDM training, `--alpha 0.25 --gamma 1` (this is Multi-Granularity Diffusion Modeling parameters; the paper Kim et al. 2025 likely uses the standard MDM, i.e., MGDM with α=γ=0 or similar).
- Decoding: `--decoding_strategy stochastic0.5-linear` (Top-K with stochastic noise, similar to but not identical to Top-Probability).

**What we'd need to do to use it:**
1. Vendor the data download (~1 day to wire in).
2. Adapt our `Transformer` to load Ye et al.'s tokenizer / config (or use `transformers.AutoModel` directly).
3. Replace their decoding with our `top_prob` and `top_prob_margin` oracles (~1 day).
4. Match their training hyperparameters in our `baseline/configs/sudoku_mdm.yaml`.

**What it does NOT provide:** the ARM-with-ordering baseline. This is the gap.

---

### Shah et al. — `kulinshah98/llm-reasoning-logic-puzzles`

**What it covers (relevant to Table 2):**

| Row in Table 2 | Shah et al. supports? |
|---|---|
| ARM (with ordering, "solver-order") | ✅ `--config.seq_order = 'solver-order'` |
| ARM (w/o ordering, "fixed-order") | ✅ `--config.seq_order = 'fixed'` |
| ARM (random ordering) | ✅ `--config.seq_order = 'random'` |
| MDM | ❌  paper is causal-LM-only |

**Format & framework:**
- **JAX / Flax**, custom GPT model (`sudoku-code/train/model.py`, `trainer.py`).
- Sudoku token format is **(row, col, value) triples — 3 tokens per cell, NOT 2**:

  ```python
  # data.py:199 (cited verbatim from upstream)
  # For each cell of a Sudoku puzzle, there is (row, column, value) in each train_input sequence.
  # Each train_input sequence is 243 size long (= 81 cells × 3 positions).
  ```

  The paper-notes / audit had this as 162 tokens (row, value) — that was wrong. The correct length is **243** (row, col, value).

- **Strategy IDs are embedded in the data**: each cell has a fourth piece of information: a strategy ID `∈ {0=given, 2=Lone single, 3=Hidden single, 4=Naked pair, 5=Naked Triplet, 6=Locked Candidate, 7=XY Wing, 8=Unique Rectangle}`. So the 7-strategy filter is **already done** by Shah upstream — we don't need to re-implement it.
- Data download: Google Drive (`Sudoku-train-data.npy`, 1.8M puzzles; `Sudoku-test-data.npy`, 100K).
- Trainer in JAX/Flax, optimizer in Optax.

**Solver-order details (the critical piece for ARM-with-ordering):**

```python
# data.py — config.seq_order branch
"solver-order"   →  cells emitted in the order the 7-strategy solver determined them
"fixed"          →  row-major order (= no ordering signal)
"random"         →  random permutation per puzzle
```

Each input sequence is 243 tokens long, with the first `3 × start_index` tokens corresponding to the given clues (in original order), then the remaining `3 × (81 − start_index)` tokens for the cells the solver determines, in solver order.

**Loss masking** (from `trainer.py:208`):
```python
mask = (mask >= 3 * start_index)
```
Loss is computed only on the **non-clue** cells (the model is not penalized for echoing the clues). This is consistent with paper semantics.

---

## Why "ARM with ordering" is hard to vendor

To reproduce the **89.49 % MDM-margin vs. 87.18 % ARM-with-ordering** comparison faithfully, we need an ARM trained on Shah's solver-order sequences. Three porting options, each with risk:

### Option A: Run Shah's JAX training directly on Bouchet

- ✅ Most faithful — Shah's exact code, exact hyperparameters.
- ❌ Bouchet's primary stack is PyTorch. JAX needs `jax`, `flax`, `optax`, `jaxlib` with the right CUDA build. Doable but adds an environment.
- ❌ Their checkpoint format won't interop with our PyTorch evaluation pipeline.
- ❌ Two parallel codebases for the project = double maintenance.
- **Time to first run on Bouchet**: 2–3 days (env setup + data wiring + Slurm).

### Option B: Port Shah's training loop to PyTorch (and reuse our `Transformer`)

- ✅ Keeps the project in one codebase.
- ✅ Lets us reuse seed plumbing, JSONL logging, checkpoint/resume.
- ❌ Their model has subtle differences from our `Transformer`: different positional embedding (learned absolute, like our `pos_type='learned'`), different LR schedule defaults, different optimizer (Optax AdamW with their specific eps).
- ❌ Cross-validating against their reported 94.21 % Sudoku accuracy is the only way to verify the port; this is **the** validation gate.
- **Time to first matched-accuracy run**: 1–2 weeks.

### Option C: Train our own ARM-with-ordering on a re-derived solving order

- ❌ Forbidden by user instruction: "do not reimplement from the paper description alone."
- Even if allowed, the 7-strategy solver in `baseline/src/sudoku.py` would need to match Shah's exactly — which is hard to verify without their reference output.
- **Not on the table.**

---

## Recommended decision tree (for the professor)

```
Is the headline 89.49 % vs 87.18 % comparison required for paper acceptance?
├── YES → adopt Option B (1–2 week port). Schedule it before Bouchet runs.
│         Side benefit: also lets us reproduce Shah's 94.21 % as a sanity check.
│
└── NO  → ship v1 with L&O-NAE-SAT + entropy filter only.
          Sudoku becomes a planned follow-up paper or supplementary appendix.
          Bouchet runs scale: 5 (N,P) × 5 variants × 3 seeds = 75 jobs at small scale.
          Total v1 GPU-hour estimate (L&O-NAE-SAT only): ~150–250 GPU-hours.
          With Sudoku: ~200–400 GPU-hours additional.
```

The authors of the paper already had the headline result; **we are testing a modification** (entropy filtering), and the modification can be evaluated cleanly on L&O-NAE-SAT without Sudoku. The Sudoku result reproduces the *paper's* claim, not the *modification's* claim. So Sudoku is more about verifying we can reproduce the original paper than about evaluating our extension.

---

## What I'm doing in the meantime

In `baseline/src/sudoku.py` I left:

- The 7-strategy solver and tokenization helpers (`solve_with_seven_strategies`, `sudoku_string_to_tokens`, etc.).
- A `synthetic_easy_puzzle` factory used by tests.
- A `filter_and_split` function for processing the Radcliffe Kaggle CSV.

These are useful for sanity-checking against Shah's annotations (do our 7-strategy IDs match Shah's strategy IDs on a sample?) but **are NOT used in any training run** in the current Phase 4/5 deliverable.

Phase 7 Slurm scripts will only schedule L&O-NAE-SAT runs until the professor decides on Sudoku scope.

---

## Citations

- Kim, Shah, Kontonis, Kakade, Chen. *Train for the Worst, Plan for the Best.* ICML 2025. [arXiv:2502.06768](https://arxiv.org/abs/2502.06768).
- Ye, Gao, Gong, Zheng, Jiang, Li, Kong. *Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning.* [arXiv:2410.14157](https://arxiv.org/abs/2410.14157). Code: [HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar).
- Shah, Dikkala, Wang, Panigrahy. *Causal Language Modeling Can Elicit Search and Reasoning Capabilities on Logic Puzzles.* NeurIPS 2024. [arXiv:2409.10502](https://arxiv.org/abs/2409.10502). Code: [kulinshah98/llm-reasoning-logic-puzzles](https://github.com/kulinshah98/llm-reasoning-logic-puzzles).
- Radcliffe, D. G. *3 Million Sudoku Puzzles with Ratings.* Kaggle, 2020. [DOI 10.5281/zenodo.5148524](https://www.kaggle.com/dsv/1495975).
