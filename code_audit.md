# Code Audit — Pre-existing Reproduction Notebooks

> **Scope.** The seven `Colab_*.ipynb` files (≈ 4,886 non-blank code lines) audited against `paper_notes.md` (the source of truth, Phase 2 output).
> **Method.** Cell-by-cell read of every code cell that defines a model, training step, inference function, or data loader. Verified critical equations against the paper.
> **Date.** 2026-04-26.
> **Run history.** None of the seven notebooks has been executed end-to-end. Colab 1 partially ran and hit two known data-prep failures (already patched). Colabs 2–7 have empty cell outputs.

---

## Findings at a glance

| Severity | Count | What they are |
|---|---|---|
| **P0** (blockers — will produce wrong results) | **3** | wrong Ye et al. URL; broken ARM-with-ordering inference; ARM-with-ordering training format does not encode positions |
| **P1** (correctness — runs but deviates from paper) | **5** | vanilla inference uses Top-K instead of Bernoulli; LLaDA oracle has no noise; SMDM tokenizer assumption; min lr; no-op permutation swaps |
| **P2** (quality — correct but unfit for Bouchet) | **11** | Colab paths everywhere; no tests; no checkpoint/resume; no requirements.txt; no seed plumbing; per-position Python loops; no logging |

The notebooks **enumerate the right experiments and encode (mostly) the right hyperparameters**. They are **not trustworthy as a runnable baseline** because of the P0 bugs and the absence of any infrastructure (tests, configs, checkpoints, logging) needed to actually run on Bouchet and produce paper-quality results.

**Recommendation: rewrite from scratch for Phase 4.** Mine the notebooks for specific algorithm pieces that are paper-correct (L&O-NAE-SAT generator, Sudoku 7-strategy filter, BP message passing, inference oracle code) but build training loops, model wrappers, configs, and tests fresh as a proper Python package. Detailed argument in §6 below.

---

## 1. P0 findings — will produce wrong results

### P0-1 · Wrong Ye et al. 2024 codebase URL in Colab 1

**Where.** `Colab_1_Data_Prep.ipynb`, code cell 6 (the cell that clones reference codebases):

```python
!git clone https://github.com/HKUNLP/discrete-diffusion.git
```

**What's wrong.** Verified in Phase 2 by fetching the arXiv abstract page for Ye et al. 2024 (arXiv:2410.14157, "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning"): the canonical repo is **`https://github.com/HKUNLP/diffusion-vs-ar`**, NOT `discrete-diffusion`. The wrong URL will either 404 or — worse — silently clone a different repo whose defaults do not match the paper.

**Severity.** P0 because the paper says (Appendix D.2): *"For the training and inference, we use the codebase of (Ye et al., 2024) with keeping most of the hyperparameters default given in the codebase."* If you train on a different codebase's defaults, you are not reproducing the paper.

**Cross-reference.** `paper_notes.md` §11 (verified codebase URLs).

**Fix.** In the Phase 4 baseline, clone `https://github.com/HKUNLP/diffusion-vs-ar`. Inspect their `configs/` and confirm Sudoku/Zebra hyperparameters before training.

---

### P0-2 · Colab 3 ARM-with-ordering inference walks the wrong order

**Where.** `Colab_3_Sudoku_Zebra.ipynb`, the function `arm_inference`:

```python
def arm_inference(model, clues=None, seq_len=81, vocab_size=10, device='cuda'):
    """Standard left-to-right ARM generation."""
    model.eval()
    x = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    if clues is not None:
        x[0] = torch.tensor(clues, dtype=torch.long, device=device)
    
    # Generate left-to-right, skipping given clues
    for pos in range(seq_len):
        if clues is not None and clues[pos] != 0:
            continue
        logits = model(x[:, :pos+1])
        ...
```

**What's wrong.** Training in `make_sudoku_arm_loader(with_ordering=True)` reorders the sequence as `clue_positions + solved_positions + remaining`. So the model is trained to emit:
1. token-at-clue-pos-0, token-at-clue-pos-1, ..., (the clues, in original order)
2. token-at-solve-step-0, token-at-solve-step-1, ..., (the cells in their solving order)

But `arm_inference` walks `for pos in range(81)` — the **grid order**, not the **solve order**. So at inference the model is asked to predict in an order it was never trained on. Accuracy will collapse, almost certainly below the without-ordering baseline. Paper expectation: 87.18 %.

**Severity.** P0. This is the single most important comparison in the paper (the headline result is "MDM-adaptive 89.49 % > ARM-with-ordering 87.18 %"). Getting it wrong invalidates the entire Sudoku reproduction.

**Cross-reference.** `paper_notes.md` §12.9 (ARM-with-ordering baseline definition).

**Fix.** Two parts:
1. **Inference.** Walk positions in the same order used during training: `for pos in clue_positions + solved_positions + remaining`. Map generated tokens back to grid positions via the inverse permutation.
2. **At test time we don't have the solve order** (it was a training-only annotation). Either (a) recompute the solve order on the test puzzle using the same 7 strategies, or (b) train the model to predict the next position as well as the next value. Shah et al. 2024 do option (b) — see the next finding.

---

### P0-3 · Colab 3 ARM-with-ordering training format omits position information

**Where.** `Colab_3_Sudoku_Zebra.ipynb`, `make_sudoku_arm_loader(with_ordering=True)`:

```python
ordered_positions = clue_positions + solved_positions + remaining
reordered = solution[ordered_positions]
batch.append(reordered)
yield torch.tensor(np.array(batch), dtype=torch.long)
```

**What's wrong.** The yielded sequence is just digits 1-9, in solve order. The model has no way to know **which grid cell** each digit refers to. At inference (even with the order fix above), the model would emit a sequence like `(7, 3, 1, 8, ...)` but cannot tell us "the 7 goes at row 2 col 5".

**Shah et al. 2024 (arXiv:2409.10502, the paper that built this dataset)** do **alternating (position, value)** tokens: the sequence becomes `pos_0, val_0, pos_1, val_1, ...`. With 81 cells, sequence length is up to 162 tokens. The model learns a joint distribution over (position, value).

**Severity.** P0. Even with the inference fix in P0-2, training data is malformed and the model will not learn what's needed.

**Cross-reference.** `paper_notes.md` §12.9.

**Fix.** Phase 4 must follow Shah et al.'s alternating-token format exactly. **First check whether Ye et al.'s `diffusion-vs-ar` codebase ships with their version of this baseline** — re-implementing it from scratch is risky.

---

## 2. P1 findings — runs but deviates from paper

### P1-1 · Vanilla MDM inference uses fixed-K Top-K, not per-position Bernoulli

**Where.** `Colab_2_LO_NAE_SAT.ipynb` (`run_mdm_inference`), `Colab_3_Sudoku_Zebra.ipynb` (`puzzle_mdm_inference`), `Colab_5_Text_GenPPL.ipynb` (`mdm_text_inference`), `Colab_6_LLaDA_8B.ipynb` (`infill_inference`, `semi_autoregressive_inference`).

The vanilla branch:

```python
if strategy == 'vanilla':
    perm = torch.randperm(num_masked, device=device)[:K]
    selected_positions = masked_positions[perm]
```

selects **exactly K** masked positions uniformly at random.

**What the paper says** (Section 2.1.2):

> "Sample a set `S ⊆ {i : x_t^i = 0}` of currently-masked positions to unmask. Each masked position is included in `S` independently with probability `(α_s − α_t) / (1 − α_t)`."

So vanilla is **Bernoulli per position**. The notebook's implementation matches the **adaptive** algorithm's Top-K (with K = expected count) but applied to vanilla.

**Severity.** P1. In expectation the two are identical. In variance they differ — Bernoulli has variance `K · (1 − K/num_masked)` while fixed-K has zero variance per step. For Section 4.2 puzzle accuracy this difference is small. But it is a paper deviation and easy to fix.

**Cross-reference.** `paper_notes.md` §2.6.

**Fix.** Vanilla branch should use `torch.bernoulli(torch.full(num_masked, p))` where `p = (α_s − α_t) / (1 − α_t)`. Adaptive branches keep Top-K.

---

### P1-2 · Colab 6 LLaDA inference defaults `gumbel_coeff = 0.0`

**Where.** `Colab_6_LLaDA_8B.ipynb`:

```python
@torch.no_grad()
def infill_inference(..., gumbel_coeff=0.0, ...):
@torch.no_grad()
def semi_autoregressive_inference(..., gumbel_coeff=0.0, ...):
```

**What's wrong.** With `gumbel_coeff = 0.0`, Top-Probability and Top-Probability-Margin are fully deterministic for any given input. The paper does not explicitly state the LLaDA oracle noise (Appendix D.3 omits it), but the text experiments (Appendix D.1.2) use Gaussian noise on margins to avoid greedy collapse. With **zero noise**, multi-token unmasking can deterministically select the same K positions every step, harming Math/MMLU performance.

**Severity.** P1. Without noise, the upper-bound for Top-Prob-Margin is achievable but variance across runs vanishes — making it hard to reproduce paper results that may rely on the average over noisy oracles.

**Cross-reference.** `paper_notes.md` §3.4.

**Fix.** Default `gumbel_coeff = 0.5` for puzzles, Gaussian σ = 0.001 for text. For LLaDA-8B on text-like benchmarks, default to Gaussian σ = 0.001. Make it a config flag.

---

### P1-3 · Colab 5 uses LLaMA-vocab mask id, but paper says GPT-2 tokenizer for SMDM

**Where.** `Colab_5_Text_GenPPL.ipynb`:

```python
MASK_TOKEN_ID = 32000
VOCAB_SIZE = 32001
```

**What's wrong.** 32000 is LLaMA's vocabulary size. SMDM's published 1.1B checkpoint may indeed use the LLaMA tokenizer (TinyLlama-based), or it may use GPT-2 (50257) per the paper's textual claim. We have not yet verified which. Loading the wrong tokenizer means generated text is gibberish and GenPPL is meaningless.

**Severity.** P1. Trivially fatal for Figure 3 if wrong; trivially correct if right. We will not know until we clone SMDM.

**Cross-reference.** `paper_notes.md` §9.4 (notes the open ambiguity).

**Fix.** When cloning the SMDM repo for Phase 4 (or for the deferred Figure 3 reproduction), inspect the published 1.1B checkpoint's tokenizer config and configure accordingly. Add an explicit assert at load time: `assert tokenizer.vocab_size == VOCAB_SIZE - 1`.

---

### P1-4 · Colab 5 uses lr = 3e-4 instead of paper's 4e-4

**Where.** `Colab_5_Text_GenPPL.ipynb`. The paper's Appendix C.1 standardizes max lr = 4 × 10⁻⁴ "throughout the paper unless otherwise specified."

**Severity.** P1. Minor — 25 % below the paper's lr. Would slightly slow convergence and may produce a different validation loss. Not catastrophic for Figure 3 (which relies on a pretrained checkpoint, not training from scratch in this notebook).

**Fix.** Use 4e-4 in Phase 4, with explicit citation in the config file.

---

### P1-5 · Permutation samplers in Colab 4 can produce no-op swaps

**Where.** `Colab_4_Scaling_Laws.ipynb`:

```python
def sample_closer_permutation(L, rng=None):
    perm = np.arange(L)
    n_swaps = L // 10
    for _ in range(n_swaps):
        i, j = rng.integers(0, L, size=2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm
```

**What's wrong.** `rng.integers(0, L, size=2)` may return `i == j`; the swap is then a no-op. Effective swap count is ≈ `n_swaps · (1 − 1/L)`.

**Severity.** P1 (negligible at L = 2048, but should be fixed). The paper says "L/10 random swapping operations" — with `L = 2048`, no-op rate is < 0.05 %. But for small L (e.g., the L&O-NAE-SAT L = 300, where the same code might be used), this matters more.

**Fix.** `i, j = rng.choice(L, size=2, replace=False)` — guarantees i ≠ j.

---

## 3. P2 findings — works but unfit for Bouchet

### P2-1 · Colab paths everywhere

`from google.colab import drive`, `drive.mount('/content/drive')`, `BASE_DIR = '/content/drive/MyDrive/mdm_reproduction'`. Pervasive across all 7 notebooks. Not portable to Bouchet.

**Fix.** Phase 4 reads paths from a config file or environment variables (`PROJECT_DIR`, `SCRATCH_DIR`).

### P2-2 · No `requirements.txt` / `pyproject.toml`

Dependencies installed inline via `!pip install`. Different versions get installed each session. Not reproducible.

**Fix.** Phase 4 ships a `requirements.txt` with pinned versions. Bouchet builds a Conda env from it.

### P2-3 · No checkpoint or resume

`train_mdm` saves once at the end. If interrupted (Bouchet preemption, walltime hit, or any error), all progress is lost.

**Fix.** Phase 4: every training script saves model + optimizer + scheduler + RNG state every N steps; supports `--resume <path>`.

### P2-4 · No structured logging

Only stdout `print` calls. No `metrics.jsonl`, no CSV, no wandb hooks. Cannot aggregate runs across seeds for a paper figure.

**Fix.** Phase 4: write one row per logging step to `metrics.jsonl` (step, loss, lr, grad-norm, time-per-step, GPU memory). Optional offline wandb integration.

### P2-5 · No tests

Zero unit tests. Forward process, loss formula, K computation, BP update rules — all have invariants checkable on toy data. Not a single one is verified.

**Fix.** Phase 4 ships `tests/` with at least:
- `test_lo_nae_sat.py` — generator output matches the worked example in `paper_notes.md` §5.4
- `test_loss.py` — MDM loss on a 4-token toy matches manual computation, including the 1/n weighting
- `test_inference.py` — vanilla and adaptive both produce valid sequences (no mask token in output, all clues preserved)
- `test_forward_process.py` — masked tokens have correct empirical frequency at each `α_t`

### P2-6 · No seed plumbing

`np.random.randint(0, dataset_size, size=batch_size)` uses global numpy RNG. Not seeded, not reproducible.

**Fix.** Phase 4: every script accepts `--seed`; calls `torch.manual_seed`, `np.random.seed`, `random.seed`; logs the seed in the checkpoint.

### P2-7 · Per-batch / per-position Python loops in hot paths

E.g. in Colab 2 `mdm_train_step`:

```python
for b in range(B):
    masked_logits = logits[b, mask[b]]
    masked_targets = x0[b, mask[b]]
    if len(masked_targets) > 0:
        sample_loss = F.cross_entropy(masked_logits, masked_targets, reduction='sum')
        total_loss = total_loss + sample_loss / n[b].float()
```

Slow (~10× throughput penalty at B = 128). Acceptable for prototyping; bad for IsoFLOP scaling-law experiments.

**Fix.** Vectorize: compute per-token CE once, then sum-per-sample with a `scatter_add`, then divide by per-sample `n`.

### P2-8 · No CPU fallback path

Models hardcoded to CUDA. Smoke testing on a laptop without GPU should still work for the L&O-NAE-SAT (5, 10) toy case.

**Fix.** Phase 4: `device = 'cuda' if torch.cuda.is_available() else 'cpu'` everywhere; smoke test runs on CPU.

### P2-9 · Per-position `torch.multinomial` calls in inference

E.g. in Colab 2:

```python
for idx in top_k:
    token = torch.multinomial(probs[idx], 1).item()
    x[0, masked_positions[idx]] = token
```

At LLaDA-8B scale (response length up to 256, batch 1), this is hundreds of GPU↔CPU sync points per inference call.

**Fix.** Batch the multinomial: `tokens = torch.multinomial(probs[top_k], 1).squeeze(-1)`, then `x[0, masked_positions[top_k]] = tokens`.

### P2-10 · Zebra dataset path is a stub

Colab 3 `if ZEBRA_AVAILABLE:` block depends on data the notebook never actually loads. Phase 1 inventory already flagged this. Acceptable since Phase 4 scope defers Zebra anyway.

### P2-11 · Belief-propagation cell in Colab 7 is correct but slow

Pure-Python loop over `m^(k-1)` assignments per clause per iteration. For `(k, m, num_clauses) = (3, 3, ~10⁵)` this is ~10⁹ inner ops. Hours per BP run.

**Fix.** Vectorize with NumPy or use `jit`. Phase 8 (deferred); not on the Phase 4 critical path.

---

## 4. Cross-cutting issues

### Inconsistent mask-token convention across notebooks

| Notebook | mask_token_id | vocab_size | Comment |
|---|---|---|---|
| Colab 2 (L&O-NAE-SAT) | 3 | 4 | Correct — avoids collision with observation 0 |
| Colab 3 (Sudoku/Zebra) | 0 | 10 | OK for Sudoku (digits 1–9 don't collide); needs verification for Zebra |
| Colab 4 (Scaling laws) | 0 | 50258 | Correct after `+1` shift of GPT-2 token IDs |
| Colab 5 (Text GenPPL) | 32000 | 32001 | Implies LLaMA tokenizer; see P1-3 |
| Colab 6 (LLaDA) | 126336 | (LLaDA's) | LLaDA's actual mask-token id; appears correct |
| Colab 7 (BP) | n/a | n/a | No tokens, just CSP variables |

Phase 4 must standardize: every dataset adapter declares its `(mask_token_id, vocab_size)` pair in its config, and every model checks them at load time.

### No shared library across notebooks

The `MDMTransformer`, `ARMTransformer`, `RoPE` classes are re-implemented in each notebook with subtle variations. Any bug fix has to be made multiple times. Phase 4 will live in one Python package.

### No package-level `BASE_DIR` configuration

Every notebook hardcodes `BASE_DIR = '/content/drive/MyDrive/mdm_reproduction'`. Phase 4 will read paths from a single `cluster_config.local.yaml` (gitignored).

---

## 5. Per-notebook short summaries

| Notebook | What it gets right | What it gets wrong |
|---|---|---|
| `Colab_1_Data_Prep.ipynb` | GPT-2 tokenizer, SlimPajama fallback, Kaggle dataset ID, 7-strategy Sudoku filter logic, LLaDA repo URL, Shah et al. repo URL, SMDM repo URL | **P0-1 wrong Ye et al. URL** (`discrete-diffusion`) |
| `Colab_2_LO_NAE_SAT.ipynb` | mask-token = 3, vocab = 4, RoPE, bidirectional attention, no time embedding, AdamW (β₂=0.95, wd=0.1), Gumbel coef 0.5, num_steps = 50, **proper 1/n loss weighting** | P1-1 vanilla uses Top-K; P2-7 per-batch Python loop |
| `Colab_3_Sudoku_Zebra.ipynb` | mask=0 (OK for Sudoku), conditional inference preserves clues, MDM bidirectional, ARM causal, all hyperparameters match paper | **P0-2 broken ARM-with-ordering inference**, **P0-3 ARM-with-ordering training omits positions** |
| `Colab_4_Scaling_Laws.ipynb` | Learnable pos emb (NOT RoPE), GPT-2 vocab + 1 shift, IsoFLOP infrastructure, weight tying, β₂ / wd / lr_max / lr_min all match paper | P1-5 no-op swaps; no IsoFLOP experiment has been run |
| `Colab_5_Text_GenPPL.ipynb` | Gaussian noise on text oracle, K formula, sampling-step sweep, GenPPL function | P1-3 vocab = 32001 implies LLaMA tokenizer; P1-4 lr = 3e-4 not 4e-4 |
| `Colab_6_LLaDA_8B.ipynb` | Semi-AR vs non-AR distinction, block_size = 32, num_steps = 64, mask-token = 126336 (LLaDA's actual value) | **P1-2 default gumbel_coeff = 0.0** (no oracle noise) |
| `Colab_7_BP_and_Plots.ipynb` | BP message updates correctly implement Eq 4 / Eq 5; planted/random init biased correctly; damping = 0.5; alphabet permutation overlap | P2-11 slow Python loop |

---

## 6. Recommendation: rewrite, don't patch

**Verdict: rewrite from scratch for Phase 4.**

Reasoning:

1. **Three P0s.** Two of them (P0-2, P0-3) are not patch-tractable in place — fixing ARM-with-ordering requires a fundamentally different training data format (alternating position/value tokens), which is incompatible with the existing notebook's data shapes and model `forward()` signatures.

2. **No tests = patch blind.** None of the existing functions has invariants checked. Every patch has unknown blast radius.

3. **Pervasive Colab-isms.** `from google.colab import drive`, `BASE_DIR = '/content/drive/...'`, `!pip install ...` cells. Stripping these is a rewrite in everything but name.

4. **No infrastructure for Bouchet.** No checkpoint/resume, no JSONL logs, no seed plumbing, no `pyproject.toml`. All required by the new mission.

5. **No shared package.** Every notebook re-implements the transformer, the RoPE block, the train step. Bug fixes must be made N times.

6. **Phase 5 modification needs clean hooks.** The entropy filter must be slotted into the training loop at a specific point (post-mask-creation, pre-loss). With the current notebook structure (per-batch Python loop with inline mask creation), the modification is awkward to graft and impossible to ablate cleanly.

What we **keep** from the notebooks (re-used as reference, not as imports):

- `paper_notes.md` is now the authoritative source.
- The L&O-NAE-SAT generator from Colab 1, Cell 10 — paper-correct, self-contained, easily ported.
- The 7-strategy Sudoku filter from Colab 1, Cells 13-14 — implements naked singles, hidden singles, naked pairs, hidden pairs, pointing pairs, box/line reduction, naked triples. Validate by counting filtered puzzles against Shah et al. 2024's published count.
- The MDM training-step skeleton from Colab 2 (the 1/n weighting is right; clean it up and vectorize).
- The conditional inference algorithm from Colab 2/3 (excluding the ARM-with-ordering bug and the vanilla Bernoulli fix).
- The BP message-passing logic from Colab 7 (vectorize after porting).
- The permutation samplers from Colab 4 (after the no-op-swap fix).

What we **discard** entirely:

- Colab 1 Sudoku CSV download cell (replaced by a script in Phase 4 that reads `KAGGLE_API_TOKEN` from env).
- All `!pip install ...` cells (replaced by `requirements.txt`).
- All `drive.mount` / `BASE_DIR = '/content/drive/...'` paths (replaced by config file).
- The Colab 3 ARM-with-ordering implementation (full rewrite required to follow Shah et al. 2024's alternating-token format).
- The Colab 6 LLaDA `gumbel_coeff = 0.0` default (replaced with σ = 0.001 Gaussian or 0.5 Gumbel as appropriate).
- Colabs 4, 5, 6 entirely (deferred — Phase 4 scope is Tables 1 & 2 only).

---

## 7. Phase 4 entry plan (preview, for user review at the Phase 4 kickoff)

The clean rewrite will be structured as:

```
Masked-Diffusions/
├── baseline/                       NEW (Phase 4)
│   ├── README.md
│   ├── pyproject.toml              pinned deps, package metadata
│   ├── configs/
│   │   ├── lo_nae_sat_25_275.yaml
│   │   ├── lo_nae_sat_30_270.yaml
│   │   ├── lo_nae_sat_40_260.yaml
│   │   ├── lo_nae_sat_50_250.yaml
│   │   ├── lo_nae_sat_100_200.yaml
│   │   ├── sudoku_mdm.yaml
│   │   └── sudoku_arm_with_order.yaml
│   ├── src/
│   │   ├── data/                   L&O-NAE-SAT generator + Sudoku filter
│   │   ├── models/                 MDM + ARM (one shared transformer)
│   │   ├── diffusion/              schedule, forward, loss
│   │   ├── inference/              vanilla, top_prob, top_prob_margin
│   │   ├── training/               train_mdm, train_arm
│   │   └── utils/                  seeding, checkpoint, logging, configs
│   ├── scripts/
│   │   ├── smoke_test.sh           <10 min, must pass before Phase 7
│   │   ├── run_lo_nae_sat.sh
│   │   └── run_sudoku.sh
│   └── tests/
│       ├── test_lo_nae_sat.py
│       ├── test_loss.py
│       ├── test_inference.py
│       └── test_forward_process.py
```

Phase 5 will then add a `entropy_filtered/` directory that imports from `baseline.src.*` and overrides only `training/train_mdm.py`.

---

**End of audit.** Ready for user review before Phase 4 begins.
