# `baseline/` — clean reproduction package

Clean Python package for reproducing **Tables 1 and 2** of Kim et al. 2025 (arXiv:2502.06768) from scratch. Every design decision is traceable to a section of [`../paper_notes.md`](../paper_notes.md), the source of truth.

## Status (Phase 4)

| Task | Status |
|---|---|
| L&O-NAE-SAT generator | done — verified against `paper_notes.md` §5.4 |
| Forward process (masking) | done — `apply_mask` is fully vectorized |
| MDM training loss with 1/n weighting | done — verified by hand against Eq. 6 (Prop E.1) |
| Vanilla MDM inference (Bernoulli per-position) | done — fixes audit P1-1 |
| Top-Probability oracle | done |
| Top-Probability-Margin oracle | done |
| Bidirectional MDM transformer (RoPE / learned pos) | done — `causal=False` flag |
| Causal ARM transformer | done — `causal=True` flag (same code) |
| Seed plumbing (torch + numpy + random) | done |
| Checkpoint + resume (model + optimizer + RNG state) | done |
| JSONL metrics logging | done |
| Unit tests (23, all numerical) | passing |
| Smoke test (CPU, < 10 min) | passing in 3.7 s |
| Sudoku data loading + 7-strategy filter | done |
| Sudoku MDM training + evaluation | data path ready; training reuses `src/train.py` |
| **Sudoku ARM-with-ordering baseline** | **deferred** — see "Deferred work" below |

## Layout

```
baseline/
├── README.md                  this file
├── pyproject.toml             pinned deps, package metadata
├── configs/
│   ├── lo_nae_sat_smoke.yaml      tiny: 100 steps on CPU < 10 s
│   ├── lo_nae_sat_25_275.yaml     Table 1 row 1
│   ├── lo_nae_sat_30_270.yaml     Table 1 row 2
│   ├── lo_nae_sat_40_260.yaml     Table 1 row 3
│   ├── lo_nae_sat_50_250.yaml     Table 1 row 4
│   └── lo_nae_sat_100_200.yaml    Table 1 row 5
├── src/
│   ├── data.py                L&O-NAE-SAT generator (paper §3.1, §5.1)
│   ├── sudoku.py              Sudoku tokenization + 7-strategy filter
│   ├── model.py               shared transformer (causal flag, RoPE / learned pos)
│   ├── diffusion.py           schedule + forward + MDM loss (paper §2.1, §2.3, Eq. 6)
│   ├── inference.py           vanilla / top_prob / top_prob_margin (paper §2.6, §3.1, §3.2)
│   ├── evaluate.py            L&O-NAE-SAT obs-token accuracy
│   ├── train.py               training loop with seed/checkpoint/JSONL
│   └── utils.py               config, seeding, JSONL logger, checkpoint helpers
├── scripts/
│   ├── smoke_test.sh          gate before Phase 5
│   └── smoke_test.py          end-to-end smoke (called by smoke_test.sh)
└── tests/
    ├── conftest.py
    ├── test_lo_nae_sat.py     5 tests + worked-example exact assertion
    ├── test_loss.py           hand-computed 4-token toy, 1e-6 tolerance
    ├── test_forward_process.py 50 % mask rate at α=0.5 over 10 000 sequences
    ├── test_inference.py      no mask token in output, clues preserved
    └── test_sudoku.py         tokenization, 7-strategy solver, synthetic helper
```

## Quick start

```bash
# from the parent of this directory:
cd baseline
pip install -e .[test]                  # pin deps from pyproject.toml
bash scripts/smoke_test.sh              # gate: must finish < 10 min on CPU
```

## Reproducing Table 1 (L&O-NAE-SAT)

```bash
python -m src.train --config configs/lo_nae_sat_25_275.yaml
# (similarly for 30_270, 40_260, 50_250, 100_200)
```

Training writes checkpoints + `metrics.jsonl` under `runs/lo_nae_sat_<N>_<P>/`.

Each run is one of the five (N, P) configurations. Paper-reported targets (`paper_notes.md` §7.1):

| (N, P)     | Vanilla | Adaptive (Top-Prob-Margin) |
|------------|---------|----------------------------|
| (25, 275)  | 78.06 % | 93.76 %                    |
| (30, 270)  | 75.70 % | 93.54 %                    |
| (40, 260)  | 74.60 % | 92.21 %                    |
| (50, 250)  | 67.94 % | 90.01 %                    |
| (100, 200) | 62.84 % | 88.91 %                    |

Naive (all-1) baseline ≈ 0.71–0.75 depending on triple-distribution; see `paper_notes.md` §5.2 caveat.

## Reproducing Table 2 (Sudoku — MDM portion)

The `src/sudoku.py` module loads the Radcliffe (2020) Kaggle CSV and runs the 7-strategy filter to produce `train / test_easy / test_hard` splits. The filter takes ~30–60 minutes on the full 3M puzzles; for smaller experiments use the `max_puzzles` argument.

The MDM training path piggy-backs on `src/train.py` once a Sudoku data adapter is registered (planned next-step work). Phase 5 only needs the L&O-NAE-SAT path operational, so Sudoku is finalized in parallel with the entropy-filter rollout.

## Deferred work

These items are intentionally NOT in the current Phase 4 scope; they will be added before any Bouchet job runs.

1. **Sudoku ARM-with-ordering baseline.** Per `code_audit.md` (P0-2, P0-3) and `paper_notes.md` §12.9, this baseline must follow Shah et al. 2024's alternating-(position, value) token format. Per the user's Phase 4 spec, we will:
   - First clone `https://github.com/HKUNLP/diffusion-vs-ar` (Ye et al. 2024, the codebase the paper uses for puzzles) and check whether they ship this baseline.
   - If yes, vendor their adapter into `src/sudoku_arm_ordering.py`.
   - If no, vendor `https://github.com/kulinshah98/logic-puzzles` directly.
   - **Do not reimplement from the paper description alone.**
2. **Sudoku training/eval scripts.** Once the data path is wired and ordering baseline decided, add `configs/sudoku_mdm.yaml` and `configs/sudoku_arm_with_order.yaml`. The training loop is already general enough to handle Sudoku — only the data adapter is task-specific.
3. **Zebra puzzle (Table 3).** Out of Phase 4 scope per the new mission ("Defer for now").

## Conventions

- **Mask-token convention.** L&O-NAE-SAT uses `mask_token_id=3` to avoid colliding with observation value `0` (`paper_notes.md` §12.1). Sudoku uses `mask_token_id=0`; legitimate digits never equal 0 so there's no collision.
- **Loss weighting.** Per-sample 1/n, mean over batch (`paper_notes.md` §12.2).
- **Vanilla inference.** Per-position Bernoulli (paper §2.6, fixes audit P1-1).
- **Adaptive inference.** Top-K with `K = round(num_masked × (α_s − α_t)/(1 − α_t))`. Score noise: Gumbel coefficient 0.5 for puzzles, Gaussian σ=0.001 for text.
- **Schedule.** Linear `α_t = 1 − t`.
- **Time embedding.** None — model takes only token IDs (`paper_notes.md` §8 rule 2).
- **Bidirectional vs causal attention.** Single shared `Transformer` class with a `causal` flag; MDM uses `False`, ARM uses `True`.
