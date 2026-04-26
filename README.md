# Masked Diffusions: Entropy-Filtered Training

Research project building on **Kim, Shah, Kontonis, Kakade, Chen — *Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions* (ICML 2025, [arXiv:2502.06768](https://arxiv.org/abs/2502.06768))**.

## Background

Masked diffusion models (MDMs) train on exponentially many infilling subproblems — one for every possible mask pattern. Kim et al. (2025) prove that some of these subproblems are computationally intractable (Proposition 3.3), then show that the right *inference* strategy can sidestep them: a 6M-parameter MDM with a Top-Probability-Margin oracle reaches 89.49% accuracy on Sudoku, beating a 42M ARM trained with supervised teacher forcing of the correct order (87.18%).

## What this project investigates

The paper fixes inference. This project asks: can we also fix **training**?

The proposed extension — **entropy-filtered training** — scores each candidate mask by the model's mean prediction entropy at the masked positions, then drops:

- **Easy masks** (entropy below `H_low`): the model is already confident, gradient ≈ 0.
- **Hard masks** (entropy above `H_high`): likely intractable subproblems the paper proves are unlearnable.

Training proceeds only on masks in the productive band. This is a data-driven analog of the paper's analytic danger zone (`D_cond < D < D_KS`, Conjecture B.13) — the empirical entropy at masked positions stands in for the physics-derived thresholds.

## Hypotheses to test

1. Entropy-filtered training matches baseline accuracy at lower wall-clock.
2. Entropy-filtered training exceeds baseline at the same wall-clock.
3. Dropping only the hard tail (`top_filter`) is sufficient — confirming the paper's theory.
4. Dropping only the easy tail (`bottom_filter`) is sufficient — counter to the theory; would say easy subproblems are the wasted compute.
5. Adaptive inference becomes even more effective on entropy-filter-trained models.

## Project structure

```
Masked-Diffusions/
├── README.md                       this file
├── paper_notes.md                  source-of-truth notes on the paper
├── phase1_inventory.md             initial repo inventory (reference)
├── code_audit.md                   audit of prior reproduction attempts (Phase 3)
├── baseline/                       clean reproduction of Tables 1 & 2 (Phase 4)
│   ├── README.md
│   ├── configs/
│   ├── src/
│   ├── scripts/
│   └── tests/
├── entropy_filtered/               5-variant ablation of the modification (Phase 5)
│   ├── README.md
│   ├── configs/
│   ├── src/
│   └── tests/
├── results/                        aggregated results (Phase 8)
├── REPRODUCTION_GUIDE_*.md         historical planning docs (reference only)
└── Colab_*.ipynb                   historical Colab drafts (reference only, not run)
```

## Reproduction targets

| Experiment | Source | Target |
|---|---|---|
| L&O-NAE-SAT vanilla vs adaptive | Table 1 | 5 (N, P) configs, 75% naive baseline → 88–94% adaptive |
| Sudoku | Table 2 | 6M-MDM Top-Prob-Margin **89.49%** (vs 42M-ARM with-ordering 87.18%) |

Deferred for later: Tables 3 (Zebra), 4 (LLaDA-8B), 5 (hard Sudoku); Figures 2, 3, 4.

## Compute

Yale's [Bouchet HPC](https://docs.ycrc.yale.edu/clusters/bouchet/) cluster (Slurm, RTX 5000 Ada / RTX Pro 6000 / H200 partitions). Cluster credentials and Slurm account live in a `.gitignore`d `cluster_config.local.yaml`, never in source control.

## Status

Phases 1–6 complete. **52 unit tests pass**, end-to-end smoke runs in ~10 s on CPU, all 5 filter modes verified to fire correctly, checkpoint/resume verified bit-exact for preemption resilience. Cluster runs have not yet been scheduled.

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `phase1_inventory.md` | done |
| 2 | `paper_notes.md` (source of truth) | done |
| 3 | `code_audit.md` | done |
| 4 | `baseline/` + 32 unit tests (incl. checkpoint/resume) + L&O-NAE-SAT smoke (3.7 s) | done |
| 5 | `entropy_filtered/` + 20 filter tests + filtered smoke (2.6 s) | done |
| 6 | `methodology_notes.md`, `sudoku_scope_decision.md`, all 5 (N, P) × 5 variant configs, extended tests | done |
| 7 | Slurm scripts → user-approved sbatch | pending |
| 8 | `findings.md` | pending |

## References

- Kim, Shah, Kontonis, Kakade, Chen. *Train for the Worst, Plan for the Best.* ICML 2025. [arXiv:2502.06768](https://arxiv.org/abs/2502.06768).
- Ye, Gao, Gong, Zheng, Jiang, Li, Kong. *Beyond Autoregression.* [arXiv:2410.14157](https://arxiv.org/abs/2410.14157). Codebase used for puzzle experiments: [HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar).
- Nie et al. *Scaling up masked diffusion models on text.* [arXiv:2410.18514](https://arxiv.org/abs/2410.18514). Codebase: [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM).
- Nie et al. *Large Language Diffusion Models* (LLaDA). [arXiv:2502.09992](https://arxiv.org/abs/2502.09992). Codebase: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).
- Shah, Dikkala, Wang, Panigrahy. *Causal language modeling can elicit search and reasoning capabilities on logic puzzles.* [arXiv:2409.10502](https://arxiv.org/abs/2409.10502). Provides the Sudoku/Zebra datasets and the ARM-with-ordering baseline.
