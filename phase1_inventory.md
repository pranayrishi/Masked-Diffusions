# Phase 1 Inventory

**Date**: 2026-04-26
**Working directory**: `/Users/rishinalem/Reproducing Token Ordering Paper/`
**Inventory rule observed**: nothing in this directory was modified during inventory.

---

## 1. Directory tree

The working directory is **flat** — there are no subdirectories, no `src/`, no `data/`, no `tests/`, no checkpoints, no logs.

```
Reproducing Token Ordering Paper/
├── Colab_1_Data_Prep.ipynb
├── Colab_2_LO_NAE_SAT.ipynb
├── Colab_3_Sudoku_Zebra.ipynb
├── Colab_4_Scaling_Laws.ipynb
├── Colab_5_Text_GenPPL.ipynb
├── Colab_6_LLaDA_8B.ipynb
├── Colab_7_BP_and_Plots.ipynb
├── REPRODUCTION_GUIDE_Phase1_Infrastructure_and_Training.md
├── REPRODUCTION_GUIDE_Phase2_Inference_and_Experiments.md
├── REPRODUCTION_GUIDE_Phase3_Integration_and_Troubleshooting.md
└── REPRODUCTION_GUIDE_REVISED_Complete.md
```

11 files total. Total disk usage ~470 KB. Mix of planning markdown (4 files) and Colab notebooks (7 files). No `.git/` *inside* this directory — the directory is tracked by a parent git repo at `/Users/rishinalem/` (see Section 5).

---

## 2. File-by-file summary

### 2.1 Planning / specification documents (markdown)

| File | Lines | Status | Summary |
|---|---|---|---|
| `REPRODUCTION_GUIDE_REVISED_Complete.md` | 1,513 | reference | Master plan. Designed for Colab + Google Drive workflow. Covers all 7 notebooks, with the L&O-NAE-SAT data generator, model architectures, MDM training loop, all three inference strategies, and Drive paths. **This is the spec the existing notebooks were written against.** |
| `REPRODUCTION_GUIDE_Phase1_Infrastructure_and_Training.md` | 668 | reference | Detail-level Phase 1: env setup, SlimPajama tokenization (GPT-2, L=2048), L&O-NAE-SAT verification (m=2 derivation), Sudoku via Shah et al. 2024 / Radcliffe 2020, π-learner training (uniform/closer/much-closer), AdamW β₁=0.9 β₂=0.95, cosine LR 4e-4→4e-5, IsoFLOP analysis. |
| `REPRODUCTION_GUIDE_Phase2_Inference_and_Experiments.md` | 979 | reference | Detail-level Phase 2: full code for vanilla / Top-Probability / Top-Probability-Margin inference, conditional inference for puzzles (Gumbel coeff 0.5), Gaussian noise for text, generative perplexity via LLaMA-2 7B, BP for planted CSP per Definition B.10. |
| `REPRODUCTION_GUIDE_Phase3_Integration_and_Troubleshooting.md` | 655 | reference | Detail-level Phase 3: linear schedule α_t = 1−t, Figure 1 schematic, master bash pipeline, gotchas (mask-token convention, causal vs bidirectional, time-embedding-free, ARM-with-ordering, semi-autoregressive LLaDA), troubleshooting tree. |

These four documents are coherent with each other and with the paper. They were written **before** the notebooks and form the specification the notebooks were intended to implement.

### 2.2 Colab notebooks (code)

All seven were written in this same session. **None has been run end-to-end successfully.** Two known-failed cells are documented in Section 4. Total code: ~4,886 non-blank lines, 171 cells across the 7 notebooks.

| Notebook | Cells (md / code) | Code lines | Reproduces | Run status |
|---|---|---|---|---|
| `Colab_1_Data_Prep.ipynb` | 21 (10/11) | 658 | Env setup, clone 4 codebases, SlimPajama tokenization, L&O-NAE-SAT data, Sudoku 7-strategy filter, Zebra/LLaDA prep | **Partially run.** SlimPajama load failed (auth-gated dataset, fixed in subsequent edit to fall back to `DKYoon/SlimPajama-6B`). Sudoku CSV download cell failed (no Kaggle creds; fixed in a subsequent edit to use `KAGGLE_API_TOKEN` from Colab Secrets). |
| `Colab_2_LO_NAE_SAT.ipynb` | 16 (7/9) | 458 | Figure 2 bottom-right (error imbalance) and Table 1 (vanilla vs adaptive). 19M MDM with RoPE, max seq 512. | Not run. |
| `Colab_3_Sudoku_Zebra.ipynb` | 22 (9/13) | 491 | Tables 2, 3, 5. Sudoku 6M MDM, Zebra 19M MDM, ARM 42M baselines (with and without solve-order teacher forcing). | Not run. Zebra path is a stub — relies on Shah et al. data format that has not been pulled. |
| `Colab_4_Scaling_Laws.ipynb` | 36 (17/19) | 975 | Figure 2 left (IsoFLOP scaling) + Figure 2 top-right (val loss table). ARM, MDM, and 9 π-learners (3 each from uniform / closer / much-closer). Learnable positional embeddings (NOT RoPE). | Not run. Most compute-intensive notebook (~200-500 A100-hours at full scale). |
| `Colab_5_Text_GenPPL.ipynb` | 25 (13/12) | 642 | Figure 3. 1.1B MDM text generation; LLaMA-2 7B as evaluator; GenPPL + unigram entropy across 250–2000 sampling steps. Gaussian (not Gumbel) oracle noise. | Not run. Depends on a pretrained 1.1B MDM (SMDM checkpoint required). |
| `Colab_6_LLaDA_8B.ipynb` | 29 (11/18) | 938 | Table 4. LLaDA 8B base (~16 GB weights) on HumanEval-Infill (Single/Multi/Split), Math, MMLU, ROCStories. Vanilla vs Top-Prob vs Top-Prob-Margin. Semi-autoregressive sampling for instruction tasks. | Not run. Requires A100 80 GB. |
| `Colab_7_BP_and_Plots.ipynb` | 22 (12/10) | 724 | Figure 4 (belief propagation on planted CSP, k=3, m=3, NAE, N=10000) + final figure compilation across all results. CPU-only. | Not run. |

### 2.3 Configuration files

**None present.**

- No `requirements.txt` or `environment.yml`.
- No `pyproject.toml` or `setup.py`.
- No `.yaml` / `.yml` config files.
- Dependencies are inlined via `!pip install ...` cells inside each notebook (typically `torch==2.1.2`, `transformers==4.36.2`, `datasets==2.16.1`, `accelerate==0.25.0`, `einops`, `wandb`, `flash-attn`).

This is a known gap for Phase 4 — the rewrite needs a proper `pyproject.toml` or `requirements.txt` with pinned versions for Bouchet.

### 2.4 Data files

**None present locally.** All data references in the notebooks use `BASE_DIR='/content/drive/MyDrive/mdm_reproduction'`, which is a Colab-specific Google Drive path. There is no local SlimPajama, no Sudoku CSV, no L&O-NAE-SAT generated data, no LLaDA weights, no checkpoints.

### 2.5 Other code or notebook files

None. No scripts, no `Makefile`, no Dockerfile.

---

## 3. Paper artifacts

| Artifact | Location | Notes |
|---|---|---|
| Paper PDF | `/Users/rishinalem/Downloads/2502.06768v3.pdf` (1.14 MB) | **Found.** This is the v3 of arXiv:2502.06768 (Kim, Shah, Kontonis, Kakade, Chen — ICML 2025). It will be the source of truth for Phase 2. |
| Extracted notes | None | No `paper_notes.md` exists yet; Phase 2 will produce one. |
| Citations / bibliography | None | No BibTeX file. |

---

## 4. Prior run history

### 4.1 Notebook execution status

- **Colab 1 was partially executed.** Two cells failed during the user's first attempt:
  1. **SlimPajama load_dataset call** failed with `DatasetNotFoundError: Dataset 'cerebras/SlimPajama-627B' doesn't exist on the Hub`. The dataset is currently gated; a try/except fallback to `DKYoon/SlimPajama-6B` was added to the notebook.
  2. **Sudoku CSV** was missing because the cell expected the Kaggle CSV pre-staged. A Kaggle-API-based auto-download with `KAGGLE_API_TOKEN` (from Colab Secrets) was added.
- **Colabs 2–7 have never been run.** They were authored but no execution output is saved in the notebooks (all cells have empty `outputs` arrays).

### 4.2 Checkpoints, logs, wandb runs

**None found.**

- No `checkpoints/`, `runs/`, `wandb/`, `logs/`, `outputs/` directories.
- No `.pt` / `.pth` / `.bin` / `.safetensors` files.
- No `metrics.csv`, `metrics.jsonl`, or any structured run output.

### 4.3 Implications for the new mission

- The existing notebooks form a substantial first draft (~4,900 lines targeted at the right experiments) but have **never produced a single reproduction result**. They cannot be trusted as a baseline.
- The Colab-specific path conventions (`/content/drive/MyDrive/...`, `from google.colab import drive`, `!pip install ...` cells) make them unsuitable for direct Bouchet use. A port to a normal Python package layout will be necessary.
- The four planning markdowns are high-quality reference material and should be kept as-is.

---

## 5. Hidden / contextual state (read-only)

- `~/Reproducing Token Ordering Paper/` is **inside** a larger git repository rooted one level up at `/Users/rishinalem/`. The current branch is `fresh-main`, and recent commits relate to a different research project (`Reproducing - A Mechanistic Analysis of Transformers for Dynamical Systems`) and a 3-D game prototype, not this MDM project. Recent commit subjects:
  - `1e6f84d v6.2: Metric overhaul — rescore v6.1 with embedding similarity and LLM-as-judge`
  - `d9b6c26 Add v6.1 rollout and confusion figures to README`
  - `3d19c6b Add v6.1 evaluation figures for professor review`
- `git status` from inside this directory reports modifications and deletions in **sibling directories** (the dynamical-systems project and the game prototype). Nothing inside `Reproducing Token Ordering Paper/` is currently modified or staged.
- **Recommendation:** confirm with the user whether this MDM project should remain in the same parent repo or move to its own git repo before Phase 4 begins. Sharing a repo with unrelated active projects creates noise and risks accidental cross-project commits.

---

## 6. External dependencies the project will need

Based on what the existing notebooks reference and what the new mission requires:

**Python packages:** `torch`, `transformers`, `datasets`, `accelerate`, `einops`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`, `wandb`, `pandas`, `huggingface_hub`, `kaggle`. Optional: `flash-attn`.

**Datasets:**
- SlimPajama (`cerebras/SlimPajama-627B` is gated; `DKYoon/SlimPajama-6B` is the fallback; both via `huggingface-cli login`).
- Sudoku 3M from Kaggle (`radcliffe/3-million-sudoku-puzzles-with-ratings`).
- Shah et al. 2024 puzzle datasets (need to clone `https://github.com/kulinshah98/logic-puzzles` — referenced but unknown if data ships with the repo).
- LLaDA 8B weights from HuggingFace (`GSAI-ML/LLaDA-8B-Base`, ~16 GB).

**Reference codebases (cloned in Colab 1, not present locally):**
- `https://github.com/ML-GSAI/SMDM` — Nie et al. 2024 scaling-law repo.
- `https://github.com/HKUNLP/discrete-diffusion` — Ye et al. 2024 puzzle repo. **Note:** the new mission says the canonical Ye et al. 2024 repo is `https://github.com/HKUNLP/diffusion-of-thoughts`. Need to confirm which is correct.
- `https://github.com/ML-GSAI/LLaDA` — LLaDA 8B inference repo.
- `https://github.com/kulinshah98/logic-puzzles` — Shah et al. 2024 puzzle data and ARM-with-ordering baseline.

---

## 7. Open questions for the user

These need answers before proceeding past Phase 3.

### About the project / paper
1. **Git layout.** Should `Reproducing Token Ordering Paper/` become its own git repository (`git init` here, separate remote), or stay tracked by the parent `~/.git`? The latter mixes it with unrelated active projects.
2. **Which Ye et al. 2024 repo is canonical?** The new mission cites `https://github.com/HKUNLP/diffusion-of-thoughts`. The existing Colab 1 cites `https://github.com/HKUNLP/discrete-diffusion`. Phase 3 audit will need to reference the right one.
3. **Reuse vs. rewrite of the Colab notebooks.** They were never run, so trusting them is risky. My recommendation (formalized in Phase 3) will likely be to extract correct pieces but build a clean Python package for Bouchet — not run the notebooks themselves.

### About the cluster
4. **NetID.** Required for `ssh <netid>@bouchet.ycrc.yale.edu`.
5. **Slurm account.** Likely `pi_<labname>`. Can the user check `sacctmgr show user $USER --associations` after first SSH?
6. **Lab name / partition.** Does the Yale professor's lab have a private partition (`pi_<labname>`)? Or should we use the public partitions (`gpu`, `gpu_h200`, `devel`, `day`, `week`, `scavenge_gpu`)?
7. **Project storage path.** Standard form is `/home/<netid>/project_pi_<labname>`. Need confirmation.
8. **Compute budget.** Has the PI authorized a specific number of GPU-hours? This drives which experiments are realistic. The full reproduction of Tables 1+2+5 alone is ~50–100 GPU-hours; adding the entropy ablation (5 variants × 5 (N,P) × 3 seeds = 75 jobs) roughly triples that.
9. **GPU type preference.** H200 is best for ML throughput on Bouchet. Default? Or use RTX 5000 Ada / RTX Pro 6000 for cost?
10. **Yale VPN setup.** Is the user already on Yale VPN / on-campus network? Required for SSH.
11. **Duo MFA.** Is the user familiar with the Duo prompt for SSH? SSH multiplexing in `~/.ssh/config` will be set up in Phase 7 to keep this from being painful.

### About the modification (preview for Phase 5)
12. **Threshold philosophy.** The professor's spec leaves entropy thresholds free. Do they prefer (a) absolute thresholds (`H_low`, `H_high` in nats), or (b) batch-relative percentiles (e.g., keep 25th–75th)? My recommendation in Phase 5 will be to implement both and ablate.
13. **Warmup.** During the first ~500 steps the model is random and entropies are not informative. Is unfiltered warmup acceptable, or should the very first step use a different proxy (e.g., random filtering)?

### Logistics
14. **wandb.** Does the user / professor have a wandb account they want runs logged to? Or stay with offline mode + local JSONL?
15. **Email notifications.** What email should `--mail-user=` use on Slurm scripts (typically `<netid>@yale.edu`)?

---

## 8. Recommendation

The existing notebooks are useful as a **specification checklist** — they enumerate the right experiments and they encode (mostly) the right hyperparameters. They are **not** trustworthy as a runnable baseline because (a) none has been executed end-to-end, (b) two of the foundational data-prep cells are known to fail, (c) Colab-specific paths are pervasive, and (d) checkpointing / resume / determinism hooks are sparse.

For Phase 4, I recommend:
- Treat `paper_notes.md` (Phase 2 output) as the **source of truth**.
- Use the four `REPRODUCTION_GUIDE_*.md` files plus the notebooks as **reference material** to cross-check decisions.
- Build a clean Python package (`baseline/src/...`) with proper configs, tests, and Bouchet-friendly paths from scratch.
- Mine the notebooks for specific algorithm pieces (e.g., the 7-strategy Sudoku solver in Colab 1, the BP message-passing in Colab 7, the L&O-NAE-SAT generator) but re-implement training loops, inference oracles, and the loss function clean.

I will commit to this recommendation in `code_audit.md` (Phase 3) once I have read the paper end-to-end (Phase 2).

---

**Phase 1 status:** complete. Awaiting user confirmation to proceed to Phase 2 (paper deep-read).
