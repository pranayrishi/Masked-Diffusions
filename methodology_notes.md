# Methodology Notes — open questions to resolve before paper submission

This file collects **methodological ambiguities** in Kim et al. 2025 (arXiv:2502.06768) and adjacent work that affect the project's claims. Each entry has:

1. **Question** — the precise ambiguity.
2. **Why it matters** — what claim depends on the resolution.
3. **Status** — open / resolved / blocked-on.
4. **Resolution path** — concrete steps to get an answer.

---

## Q1. With-replacement vs. without-replacement triple sampling for L&O-NAE-SAT

**Question.** When constructing the `P` random observation triples for an L&O-NAE-SAT distribution (paper §3.3, Definition 3.1), does each triple `(i_1, i_2, i_3) ∈ [N]³` get its three indices sampled **independently with replacement** (allowing degenerate triples like `(2, 2, 2)`) or **without replacement** (each triple has three distinct indices)?

**Why it matters.**

- The paper's Table 1 caption states "naive guessing leads to 75% accuracy."
- Mathematical fact: 75% is exact only when *every* triple has three distinct indices (the `1 − 1/m²` regime for `m=2`).
- With with-replacement triples and `N` finite, some triples are degenerate. For `(N, P) = (20, 280)` we computed (and verified empirically in `baseline/tests/test_lo_nae_sat.py`) the analytical population P(NAE = 1) ≈ **0.7125**, not 0.75.
- The discrepancy decreases as `N` grows. For `(N, P) = (100, 200)` the asymptotic 75% is essentially reached.
- **Implication for our entropy filter.** The "intractability danger zone" predicted by Prop 3.3 sits at a specific masking-fraction interval that depends on the exact triple distribution. If the paper's triples are without-replacement (planted-CSP-style) but our generator does with-replacement (or vice versa), the danger zone moves. Our `top_filter` ablation may target a slightly different α-interval than the paper's theoretical prediction.
- **Implication for reporting.** Our reproduction of Table 1 is implicitly a stronger claim if we match the paper's triple distribution. If we mismatch, we should either match the paper exactly or document the difference.

**Status.** **Open.** Both interpretations are textually consistent with the paper's prose: "for some randomly chosen (pre-fixed) triples (i₁, i₂, i₃) ∈ [N]" (Section 3.3) does not specify the joint distribution.

**Resolution path** (in order of expected effort):

1. **Inspect the official codebases.**
   - `https://github.com/HKUNLP/diffusion-vs-ar` (Ye et al. 2024, the codebase the paper uses for puzzles) — does it ship an L&O-NAE-SAT generator? If yes, read the triple-sampling line.
   - `https://github.com/ML-GSAI/SMDM` (Nie et al. 2024) — similar check.
   - **Note:** the L&O-NAE-SAT distribution is *new* in Kim et al. 2025; it may not be in the prior codebases. In that case the authors likely ship their own data-generation code. If so, look for a Kim et al. release.
2. **Inspect the planted-CSP literature linked in §B.4.** The 1-RSB cavity prediction (Conjecture B.13) is stated for the planted-random-CSP model, which conventionally uses the *random k-uniform hypergraph* construction — i.e., choose each ordered k-tuple `S` of distinct elements with probability `φ / N^{k−1}`. This is **without-replacement** at the per-tuple level. If the paper inherits this convention, with-replacement would be inconsistent with their theoretical analysis.
3. **Email the authors.** If steps 1–2 are inconclusive, email Kulin Shah (`kulin-shah@utexas.edu`, listed as correspondence on the title page).

**Working assumption** (until resolved): our `baseline/src/data.py` uses **with-replacement** triple sampling, matching the existing notebook convention. This is documented in `paper_notes.md` §5.2 with a caveat. The smoke test asserts both regimes (asymptotic 0.75 with explicitly-distinct triples, finite-N analytical 0.7125 with the seed-42 generator).

**If we discover the paper uses without-replacement:** we will (a) flip the default in `baseline/src/data.py` to without-replacement and (b) re-run all configs. The training-data generator is cheap, so this is a 1-day fix.

---

## Q2. Gaussian noise σ for the text inference oracle

**Question.** Appendix D.1.2 specifies that the text-data adaptive oracle adds Gaussian noise `ε` to the top-probability margin: `F = TopK(|p(j₁) − p(j₂)| + ε)`. **What value of σ²?**

**Why it matters.**

- The paper does not pin σ. Larger σ moves the oracle toward random; smaller σ moves it toward greedy.
- The notebook draft used σ = 0.001. There is no published justification for this number.
- Figure 3 (text generative perplexity) is sensitive to σ — too small and adaptive collapses into greedy and entropy drops; too large and adaptive ≈ vanilla.

**Status.** **Open.** Affects only the deferred Figure 3 work, not the headline Table 1 / Table 2 reproduction.

**Resolution path.** When we get to the deferred text experiments, sweep σ ∈ {0, 0.001, 0.01, 0.1} and report the σ at which our results match the paper's reported curves. Document the chosen σ in the camera-ready paper.

---

## Q3. ARM-with-ordering training format for Sudoku

**Question.** Paper Appendix D.2 / Table 2 reports "ARM (with ordering)" achieving 87.18 % on Sudoku. The baseline traces back to Shah et al. 2024. **What is the exact training format?** Specifically:

- Does the model emit `(position_token_0, value_token_0, position_token_1, value_token_1, ...)` (alternating), or
- Does it emit `(value_token_0, value_token_1, ...)` with the position information conveyed implicitly by some other mechanism?

The paper's prose does not specify; the audit (`code_audit.md` P0-3) flagged this as a blocker.

**Why it matters.** This is the **headline comparison**: 6 M MDM with adaptive inference (89.49 %) > 42 M ARM with ordering (87.18 %). Reproducing this number correctly requires getting the ARM-with-ordering format right.

**Status.** **Blocked on Phase 6 / Sudoku scope decision** (`sudoku_scope_decision.md`). Cloning both `HKUNLP/diffusion-vs-ar` and `kulinshah98/logic-puzzles` to determine which (if any) ships a runnable version. Per user instruction, **do not reimplement from the paper description alone.**

**Resolution path.** Phase 6.4 of the project plan.

---

## Q4. Sudoku 7-strategy filter exactness

**Question.** Shah et al. 2024 filter the Radcliffe (2020) 3M Kaggle puzzles by 7 fixed strategies (no backtracking) to define the easy/hard split (paper Appendix D.2). **Which exact 7?** And what are their tie-breaking rules?

**Why it matters.**

- Different "naked triples" implementations or different orderings of strategies can produce different easy/hard splits.
- Our reproduction produces an "easy" subset whose size should match Shah et al.'s reported count. If it doesn't, we are training on slightly different data.
- For Table 5 (hard Sudoku generalization), the test-set composition is the *complement* — any deviation in the filter changes both train and test sets.

**Status.** **Open.** Our `baseline/src/sudoku.py` implements the 7 strategies the prior notebook authors enumerated (Naked Singles, Hidden Singles, Naked Pairs, Hidden Pairs, Pointing Pairs, Box/Line Reduction, Naked Triples). This is the standard textbook list, but Shah et al.'s exact choice is not documented in their paper either.

**Resolution path.**

1. When cloning `kulinshah98/logic-puzzles` for Q3, look for their filter implementation and verify our list.
2. If Shah et al. ship a pre-filtered dataset, use theirs directly and skip re-filtering.
3. If implementations differ, count puzzles in each "easy" subset and report the difference; this is a legitimate `findings.md` entry rather than a blocker.

---

## Q5. Number of evaluation samples for Table 1

**Question.** Paper Appendix D.1.1 states "we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference using top probability margin." The number of test samples used for the accuracy estimate is not stated.

**Why it matters.** Modest — affects the size of confidence intervals on our reported numbers. With 100 samples the standard error is ~5%; with 10,000 samples it's ~0.5%.

**Status.** **Open.** Our default is 500 samples per (N, P, strategy). This is enough for clean rankings but not for tight confidence intervals.

**Resolution path.** Increase to 5,000 samples for the camera-ready paper; smoke and dev runs stay at 500.

---

## Q6. Per-step LR schedule details for the L&O-NAE-SAT experiment

**Question.** Paper Appendix C.1 gives global hyperparameters: AdamW (β₁=0.9, β₂=0.95, wd=0.1), cosine schedule, max LR = 4e-4, min LR = 4e-5. Appendix D.2 gives puzzle-specific: lr = 0.001, batch = 128. **For L&O-NAE-SAT (Appendix C.2.1, D.1.1) — which lr? 4e-4 from the global default, or 0.001 from the puzzle setting?**

**Why it matters.** Modest — affects convergence speed but not (likely) final accuracy at the 5×10⁴-iteration scale of the paper's proxy MDM. Our `baseline/configs/lo_nae_sat_*.yaml` uses lr = 1e-3 (matching the puzzle setting); this is a defensible interpretation but undocumented.

**Status.** **Open.** Soft.

**Resolution path.** If our reproduction misses paper numbers by > 2% on Table 1, sweep lr ∈ {4e-4, 1e-3} and report.

---

## Q7. Batch composition for the proxy Bayes-optimal MDM

**Question.** Paper Appendix C.2.1 trains a "proxy MDM" for `5 × 10⁴ iterations` to approximate the Bayes-optimal predictor for the error-imbalance computation (Figure 2 bottom-right). **Is this the same model size (19M) as the model under study, or larger?**

**Why it matters.** A proxy that is too small to converge to the Bayes-optimal posterior gives biased error estimates. The paper's prose says "proxy MDM for the Bayes optimal predictor" without specifying size. Implicit assumption: same size, longer training.

**Status.** **Open.** Affects only the deferred Figure 2 reproduction, not the headline Table 1 / Table 2.

**Resolution path.** If we go after Figure 2, train one 19M proxy at 5×10⁴ iterations and compare its error against a larger proxy (e.g., 42M @ 5×10⁴). If they agree, the choice doesn't matter; if they don't, document the question.

---

## Adding to this file

When a new methodological ambiguity is discovered, add an entry with the same five fields. Linkable from the paper draft and from the project README so the professor can audit the open questions before submission.
