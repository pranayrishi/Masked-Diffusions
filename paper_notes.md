# Paper Notes — "Train for the Worst, Plan for the Best"

> **Citation.** Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, Sitan Chen.
> *Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions.*
> ICML 2025. arXiv:2502.06768v3 (2025-08-19).
> Local PDF: `/Users/rishinalem/Downloads/2502.06768v3.pdf` (21 pages).

> **Status.** This is the **source of truth** for all coding decisions. When the existing notebooks, the four `REPRODUCTION_GUIDE_*.md` files, or my own memory disagree with this document, **this document wins** — and if this document disagrees with the paper, fix this document.

> **Read.** Pages 1–21 of arXiv v3 read in full on 2026-04-26.

---

## 0. Mental model in one paragraph

Masked diffusion models (MDMs) are trained on **all** L! infilling orderings simultaneously by uniformly masking tokens and predicting the rest. Autoregressive models (ARMs) train on a *single* ordering (left-to-right). MDMs therefore solve exponentially more subproblems during training, and some of those subproblems are provably computationally intractable (Proposition 3.3). However, MDM **inference** can pick the ordering on the fly: at each reverse step, decide which token to unmask using the model's own confidence. With a good ordering oracle (Top-Probability or Top-Probability-Margin), an MDM's logits already contain enough information to **avoid** the hard subproblems, so a tiny MDM with adaptive inference can beat a large ARM trained with supervised teacher forcing of the correct order. Headline numbers: 6M-MDM with Top-Probability-Margin = 89.49% on Sudoku, beating 42M-ARM with-ordering at 87.18%.

The professor's proposed modification — **entropy-filtered training** — is the inverse: train only on subproblems whose mean masked-position entropy lies in a productive band, dropping easy (no-gradient) and hard (intractable) tails. This is a data-driven analog of the paper's theoretical danger zone (D_cond < D < D_KS, Conjecture B.13).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `L` | sequence length |
| `m` | alphabet size of data tokens. Section 2 says data is on `{1,...,m}^L`. Section 3.1 (Def 3.1) extends this to `{0,...,m}^L` for L&O distributions because observations may take value 0. |
| `0` | mask token. **Collides with observation value 0 in L&O-NAE-SAT** — see §12.1. |
| `t ∈ [0, 1]` | continuous noise level. `t = 0` ≈ clean, `t = 1` ≈ fully masked. |
| `α_t` | noise schedule. `α_0 ≈ 1`, `α_1 ≈ 0`. The paper uses linear `α_t = 1 − t` throughout (implicit). |
| `α'_t` | derivative `dα_t/dt`. For linear schedule, `α'_t = −1`. |
| `x_0` | clean (data) sequence, sampled from `p_data`. |
| `x_t` | sequence at noise level `t`: each token independently masked with probability `1 − α_t`. |
| `x_0^i` | the `i`-th token of `x_0`. |
| `x_t^i` | the `i`-th token of `x_t`. Either equals `x_0^i` or equals 0 (masked). |
| `M ⊆ [L]` | a "mask set": positions that are masked. `|M|` = number of masked positions. |
| `x_0[M]` | the **complement** in the paper's notation — the sequence with `M` positions replaced by 0. So `p_θ(x_0^i \| x_0[M])` is "predict token at position `i` given the partially masked sequence." |
| `p_θ` | denoising network. Takes only `x_t` as input (not `t`). Outputs categorical distribution over `{0,1,...,m}` per position. |
| `g_θ` | reverse-process kernel built from `p_θ`. |
| `S` | set of positions to unmask at a reverse-process step. |
| `K` | size of `S`. Chosen so expected unmask count matches vanilla inference. |
| `π` | a permutation of `[L]`. `Unif(S_L)` = uniform over all `L!` permutations. `id` = identity. |
| `π-learner` | a causal autoregressive model trained on `π(x_0)` instead of `x_0`. |
| `Closer(S_L)` | distribution: identity + `L/10` random transpositions. |
| `Much-Closer(S_L)` | distribution: identity + `√L` random transpositions. (Bormashenko 2011 says `L log L` swaps gives ≈ uniform.) |
| `D_KS, D_cond` | Kesten-Stigum and condensation thresholds for a planted CSP — see §4. |
| `γ` | probability that the predicate `g` is satisfied by a random assignment. For NAE on `{1,...,m}^k`, `γ = 1 − 1/m^{k−1}`. |

---

## 2. The MDM framework (Section 2)

### 2.1 Forward process

For each `i = 0, ..., L−1`, **independently**:

```
q_{t|0}(x_t^i | x_0^i) = Cat( α_t · e_{x_0^i} + (1 − α_t) · e_0 )
```

In words: at noise level `t`, each token independently equals its original value with probability `α_t` and equals the mask token 0 with probability `1 − α_t`.

The full forward kernel is the product over coordinates: `q_{t|0}(x_t | x_0) = ∏_i q_{t|0}(x_t^i | x_0^i)`.

### 2.2 Reverse process

For `s < t`, the reverse kernel `q_{s|t}(x_s | x_t, x_0)` factors per coordinate:

```
q_{s|t}(x_s^i | x_t, x_0) =
    Cat(e_{x_t^i})                                       if x_t^i ≠ 0    (already unmasked, stays put)

    Cat( ((1 − α_s)/(1 − α_t)) · e_0 + ((α_s − α_t)/(1 − α_t)) · e_{x_0^i} )
                                                          if x_t^i = 0   (masked: stay masked w/p (1−α_s)/(1−α_t),
                                                                          unmask to true value w/p (α_s−α_t)/(1−α_t))
```

This is the **exact** posterior of the forward process. The reverse-process *model* approximates the unknown `x_0^i` term using `g_θ(x_s^i | x_t) ≜ q_{s|t}(x_s^i | x_t, x_0 ← p_θ(·|x_t))`.

### 2.3 Training loss (continuous-time integral form)

```
L_θ = ∫_0^1  α'_t / (1 − α_t)  ·  E_{x_0 ~ p_data, x_t ~ q_{t|0}(·|x_0)}  [ Σ_{i: x_t^i = 0}  − log p_θ(x_0^i | x_t, t) ]  dt
```

In practice the network is **time-embedding-free**: `p_θ(· | x_t, t) = p_θ(· | x_t)`. The model infers `t` implicitly from how many positions are masked.

### 2.4 Equivalent loss as average over mask sets — Eq. (1)

Proposition 2.1 (whose proof reduces to Prop 3.1 of Zheng et al. 2024, reproduced in Appendix E) gives:

> Assume `α_0 = 1`, `α_1 = 0`, and that `p_θ` is time-embedding-free. Then `L_θ ≤ −E_{x_0 ~ p_data}[log p_θ(x_0)]` and
>
> ```
> L_θ = − Σ_{M ⊆ [L], i ∈ M}  (1/|M|) · (1 / C(L,|M|))  ·  E_{x_0 ~ p_data} [ log p_θ(x_0^i | x_0[M]) ]   ... (1)
> ```

The MDM loss is a **uniform average over all `(2^L − 1) · L` (mask-set, masked-position) pairs** (excluding `M = ∅`). This is what "MDMs train on exponentially more subproblems than ARMs" means.

### 2.5 Equivalence to any-order autoregressive loss — Equation (3)

```
L_θ = − E_{x_0 ~ p_data, π ~ Unif(S_L)}  [ Σ_{i=0}^{L−1}  log p_θ( x_0^{π(i)} | x_0[π{i, ..., L−1}] ) ]
```

i.e., **MDM loss = uniform average over permutations of the AR loss for that permutation**. Setting `π = id` and putting all expectation mass on `id` recovers standard left-to-right AR. Sampling `π ~ Closer / Much-Closer / Unif` interpolates between the two — the basis of the Section 3.2 scaling-law experiment (Figure 2 left).

### 2.6 Vanilla MDM inference algorithm — Section 2.1.2

> **Vanilla MDM inference**
> Initialize `x_1 = (0, 0, ..., 0)` (fully masked).
> For each step from `t` to `s` (`t > s`):
>
> 1. Sample a set `S ⊆ {i : x_t^i = 0}` of currently-masked positions to unmask. Each masked position is included in `S` independently with probability `(α_s − α_t) / (1 − α_t)`.
> 2. For each `i ∈ S`, sample `x_s^i ~ p_θ(x^i | x_t)`.

Tokens already unmasked are *never* changed. The choice of which positions to unmask is **completely random** in vanilla — that's the lever the adaptive variants change.

---

## 3. Adaptive MDM inference (Section 4)

> **Adaptive MDM inference**
> Initialize `x_1 = (0, ..., 0)`.
> For each step from `t` to `s`:
>
> 1. `S = F(θ, x_t) ⊆ {i : x_t^i = 0}` — chosen by an oracle.
> 2. For each `i ∈ S`, sample `x_s^i ~ p_θ(x^i | x_t)`.

### 3.1 Top-Probability oracle (Zheng et al. 2023, also called "Top-K" in some priors)

The certainty at position `i` is

```
c_i = max_{j ∈ {1,...,m}}  p_θ(x^i = j | x_t)
```

and `F(θ, x_t) = TopK_i(c_i)`.

### 3.2 Top-Probability-Margin oracle (the paper's main proposal)

If `j_1, j_2` are the two most-probable values at position `i`,

```
c_i = | p_θ(x^i = j_1 | x_t) − p_θ(x^i = j_2 | x_t) |
```

and `F(θ, x_t) = TopK_i(c_i)`.

**Why margin beats max.** If the model places `(0.45, 0.44, 0.11)` at a position, top-prob says "high certainty (0.45)" but margin says `|0.45 − 0.44| = 0.01` — correctly low certainty. This matters most in puzzles where multiple values can be plausible (near-tie).

### 3.3 Number of positions to unmask per step (`K`) — Appendix D.1.2

To match vanilla in expectation:

```
K = (# masked tokens in current x_t) · (α_s − α_t) / (1 − α_t)
```

For linear schedule `α_t = 1 − t`, this is `K = (# masked) · (t − s) / t`.

The paper notes both deterministic K (above) and stochastic `K ~ Binom(# masked, (α_s−α_t)/(1−α_t))` give comparable generative perplexity.

### 3.4 Noise injection on the oracle scores

**Puzzles (Sudoku, Zebra) — Appendix D.2:**
> "We add Gumbel noise with a coefficient of 0.5 to the MDM inference oracle F."

Concretely: `F = TopK( c_i + 0.5 · Gumbel(0,1) )`. The noise is added to **scores, not to logits**.

**Text data — Appendix D.1.2:**
> "Adding a certain level of temperature to the oracle is useful. … Therefore, we consider a variant … `F(θ, x_t) = TopK( |p_θ(x^i=j_1|x_t) − p_θ(x^i=j_2|x_t)| + ε )`"

`ε` is Gaussian noise with a tunable std (paper does not pin the exact std; the existing notebook uses `σ = 0.001`).

**LLaDA-8B (Section 4.4 / Appendix D.3)** — semi-autoregressive sampling for instruction-answering tasks (Math, MMLU); fully non-autoregressive for infilling (HumanEval-Infill, ROCStories). Length must be specified for instruction-answering. Follows the LLaDA paper (Nie et al. 2025) sampling configuration.

---

## 4. The hardness theory (Section 3, Appendices B.3, B.4)

### 4.1 L&O distribution (Definition 3.1)

A **latents-and-observations** distribution `p_data` over sequences of length `L = N + P` with alphabet `{0, ..., m}` is parameterized by:

- A permutation `π` over `[L]` (specifies generation order — for L&O-NAE-SAT, `π = id`).
- Number of latent tokens `N`, number of observation tokens `P`.
- A prior `p_prior` over `{1, ..., m}` for the latents.
- For each `j = 1, ..., P`, an observation function `O_j : {1, ..., m}^N → {0, ..., m}`, **efficiently learnable** in the PAC sense.

Sampling:
1. `x^{π(i)} ~ p_prior` for `i = 1, ..., N` (latents).
2. `x^{π(N+j)} = O_j( x^{π(1)}, ..., x^{π(N)} )` for `j = 1, ..., P` (observations are *deterministic functions* of the latents).

### 4.2 Example 3.2 — sparse predicate observations

Fix arity `k ≥ 2` and a predicate `g : {1, ..., m}^k → {0, 1}`. Take `P = N · (N−1) · ... · (N−k+1)` (ordered k-tuples from `[N]`). For each ordered subset `S ⊂ [N]` of size `k`, the corresponding observation is `g( {x^{π(i)}}_{i∈S} )`.

The two specific instances in the paper:
- **NAE** (Section 3.3, 4.2): `g(x_1, x_2, x_3) = 1 − 1[x_1 = x_2 = x_3]`. Used in L&O-NAE-SAT and in the Figure 4 BP simulation.
- **Disagreement** (= planted m-coloring): `g(x', x'') = 1[x' ≠ x'']`. Discussed for theoretical motivation only.

### 4.3 Proposition 3.3 — formal hardness (informal restatement)

Let `x` be a sample from an L&O distribution with sparse predicate observations of arity `k` and predicate `g`. Let `γ = P(g(uniform random k-tuple) = 1)`. Let `D_KS`, `D_cond` be predicate-specific thresholds (defined via belief propagation, see §4.5). Suppose each token in `x` is independently masked with probability `α`. Let `M` be the masked set. **If**

```
1 − γ^{−1} · D_KS / (k · N^{k−1})  ≤  α  ≤  1 − γ^{−1} · D_cond / (k · N^{k−1})
```

**then**, under the **1-RSB cavity prediction** (Conjecture B.13), with probability `Ω_k(1)` over the masking, **no polynomial-time algorithm can solve the resulting masking subproblem of predicting any of the masked tokens among `x^{π(1)}, ..., x^{π(N)}` given `x[M]`.**

In plain English: there is a **range of masking fractions `α`** (equivalently, of noise levels `t = 1 − α_t`) where the masking subproblem the MDM is being trained on is **provably computationally intractable** (under a widely-believed but unproven physics-style conjecture).

### 4.4 Proof outline (Appendix B.3, B.4)

1. **Reduce to planted CSP.** Mask all latents and most observations. Kept observations are constraints `g(σ|_S) = 1` for the planted ground-truth assignment `σ` of the latents. The masking subproblem ≡ recovering `σ` from these constraints.
2. **Apply 1-RSB physics prediction.** Statistical-physics theory for planted random CSPs predicts a **gap between information-theoretic and computational solvability**: for `D_cond < kP/N < D_KS`, the planted solution is information-theoretically distinguishable from null but **no efficient algorithm achieves optimal overlap** (Krzakala & Zdeborová 2009).
3. **Belief Propagation behavior** distinguishes the regimes:
   - Below `D_cond`: planted-init BP and random-init BP both find the planted solution.
   - Between `D_cond` and `D_KS`: planted-init BP succeeds, random-init BP fails (paramagnetic fixed point is locally stable).
   - Above `D_KS`: both succeed (the trivial 1/m messages no longer fixed).
4. Figure 4 numerically computes `D_KS`, `D_cond` for `k=3, m=3, g=NAE`: `D_KS / k = 64` (analytic), `D_cond / k ≈ 50` (empirical from BP transition).

### 4.5 Definitions B.9–B.13 — message passing for planted CSPs

**Definition B.10 — BP update rules.** Messages `M^{i→S}_c`, `M^{S→i}_c` (variable→clause, clause→variable) for color `c ∈ {1, ..., m}`:

```
M^{i→S}_c[t+1]  ∝  ∏_{T ∋ i, T ≠ S}  M^{T→i}_c[t]                                              (Eq. 4)

M^{S→i}_c[t+1]  ∝  Σ_{σ̄ ∈ {1,...,m}^{S\i}}  g(σ̄ ∪_i c) · ∏_{j ∈ S, j ≠ i}  M^{j→S}_{σ̄_j}[t]   (Eq. 5)
```

Marginal at variable `i`: `μ_i^c ∝ ∏_{T ∋ i} M^{T→i}_c`. Recovered assignment `σ̂_i = argmax_c μ_i^c`.

**Overlap** (Definition B.9): `d(σ, σ̂) = (1/N) · min_{ρ ∈ S_m} Σ_i 1[σ_i = ρ(σ̂_i)]` — alphabet-symmetry-quotiented.

**Assumption B.11 — paramagnetic fixed point.** `Σ_{σ̄ ∈ {1,...,m}^{k−1}} g(σ̄ ∪_i c)` is constant across `c ∈ {1,...,m}`, so the all-`1/m` messages are a fixed point. NAE with `m=3` satisfies this.

**Definition B.12 — `D_KS`.** Largest avg degree at which BP is *locally stable* around the paramagnetic fixed point.

**Definition B.12 — `D_cond`.** Largest avg degree at which the planted CSP ensemble and the null model become mutually contiguous (statistically indistinguishable).

**Conjecture B.13 (1-RSB).** For `D_cond < kP/N < D_KS`, the best computationally efficient overlap is *strictly less than* the best information-theoretic overlap. ⇒ Prop 3.3.

### 4.6 Connection to the entropy filter (the modification)

The paper's Prop 3.3 says "in a specific range of masking fractions α, the subproblem is intractable, so MDM training wastes gradient there." This range is **defined by graph-theoretic / physics quantities** that depend on the data distribution.

The professor's idea: replace those analytic thresholds with a **data-driven proxy** — the model's own entropy at masked positions. Concretely:

| Subproblem regime | Analytic signature | Empirical signature (entropy of `p_θ` at masked positions) |
|---|---|---|
| Easy / already learned | low `α` (few masks) | **low entropy** — model is confident, gradient is near zero |
| Productive | mid `α` | **mid entropy** — useful learning signal |
| Intractable (Prop 3.3 region) | `1 − γ⁻¹·D_KS/(kN^{k−1}) ≤ α ≤ 1 − γ⁻¹·D_cond/(kN^{k−1})` | **high entropy** — model can't reduce uncertainty; gradient is noise |

The hypothesis: dropping the high-entropy and/or low-entropy tails of each batch's mask distribution should match or beat baseline at fixed wall-clock. This generalizes the paper's insight to any data distribution where the analytic thresholds are unknown.

---

## 5. The L&O-NAE-SAT distribution (Section 3.3, 4.2)

### 5.1 Exact construction

- Permutation `π = id`.
- Latents at positions `0, ..., N−1`.
- Observations at positions `N, ..., N+P−1`.
- **Pre-fixed (random)** triples `(i_{1,j}, i_{2,j}, i_{3,j}) ∈ [N]^3` for `j = 1, ..., P`.
- Latent prior: uniform over `{1, ..., m}`.
- Observation `O_j(x^0, ..., x^{N−1}) = NAE(x^{i_{1,j}}, x^{i_{2,j}}, x^{i_{3,j}}) = 1 − 1[x^{i_{1,j}} = x^{i_{2,j}} = x^{i_{3,j}}]` ∈ {0, 1}.

Sampling a sequence:
1. Draw `x^0, ..., x^{N−1}` iid uniform from `{1, ..., m}`.
2. Compute `x^{N+j} = O_j(x^0, ..., x^{N−1})` for each `j`.

### 5.2 m = 2 (derived, not stated)

Naive guessing on observation tokens:

```
P(NAE = 1 | iid uniform from {1,...,m}^3) = 1 − P(all equal)
                                          = 1 − m · (1/m)^3
                                          = 1 − 1/m^2
```

For `m = 2`: `1 − 1/4 = 0.75`. Table 1 caption explicitly says "naive guessing leads to 75% accuracy" — therefore **`m = 2` for L&O-NAE-SAT**.

(Figure 4 / Appendix B.4's BP simulation uses `m = 3`, a *different* setting.)

> **Resolution (2026-04-26):** Triple sampling is **without replacement** (each triple has three distinct indices). This makes population P(NAE = 1) exactly `1 − 1/m² = 0.75` for m = 2 — matching the paper's Table 1 caption — and aligns the data with the planted-CSP convention used in Conjecture B.13 (the 1-RSB cavity prediction is stated for the random k-uniform hypergraph = distinct-index tuples). See methodology_notes.md Q1.

### 5.3 Padding (Section 3.3 / Appendix C.2.1)

For the (N=20, P=280) experiment: `N + P = 300`. Pad with 212 tokens of value `2` to reach total length 512, which is the max sequence length of the 19M MDM with RoPE.

### 5.4 Worked example, (N, P) = (5, 10), m = 2

**Generator convention** (binding for `tests/test_lo_nae_sat.py`, switched to without-replacement on 2026-04-26 per the user's Phase 7 decision; methodology Q1 resolved):

```python
rng = np.random.RandomState(42)
triples = np.empty((P, 3), dtype=np.int64)
for j in range(P):
    triples[j] = rng.choice(N, size=3, replace=False)   # three distinct indices per triple
```

This matches the planted-CSP convention used in Conjecture B.13: the 1-RSB cavity prediction is stated for the random k-uniform hypergraph, i.e., distinct-index k-tuples. Aligning our triple distribution with the theoretical danger zone is what makes Prop 3.3's prediction directly applicable to our experiments.

For `(N, P, seed) = (5, 10, 42)`, this produces the following ten triples (verified empirically 2026-04-26 with the test suite):

```
j=0: (1, 4, 2)    j=5: (4, 2, 0)
j=1: (3, 1, 2)    j=6: (2, 0, 4)
j=2: (1, 0, 3)    j=7: (1, 2, 4)
j=3: (0, 1, 2)    j=8: (0, 3, 1)
j=4: (0, 2, 3)    j=9: (0, 1, 2)
```

Note: every triple has three *distinct* indices. (Different triples can share the *set* of indices in different orders — e.g., j=3 `(0,1,2)` and j=9 `(0,1,2)` happen to be identical here; that is allowed.)

Now fix latents `(x^0, x^1, x^2, x^3, x^4) = (1, 2, 1, 2, 1)` (chosen by hand; **not** drawn from the rng — the test injects them directly to make the assertion deterministic of triples alone).

Observations:
- `j=0` triple `(1,4,2)` → values `(2,1,1)` → NAE = 1
- `j=1` triple `(3,1,2)` → values `(2,2,1)` → NAE = 1
- `j=2` triple `(1,0,3)` → values `(2,1,2)` → NAE = 1
- `j=3` triple `(0,1,2)` → values `(1,2,1)` → NAE = 1
- `j=4` triple `(0,2,3)` → values `(1,1,2)` → NAE = 1
- `j=5` triple `(4,2,0)` → values `(1,1,1)` → all-equal → NAE = 0
- `j=6` triple `(2,0,4)` → values `(1,1,1)` → all-equal → NAE = 0
- `j=7` triple `(1,2,4)` → values `(2,1,1)` → NAE = 1
- `j=8` triple `(0,3,1)` → values `(1,2,2)` → NAE = 1
- `j=9` triple `(0,1,2)` → values `(1,2,1)` → NAE = 1

Full sequence (no padding): `[1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]` (length L = N + P = 15).

This is the ground truth for `tests/test_lo_nae_sat.py`.

---

## 6. The π-learner experiment (Section 3.2, Figure 2 left)

A **π-learner** is a causal autoregressive model trained on the **permuted** sequence `π(x_0)`. The likelihood it computes is

```
log p_θ(x_0) = Σ_{i=0}^{L−1}  log p_θ( x_0^{π(i)} | x_0[π{i, ..., L−1}] )    ... (3)
```

Five lines on Figure 2 left (negative log-likelihood vs log(FLOPs)):

| Line | Permutation distribution | Interpretation |
|---|---|---|
| AR | `π = id` | Standard ARM |
| MDM | full MDM | Equivalent (Eq. 1) to averaging π-learners over `Unif(S_L)` |
| π-learner-much-closer | identity + `√L` random transpositions, 3 samples | Almost identity, slight mixing |
| π-learner-closer | identity + `L/10` random transpositions, 3 samples | More mixing |
| π-learner-unif | `π ~ Unif(S_L)`, 3 samples | Fully random ordering |

**Result.** `AR < much-closer < closer < unif < MDM` in validation loss on text. ARM has lowest loss because text *is* left-to-right. MDM is slightly worse than uniform π-learner — explained by **multitask learning bonus** ("blessing of task diversity"); Tripuraneni 2021, Maurer 2016, Ruder 2017.

**Critical architectural detail.** All π-learners use **learnable absolute positional embeddings** (NOT RoPE). RoPE encodes left-to-right inductive bias; using it for π-learners would conflate position with order. Paper:
> "Given that RoPE has an inductive bias towards left-to-right ordering, we employ a learnable positional embedding layer for all experiments to correct this. Consequently, we also re-run the baseline results, where RoPE was employed."

**IsoFLOP analysis.** For each FLOPs budget `C`: vary non-embedding parameter count, set training tokens = `C / (6N)` (Hoffmann 2022 / Kaplan 2020), pick model size with lowest validation loss → one data point. Repeat across budgets.

---

## 7. Reproduction targets — copy verbatim from paper

These are the exact numbers `baseline/` and `entropy_filtered/` will be measured against. Source pages cited.

### 7.1 Table 1 — L&O-NAE-SAT vanilla vs adaptive (page 8)

Naive guessing baseline: **75%**.

| (N, P) | Vanilla inference | Adaptive inference (Top-Prob-Margin) |
|---|---|---|
| (25, 275) | 78.06 % | 93.76 % |
| (30, 270) | 75.70 % | 93.54 % |
| (40, 260) | 74.60 % | 92.21 % |
| (50, 250) | 67.94 % | 90.01 % |
| (100, 200) | 62.84 % | 88.91 % |

### 7.2 Table 2 — Sudoku (page 8)

| Method | # Param | Accuracy |
|---|---|---|
| ARM (w/o ordering) | 42M | 9.73 % |
| ARM (with ordering) | 42M | 87.18 % |
| MDM (vanilla) | 6M | 6.88 % |
| MDM (Top probability) | 6M | 18.51 % |
| MDM (Top prob. margin) | 6M | **89.49 %** |

### 7.3 Table 3 — Zebra (Einstein) puzzle (page 9)

| Method | # Param | Accuracy |
|---|---|---|
| ARM (w/o ordering) | 42M | 80.31 % |
| ARM (with ordering) | 42M | 91.17 % |
| MDM (vanilla) | 19M | 76.9 % |
| MDM (Top probability) | 19M | **98.5 %** |
| MDM (Top prob. margin) | 19M | 98.3 % |

### 7.4 Table 4 — LLaDA 8B on language tasks (page 10)

| Method | HumanEval-Single | HumanEval-Multi | HumanEval-Split | Math | MMLU | ROCStories |
|---|---|---|---|---|---|---|
| Vanilla | 31.8 % | 16.5 % | 14.2 % | 28.5 % | 33.2 % | 21.23 % |
| Top probability | 32.9 % | 20.8 % | 18.4 % | 31.3 % | **36.5 %** | 21.10 % |
| Top prob. margin | **33.5 %** | **25.4 %** | **22.3 %** | **34.3 %** | 35.4 % | **21.41 %** |

### 7.5 Table 5 — Hard Sudoku generalization (page 10)

Same models as Table 2; evaluated on the puzzles **not** solvable by Shah et al.'s 7 strategies (~1M puzzles requiring backtracking).

| Method | # Param | Accuracy |
|---|---|---|
| ARM (with ordering) | 42M | 32.57 % |
| MDM (random) | 6M | 3.62 % |
| MDM (Top probability) | 6M | 9.44 % |
| MDM (Top prob. margin) | 6M | **49.88 %** |

### 7.6 Figure 2 top-right — π-learner validation loss (170M MDM, text)

| Task | π = id | π ~ Closer(S_L) | π ~ Unif(S_L) |
|---|---|---|---|
| Val. Loss | 3.171 | 3.212 | 3.245 |

### 7.7 Figure 3 — Generative Perplexity (text, 1.1B MDM, page 7)

The plot shows GenPPL and Entropy vs Sampling Steps ∈ {250, 500, 750, 1000, 1250, 1500, 1750, 2000}. Approximate endpoints (read off the plot):

| Steps | Vanilla GenPPL | Adaptive GenPPL | Vanilla Entropy | Adaptive Entropy |
|---|---|---|---|---|
| 250 | ~24 | ~17 | ~5.2 | ~5.0 |
| 2000 | ~21 | ~14 | ~5.1 | ~4.6 |

The take-away (per paper): adaptive is **always better in GenPPL**, and the entropy gap is small enough that diversity is preserved.

### 7.8 Figure 4 — BP overlap on planted CSP (k=3, m=3, NAE, N=10000)

x-axis: D/k ∈ [43, 68]. y-axis: overlap.

- Random init: overlap stays at ≈ 0.33 (paramagnetic) until D/k ≈ 64, then jumps to ≈ 1.0.
- Planted init: overlap jumps from ≈ 0.33 to ≈ 1.0 at D/k ≈ 50.
- Annotated: `D_KS / k = 64`, `D_cond / k ≈ 50`.

---

## 8. Architecture decisions (binding for `baseline/`)

1. **MDM** uses **bidirectional self-attention**. No causal mask. The model must attend left and right to use unmasked context.
2. **MDM denoiser is time-embedding-free.** `p_θ(· | x_t, t) = p_θ(· | x_t)`. Do not condition on `t`. Section 2:
   > "In practice, a time-embedding-free architecture for the denoising network … is generally used as `x_t` implicitly contains information about `t` via the number of masked tokens."
3. **ARMs and π-learners** use **causal self-attention** (standard GPT-style triangular mask).
4. **Positional embeddings.**
   - π-learner / scaling-law text experiments (Section 3.2): **learnable absolute** (NOT RoPE). Paper explicitly re-runs RoPE baselines.
   - L&O-NAE-SAT (Section 3.3 / Appendix C.2.1): **RoPE**, max sequence length 512.
   - Sudoku, Zebra (Section 4.2 / Appendix D.2): the codebase of Ye et al. 2024 — GPT-2 architecture defaults. Verify after cloning whether their default is RoPE or learnable.
5. **Output dimension.** The denoiser outputs a categorical over the data alphabet (and only the data alphabet — the mask token 0 is never a valid prediction; in code, mask the mask-token logit to `−∞` before softmax during inference).
6. **No dropout** beyond what the codebase defaults specify (Ye et al. 2024 / SMDM defaults).

### 8.1 Model sizes

| Setting | Params | Rough config (hidden / layers / heads / FF) |
|---|---|---|
| Sudoku MDM | 6M | ~256 / 6 / 4 / 1024 (verify via Ye et al. config) |
| Zebra MDM, L&O-NAE-SAT MDM | 19M | ~384 / 8 / 6 / 1536 |
| ARM baselines | 42M | ~512 / 10 / 8 / 2048 |
| Text MDM (Figure 2 top-right val loss) | 170M | 12 / 768 / 12 / 3072 (GPT-2 small-ish) |
| Text MDM (Figure 3) | 1.1B | TinyLlama config (Zhang et al. 2024) |
| LLaDA | 8B | LLaDA-8B-Base from Nie et al. 2025 |

Exact configs depend on the cloned codebases (SMDM for scaling laws, Ye et al. 2024 for puzzles).

---

## 9. Hyperparameters by experiment (Appendices C, D)

### 9.1 Optimizer / scheduler (Appendix C.1, applies "throughout the paper unless otherwise specified")

| Field | Value |
|---|---|
| Optimizer | AdamW (Loshchilov & Hutter 2017) |
| β₁ | 0.9 |
| β₂ | 0.95 |
| Weight decay | 0.1 |
| LR schedule | cosine |
| Max LR | 4 × 10⁻⁴ |
| Min LR | 4 × 10⁻⁵ |
| Sequence length (text) | L = 2048 |

### 9.2 L&O-NAE-SAT

| Field | Section 3.3 (Fig. 2 right bot.) | Section 4.2 / Table 1 |
|---|---|---|
| (N, P) | (20, 280) | (25,275), (30,270), (40,260), (50,250), (100,200) |
| Padding | last 212 tokens → value 2, total length 512 | total length 300, padded to 512 (consistent) |
| Model | 19M MDM with **RoPE**, max seq 512 | 19M MDM (presumably same arch) |
| Iterations | 2 × 10³ for the model under study, 5 × 10⁴ for the proxy (Bayes-optimal approximation) | not stated; use 5 × 10⁴ to converge |
| Error metric | `E_{x_0}[ |log p_θ(x_0|x_0[M]) − log p_data(x_0|x_0[M])|² ]`, proxy approximated by long-trained MDM | accuracy of observation tokens under sampling |
| Trial setup | for each `ℓ ∈ [1, N−1]`, mask `ℓ` latents and `ℓ · P/N` observations; repeat 1000 times (Figure 2 uses ℓ=11) | 50 reverse steps per sample, Gumbel coeff 0.5 |

### 9.3 Sudoku and Zebra (Appendix D.2)

| Field | Value |
|---|---|
| Dataset source | **Shah et al. 2024**, who filtered Radcliffe (2020) Kaggle 3M puzzles by 7 strategies (no backtracking) |
| Hard test set | the *complement* — Radcliffe puzzles that require strategies outside the 7 or backtracking (~1M) |
| Codebase | **Ye et al. 2024** (`https://github.com/HKUNLP/diffusion-vs-ar`, verified 2026-04-26) |
| Sudoku MDM | 6M GPT-2 model |
| Zebra MDM | 19M model |
| LR | 0.001 |
| Batch size | 128 |
| Epochs | 300 |
| Reverse sampling steps | 50 |
| Oracle noise | Gumbel, coefficient 0.5 |

### 9.4 Text (Section 3.2 + Figures 2 left, 2 top-right, 3) — Appendices C.1, C.2.2, D.1.2

| Field | Value |
|---|---|
| Dataset | SlimPajama (Soboleva et al. 2023) |
| Tokenizer | (codebase default — SMDM uses GPT-2 tokenizer; verify) |
| L | 2048 |
| Model for Figure 2 top-right | 170M MDM |
| Model for Figure 3 | 1.1B MDM |
| Val-loss expectation samples | 1024 of `x_0 ~ p_data` per task per π distribution (Appendix C.2.2) |
| Permutation samples | 3 per distribution (Closer / Much-Closer / Unif) |
| Generative-PPL evaluator | LLaMA-2 7B (Touvron et al. 2023, base model not chat) |
| Entropy metric | unigram, `Σ p_i log p_i` where `p_i = #{x^j = i} / L` |
| Sampling-step counts (Figure 3) | 250, 500, 750, 1000, 1250, 1500, 1750, 2000 |
| Oracle noise | Gaussian `ε`, std unspecified (existing notebook uses σ=0.001; tune if needed) |

### 9.5 LLaDA-8B (Section 4.4 / Appendix D.3)

| Field | Value |
|---|---|
| Base model | LLaDA-8B-Base, `https://huggingface.co/GSAI-ML/LLaDA-8B-Base` (Nie et al. 2025, arXiv:2502.09992) |
| Infilling tasks | HumanEval-Infill (Bavarian et al. 2022): single-line, multi-line, split. ROCStories. *Non-autoregressive sampling*; output length = mask span. |
| Instruction tasks | Math, MMLU. *Semi-autoregressive sampling*; output length must be specified. Sampling configuration follows Nie et al. 2025. |

---

## 10. Datasets (where to actually get them)

| Dataset | Source | Notes |
|---|---|---|
| SlimPajama | `cerebras/SlimPajama-627B` (HF, gated; needs `huggingface-cli login` + accept terms) or `DKYoon/SlimPajama-6B` (community subset, ungated) | ~627B tokens total. Pre-tokenize to length-2048 packed sequences. |
| Sudoku 3M | `radcliffe/3-million-sudoku-puzzles-with-ratings` on Kaggle. Needs `KAGGLE_API_TOKEN`. | Filter via Shah et al. 2024's 7 strategies to get the easy split; remainder is the Table 5 hard split. |
| Sudoku train/test (filtered) | `https://github.com/kulinshah98/logic-puzzles` (verify it ships pre-filtered data, otherwise reproduce filter) | ARM-with-ordering uses Shah's solving order. |
| Zebra | Same Shah et al. 2024 repo | Tokenization follows their format; check repo. |
| LLaDA-8B-Base | `GSAI-ML/LLaDA-8B-Base` (HF) | ~16 GB. Pre-download on Bouchet login node before any compute job. |

---

## 11. Reference codebases (canonical URLs, verified 2026-04-26)

| Codebase | URL | Used for |
|---|---|---|
| **SMDM** (Nie et al. 2024, "Scaling up masked diffusion models on text", arXiv:2410.18514) | `https://github.com/ML-GSAI/SMDM` | Section 3.2 scaling-law experiments; 170M / 1.1B text MDM training. |
| **Diffusion-vs-AR** (Ye et al. 2024, arXiv:2410.14157) — VERIFIED via arXiv abstract page | `https://github.com/HKUNLP/diffusion-vs-ar` | Section 4.2 puzzles (Sudoku, Zebra) training and inference. **Both prior cites in our docs were wrong** — `HKUNLP/discrete-diffusion` and `HKUNLP/diffusion-of-thoughts` both wrong. |
| **LLaDA** (Nie et al. 2025, arXiv:2502.09992) | `https://github.com/ML-GSAI/LLaDA` | Section 4.4 LLaDA-8B inference, semi-autoregressive sampling configuration. |
| **llm-reasoning-logic-puzzles** (Shah et al. 2024, arXiv:2409.10502, NeurIPS 2024) | `https://github.com/kulinshah98/llm-reasoning-logic-puzzles` (verified 2026-04-26) | Sudoku/Zebra datasets WITH strategy IDs and solver-order sequences embedded; ARM-with-ordering teacher forcing implemented in JAX/Flax. **Note:** the URL `kulinshah98/logic-puzzles` cited in earlier drafts is a 404; the correct repo name is `llm-reasoning-logic-puzzles`. |

---

## 12. Implementation gotchas (paper-derived, must respect in `baseline/`)

### 12.1 Mask-token / observation-value collision in L&O-NAE-SAT

The paper says (Section 2): "We use 0 to denote the 'mask' token." And (Section 3.1, Definition 3.1): `p_data` is over `{0, ..., m}^L`. NAE-output observations take values in `{0, 1}`, so an unmasked observation can have token value 0 — **identical to the mask token**.

**Resolution.** Use a separate mask token id distinct from any data value. Concretely for L&O-NAE-SAT with m=2:

- Latent tokens take values 1 or 2 (the m=2 alphabet, shifted by +1).
- Observation tokens take values 0 or 1 (raw NAE outputs).
- Padding tokens take value 2 (per Appendix C.2.1).
- Mask token = 3 (a fresh id, never appears as data).
- Embedding vocab size = 4: {0, 1, 2, 3}.

This matches the existing notebook convention and avoids ambiguity. Document in `baseline/src/data/lo_nae_sat.py`.

### 12.2 Loss weighting (Section 2.3)

For linear schedule `α_t = 1 − t`: `α'_t = −1`, so the weight `α'_t / (1 − α_t) = −1 / t`. The paper writes the loss as a positive integral with `−log p_θ`, so the negative cancels. In practice, the equivalent discrete-time loss (Prop E.1, Eq. 6) is

```
L_θ = − Σ_{n=1}^{L}  E_{x(n) ~ q̃(·|x_0)}  [ (1/n) · Σ_{ℓ : x^ℓ(n) = 0}  log p_θ(x_0^ℓ | x(n)) ]
```

Implementation: per training step, sample mask count `n ~ Uniform{1,...,L}`, mask `n` random positions, compute cross-entropy at masked positions, **divide per-sample loss by `n`**, then mean over batch. Critical: the `1/n` weight, not `1/L` and not `1/|M_in_batch|`.

### 12.3 K computation in inference

```
K = max(1, round(num_masked · (α_s − α_t) / (1 − α_t)))   # bound by num_masked
```

For linear `α_t = 1 − t`: `K = max(1, round(num_masked · (t − s) / t))`. Off-by-one and rounding issues will silently degrade adaptive inference.

### 12.4 Conditional inference for puzzles (Sudoku, Zebra)

Given clues are **never modified**. The set of unmaskable positions is `{i : x_t^i = mask_token AND fixed_mask[i] = False}`. Verify by an assertion at the end of inference: `clues == final_x[clue_positions]`.

### 12.5 Gumbel noise on scores, not logits

```
score_i = c_i + 0.5 · Gumbel(0, 1)        # scores
S = TopK_i(score_i)                        # selection
x_s^i ~ Cat( softmax(logits_i) )            # sampling — uses raw logits, NOT score-perturbed
```

The Gumbel is only there to break ties between near-equal certainties, NOT to reweight token probabilities.

### 12.6 Inference output excludes the mask token

Before sampling at a position to unmask:
```python
logits[mask_token_id] = float('-inf')
probs = softmax(logits)
```
Otherwise the model can sample the mask token, leaving stranded masks at the end of generation.

### 12.7 Text experiments use Gaussian (NOT Gumbel) noise

Appendix D.1.2 uses `ε ~ Normal(0, σ²)` added to the margin. σ unspecified — start with 0.001 (the existing notebook's value) and tune.

### 12.8 Semi-autoregressive sampling for LLaDA instruction tasks

For Math and MMLU on LLaDA-8B, use the LLaDA paper's sampling config — block-by-block generation rather than full non-autoregressive. For HumanEval-Infill and ROCStories (infilling), use full non-autoregressive.

### 12.9 ARM-with-ordering baseline (Tables 2, 3, 5)

This is **not** standard left-to-right ARM. Per Shah et al. 2024:
1. For each puzzle, compute a valid solving order using constraint propagation (the 7 strategies).
2. Reorder the sequence so the solving order is left-to-right.
3. Train an ARM via teacher forcing on this reordered sequence.

The fact that MDM with adaptive inference *beats* this baseline (89.49% vs 87.18% on Sudoku) is the paper's strongest empirical claim and **must** be reproduced exactly.

---

## 13. Scope for the new project

### 13.1 Reproduction targets (Phase 4 — `baseline/`)

Minimum viable reproduction set (per the new mission):

- **Table 1** — L&O-NAE-SAT, 5 (N, P) configs, vanilla + adaptive (Top-Prob-Margin). Cheap, fast.
- **Table 2** — Sudoku, all 5 methods. Headline result.

Defer (potential follow-up):
- Table 3 (Zebra), Table 4 (LLaDA), Table 5 (hard Sudoku), Figure 2 (scaling laws), Figure 3 (text GenPPL), Figure 4 (BP simulation).

### 13.2 Modification targets (Phase 5 — `entropy_filtered/`)

Five training-loop variants, all sharing data / arch / optimizer / seed list:

| Variant | Mode | Filtering |
|---|---|---|
| `none` | unfiltered | (control) |
| `bottom_filter` | drop low-entropy masks | "remove wasted gradient on trivial subproblems" |
| `top_filter` | drop high-entropy masks | "remove intractable subproblems" — direct test of Prop 3.3 |
| `band_filter` | drop both ends, absolute thresholds | full proposal, fixed thresholds |
| `percentile_band` | drop both ends, batch-relative percentiles | self-calibrating to model state |

User-confirmed defaults:
- **500 steps unfiltered warmup** before turning the filter on (configurable).
- **Both** absolute and percentile thresholds implemented; selected via config flag.
- Per-batch acceptance rate **logged** (audit trail).
- Compute budget **matched on wall-clock**, not on accepted gradient steps.
- 3 seeds per variant; mean ± std reported.

### 13.3 Headline experiment

For each variant in `{none, bottom, top, band, percentile_band}`, for each (N, P) ∈ Table 1, for 3 seeds:
- Train under matched wall-clock.
- Evaluate every 5K steps with vanilla and adaptive (Top-Prob-Margin) inference on 1000 held-out samples.
- Plot training curves and a final-accuracy table.

Then repeat on Sudoku.

---

## 14. Open ambiguities flagged for design decisions

1. **Noise schedule for puzzles.** Paper does not state the schedule for Sudoku/Zebra explicitly. Ye et al. 2024's codebase uses linear by default. **Decision for `baseline/`**: linear (`α_t = 1 − t`). Re-verify after cloning their repo.
2. **σ for Gaussian noise on text oracle.** Not stated. Existing notebook used 0.001. **Decision for `baseline/`**: 0.001 default; sweep `{0.0, 0.001, 0.01, 0.1}` if Figure 3 reproduction is off.
3. **Sudoku ARM-with-ordering solve-order ties.** Multiple cells may be solved in the same iteration of a strategy. **Decision for `baseline/`**: break ties by row-major position. Document; ablate later if results diverge from paper.
4. **Padding for Table 1 (N, P) ≠ (20, 280).** Paper explicitly pads (20, 280) to 512 with token value 2. For other configs (N+P=300), same padding (212 tokens of value 2 → length 512) is the natural extension. **Decision for `baseline/`**: pad all 5 configs to 512.
5. **Tokenizer for SMDM scaling-law experiment.** Paper says "we leverage the codebase from Nie et al. 2024" — verify SMDM's tokenizer choice after cloning.
6. **Number of evaluation samples for Table 1.** Paper does not state. **Decision for `baseline/`**: 500 samples per (N, P, strategy), seed-independent.

---

## 15. Quick lookup index

When implementing, search for:

- "MDM forward process" → §2.1
- "MDM reverse process exact posterior" → §2.2
- "Loss equation 1 (the integral form)" → §2.3 + §12.2
- "Vanilla inference" → §2.6 + §12.3
- "Top-Probability margin oracle" → §3.2
- "Gumbel noise" → §3.4 + §12.5
- "L&O-NAE-SAT exact spec" → §5
- "Why m=2" → §5.2
- "How to filter Sudoku train set" → §10 (Shah et al. repo) + §12.9
- "Hyperparameters table" → §9
- "Reproduction targets verbatim" → §7
- "Architectural decisions" → §8
- "1-RSB hardness theory" → §4.3, §4.5
- "Connection from theory to entropy filter" → §4.6, §13.2

---

**End of paper notes.** Continue to Phase 3 (audit existing notebooks against this document → `code_audit.md`).
