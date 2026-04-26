"""Data: L&O-NAE-SAT generator (and Sudoku loaders, added later).

L&O-NAE-SAT — paper Section 3.3 / Definition 3.1.

Notation in this file:
  N latent positions (values 1..m), P observation positions (values 0/1 from NAE),
  m alphabet size (=2 by paper-notes §5.2 derivation), L_data = N + P.
  Optionally pad to `pad_to` with `pad_value` (defaults: 512, 2 — paper Appendix C.2.1).
  mask_token_id is distinct from any data value; default 3 (= max data value + 1).

Vocabulary used by the model:
  {0, 1}                   observation tokens (NAE outputs)
  {1, 2}                   latent tokens     (m=2 alphabet, shifted to 1..m)
  {2}                      padding token     (paper Appendix C.2.1: value 2)
  {3}                      mask token        (chosen here; distinct from all data values)
  → vocab_size = 4 ({0, 1, 2, 3})
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoNaeSatConfig:
    N: int                       # number of latent positions
    P: int                       # number of observation positions
    m: int = 2                   # alphabet size (paper-derived: 75% naive ↔ m=2)
    pad_to: int | None = 512     # final sequence length after padding (paper §C.2.1)
    pad_value: int = 2           # padding token (paper §C.2.1)
    mask_token_id: int = 3       # mask token id (must not collide with any data value)
    vocab_size: int = 4          # size of model output head: {0, 1, 2, 3}
    seed: int = 42               # RandomState seed for triple selection

    @property
    def L_data(self) -> int:
        return self.N + self.P

    @property
    def L(self) -> int:
        return self.pad_to if self.pad_to is not None else self.L_data


def _nae(values: tuple[int, int, int]) -> int:
    """NAE(x1, x2, x3) = 0 if all equal, 1 otherwise (paper §5.1)."""
    a, b, c = values
    return 0 if (a == b == c) else 1


def make_triples(N: int, P: int, seed: int) -> np.ndarray:
    """Sample the P fixed observation triples — pre-fixed once for the distribution.

    Convention (binding for tests, switched 2026-04-26 per the user's Phase 7
    decision): WITHOUT replacement per triple. Each of the P triples is sampled
    independently as `np.random.RandomState(seed).choice(N, size=3, replace=False)`,
    so every triple has THREE DISTINCT indices.

    This matches the planted-CSP convention used in Conjecture B.13 (the
    1-RSB cavity prediction is stated for the random k-uniform hypergraph =
    distinct-index k-tuples) and makes the paper's "naive guessing leads to
    75% accuracy" claim hold *exactly* (no degenerate triples lower P(NAE=1)
    below the asymptotic 0.75).

    See methodology_notes.md Q1 for the resolution rationale.
    """
    rng = np.random.RandomState(seed)
    triples = np.empty((P, 3), dtype=np.int64)
    for j in range(P):
        triples[j] = rng.choice(N, size=3, replace=False)
    return triples


def sample_latents(N: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample one latent assignment uniformly from {1, ..., m}^N."""
    return rng.randint(1, m + 1, size=N)


def encode_sequence(
    latents: np.ndarray,
    triples: np.ndarray,
    pad_to: int | None = None,
    pad_value: int = 2,
) -> np.ndarray:
    """Build a single (latents || observations || padding) sequence.

    latents: shape (N,), int values in {1, ..., m}
    triples: shape (P, 3), int values in [0, N)
    Returns int64 array of length pad_to (if set) or N+P.
    """
    N = len(latents)
    P = len(triples)
    L_data = N + P
    L = pad_to if pad_to is not None else L_data

    seq = np.empty(L, dtype=np.int64)
    seq[:N] = latents
    for j in range(P):
        i1, i2, i3 = triples[j]
        seq[N + j] = _nae((latents[i1], latents[i2], latents[i3]))
    if L > L_data:
        seq[L_data:] = pad_value
    return seq


def generate_dataset(cfg: LoNaeSatConfig, num_samples: int, sample_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of `num_samples` sequences from a single L&O-NAE-SAT distribution.

    The triples are deterministic in `cfg.seed`; the latent draws are deterministic in
    `sample_seed`. Splitting these means train/test splits use the SAME triples (the
    distribution is fixed once) but DIFFERENT latent draws.

    Returns (sequences, triples):
        sequences: shape (num_samples, cfg.L) int64
        triples:   shape (P, 3) int  — same triples used to build all sequences
    """
    triples = make_triples(cfg.N, cfg.P, cfg.seed)
    rng = np.random.RandomState(sample_seed)

    sequences = np.empty((num_samples, cfg.L), dtype=np.int64)
    for s in range(num_samples):
        latents = sample_latents(cfg.N, cfg.m, rng)
        sequences[s] = encode_sequence(latents, triples, pad_to=cfg.pad_to, pad_value=cfg.pad_value)
    return sequences, triples


def naive_observation_accuracy(m: int) -> float:
    """Probability that NAE evaluates to 1 on three iid uniform draws from {1,...,m}.

    Predicting always "1" (the majority class) gives this fraction correct.
    For m=2 returns 0.75, matching the paper's Table 1 caption.
    """
    return 1.0 - 1.0 / (m * m)
