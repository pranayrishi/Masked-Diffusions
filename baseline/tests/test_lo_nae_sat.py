"""Test the L&O-NAE-SAT generator against the worked example in paper_notes.md §5.4.

Concrete assertion (binding):
  With seed 42 and (N, P) = (5, 10), and latents (1, 2, 1, 2, 1) injected by hand,
  the encoded sequence (no padding) must equal:
      [1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]

Also verifies:
  - The seed-42 triples match the published §5.4 table.
  - Naive observation accuracy = 0.75 for m = 2 (paper Table 1 caption).
  - Empirical-ish: random guessing on observations of a 1000-sample dataset is
    near 0.75 (within a binomial confidence band).
"""

from __future__ import annotations

import numpy as np

from src.data import (
    LoNaeSatConfig,
    encode_sequence,
    generate_dataset,
    make_triples,
    naive_observation_accuracy,
)


# ---------------------------------------------------------------------------
# Test 1: deterministic worked example
# ---------------------------------------------------------------------------

EXPECTED_TRIPLES_SEED42_5_10 = np.array(
    [
        [1, 4, 2],
        [3, 1, 2],
        [1, 0, 3],
        [0, 1, 2],
        [0, 2, 3],
        [4, 2, 0],
        [2, 0, 4],
        [1, 2, 4],
        [0, 3, 1],
        [0, 1, 2],
    ],
    dtype=np.int64,
)

EXPECTED_SEQUENCE_FOR_LATENTS_12121 = np.array(
    [1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    dtype=np.int64,
)


def test_seed42_triples_match_paper_notes():
    triples = make_triples(N=5, P=10, seed=42)
    np.testing.assert_array_equal(
        triples, EXPECTED_TRIPLES_SEED42_5_10,
        err_msg="seed-42 triples drifted from paper_notes.md §5.4",
    )


def test_worked_example_sequence_exact():
    """The §5.4 worked example: latents (1,2,1,2,1) + seed-42 triples → exact sequence."""
    triples = make_triples(N=5, P=10, seed=42)
    latents = np.array([1, 2, 1, 2, 1], dtype=np.int64)
    seq = encode_sequence(latents, triples, pad_to=None)
    np.testing.assert_array_equal(
        seq, EXPECTED_SEQUENCE_FOR_LATENTS_12121,
        err_msg="encode_sequence does not reproduce paper_notes.md §5.4 expected output",
    )


# ---------------------------------------------------------------------------
# Test 2: naive accuracy = 0.75 for m = 2
# ---------------------------------------------------------------------------

def test_naive_accuracy_75_percent_for_m2():
    """Paper Table 1 caption says naive guessing → 75 % observation accuracy.

    Mathematically: P(NAE=1) = 1 - 1/m². For m=2 this is 0.75. We assert this.
    """
    p = naive_observation_accuracy(m=2)
    assert abs(p - 0.75) < 1e-12, f"naive accuracy for m=2 should be 0.75, got {p}"


# ---------------------------------------------------------------------------
# Test 3: empirical observation distribution on a 1000-sample dataset
# ---------------------------------------------------------------------------

def test_seed42_triples_have_three_distinct_indices():
    """Without-replacement per triple: every triple's three indices are distinct."""
    triples = make_triples(N=20, P=280, seed=42)
    for j, t in enumerate(triples):
        assert len(set(t.tolist())) == 3, f"triple j={j} has duplicate indices: {t.tolist()}"


def test_empirical_observation_one_rate_close_to_75_percent():
    """With without-replacement triples, the population P(NAE=1) is exactly 1 − 1/m².

    For m=2 this is 0.75. We assert this with a 1000-sample × 280-observation dataset:
    280,000 Bernoulli(0.75) draws have std ≈ 0.0008 — a 0.01 band is conservative.
    """
    cfg = LoNaeSatConfig(N=20, P=280, m=2, pad_to=None, mask_token_id=3, vocab_size=4, seed=42)
    seqs, _ = generate_dataset(cfg, num_samples=1000, sample_seed=123)
    obs = seqs[:, cfg.N : cfg.L_data]
    empirical = float((obs == 1).mean())
    assert abs(empirical - 0.75) < 0.01, (
        f"empirical P(NAE=1) drifted from asymptotic 0.75: {empirical:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: padding shape and padding values
# ---------------------------------------------------------------------------

def test_padding_shape_and_value():
    """For (N, P) = (20, 280) with pad_to=512, last 212 tokens must equal pad_value=2."""
    cfg = LoNaeSatConfig(N=20, P=280, m=2, pad_to=512, pad_value=2, mask_token_id=3, vocab_size=4, seed=42)
    seqs, _ = generate_dataset(cfg, num_samples=4, sample_seed=999)
    assert seqs.shape == (4, 512)
    pad = seqs[:, cfg.L_data :]
    assert (pad == cfg.pad_value).all(), f"padding region contains values other than {cfg.pad_value}"
    # Latents are in {1, 2}; observations are in {0, 1}; mask token (3) must NOT appear in raw data
    assert (seqs != cfg.mask_token_id).all(), "raw L&O-NAE-SAT data must not contain the mask token"
