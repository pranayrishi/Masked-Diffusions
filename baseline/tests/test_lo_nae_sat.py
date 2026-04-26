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
        [3, 4, 2],
        [4, 4, 1],
        [2, 2, 2],
        [4, 3, 2],
        [4, 1, 3],
        [1, 3, 4],
        [0, 3, 1],
        [4, 3, 0],
        [0, 2, 2],
        [1, 3, 3],
    ],
    dtype=np.int64,
)

EXPECTED_SEQUENCE_FOR_LATENTS_12121 = np.array(
    [1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
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

def _expected_p_nae1_given_triples(triples: np.ndarray, m: int) -> float:
    """Analytical E[P(NAE=1)] over uniform iid latents, marginalizing over the FIXED triples.

    For a triple (i, j, k):
      - all three indices equal: P(NAE=1) = 0
      - exactly two indices equal: P(NAE=1) = (m-1)/m
      - all three indices distinct: P(NAE=1) = 1 - 1/m^2

    The paper's "75 % naive accuracy" claim is the all-distinct case for m=2.
    For finite N with with-replacement triples, the population mean is < 75 %
    because some triples are degenerate. We assert against the *correct* mean
    so the test is exact, not approximate.
    """
    p_per_triple = []
    for (i, j, k) in triples.tolist():
        a, b, c = (i == j), (j == k), (i == k)
        if a and b and c:                # all three equal
            p_per_triple.append(0.0)
        elif a or b or c:                # exactly two equal
            p_per_triple.append((m - 1) / m)
        else:                            # all distinct
            p_per_triple.append(1 - 1 / (m * m))
    return float(np.mean(p_per_triple))


def test_empirical_observation_one_rate_matches_analytical_expectation():
    """Generate 1000 sequences and compare the observation-1 frequency to the
    *exact* analytical expectation for the seed-42 triples (not the asymptotic 75%)."""
    cfg = LoNaeSatConfig(N=20, P=280, m=2, pad_to=None, mask_token_id=3, vocab_size=4, seed=42)
    seqs, triples = generate_dataset(cfg, num_samples=1000, sample_seed=123)
    obs = seqs[:, cfg.N : cfg.L_data]
    empirical = float((obs == 1).mean())
    expected = _expected_p_nae1_given_triples(triples, m=cfg.m)
    # 1000 × 280 = 280k draws of a Bernoulli around `expected`; std ≈ 0.001. Use 0.01 band.
    assert abs(empirical - expected) < 0.01, (
        f"empirical {empirical:.4f} vs analytical {expected:.4f}"
    )


def test_naive_75_percent_holds_when_triples_are_unique():
    """The paper's 75% is exact when triples are all-distinct (no degenerates).

    We construct a mini-dataset whose triples are explicitly distinct and verify the
    population P(NAE=1) is exactly 0.75 in the limit of many samples.
    """
    rng = np.random.RandomState(7)
    N, P = 20, 200
    # Sample distinct triples (no repeated indices within a triple)
    triples = []
    while len(triples) < P:
        cand = rng.choice(N, size=3, replace=False)
        triples.append(cand)
    triples = np.asarray(triples, dtype=np.int64)

    # Generate 5000 sequences using these triples directly (bypass the seed-42 generator)
    L = N + P
    rng2 = np.random.RandomState(123)
    seqs = np.empty((5000, L), dtype=np.int64)
    for s in range(5000):
        latents = rng2.randint(1, 3, size=N)
        seqs[s, :N] = latents
        for j in range(P):
            i1, i2, i3 = triples[j]
            seqs[s, N + j] = 0 if (latents[i1] == latents[i2] == latents[i3]) else 1
    obs = seqs[:, N:]
    frac_one = float((obs == 1).mean())
    assert abs(frac_one - 0.75) < 0.005, f"unique-triple naive accuracy drifted: {frac_one:.4f}"


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
