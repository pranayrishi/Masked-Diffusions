"""Test the MDM loss against a hand-computed 4-token toy example using Eq. 6.

Eq. 6 (Prop E.1, Zheng et al. 2024 — paper-notes §2.4 / §12.2):
    L_θ = Σ_{n=1}^{L} (1/n) · E_{x(n) ~ q̃(·|x_0)} [ Σ_{ℓ : x^ℓ(n)=0}  -log p_θ(x_0^ℓ | x(n)) ]

Practical implementation (paper-notes §12.2 / src.diffusion.mdm_loss):
    For each sample b in a batch:
        n_b ~ Uniform{1, ..., L}
        choose n_b positions to mask uniformly at random
        per-sample loss = (1 / n_b) × Σ_{i ∈ M_b}  -log p_θ(x_0^{i,b} | x_masked^b)
    final_loss = mean over batch of per-sample losses

This file constructs a small but explicit example and compares the output of
`src.diffusion.mdm_loss` against a hand computation to within 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.diffusion import mdm_loss


def _hand_loss_one_sample(logits_row: torch.Tensor, x0_row: torch.Tensor, mask_row: torch.Tensor) -> float:
    """Compute the per-sample 1/n × Σ -log p_θ(x_0^i) directly, with no library help."""
    L, V = logits_row.shape
    log_softmax = torch.log_softmax(logits_row.double(), dim=-1)
    masked_positions = mask_row.nonzero(as_tuple=True)[0]
    n = int(masked_positions.numel())
    if n == 0:
        return 0.0
    # Σ -log p_θ(x_0^i)
    s = 0.0
    for i in masked_positions.tolist():
        s += -log_softmax[i, int(x0_row[i].item())].item()
    return s / n


def test_mdm_loss_matches_hand_computation_on_4_token_toy():
    """4-token sequences, batch size 2, vocab size 5. Hand-compute and compare."""
    torch.manual_seed(0)
    B, L, V = 2, 4, 5

    # Construct deterministic logits (NOT from a random init, so failures are reproducible)
    logits = torch.tensor(
        [
            [   # sample 0
                [ 1.0, -2.0,  0.5,  3.0,  0.0],
                [ 0.0,  0.0,  0.0,  0.0,  0.0],
                [-1.0,  2.0,  4.0, -2.0,  1.0],
                [ 0.5,  0.5,  0.5,  0.5,  0.5],
            ],
            [   # sample 1
                [ 2.0,  1.0,  0.0, -1.0, -2.0],
                [ 3.0, -3.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  0.0,  5.0],
                [ 1.0,  2.0,  3.0,  4.0,  0.0],
            ],
        ],
        dtype=torch.float64,
    )

    x0 = torch.tensor([
        [3, 1, 2, 4],   # sample 0
        [0, 0, 4, 3],   # sample 1
    ], dtype=torch.long)

    # Mask sample 0 positions {1, 2}; mask sample 1 positions {0, 1, 3}.
    mask = torch.tensor([
        [False, True,  True,  False],
        [True,  True,  False, True ],
    ], dtype=torch.bool)

    # --- Library output ---
    out = mdm_loss(logits.float(), x0, mask)
    lib_loss = float(out.loss.detach().cpu().double().item())

    # --- Hand computation ---
    hand_per_sample = [
        _hand_loss_one_sample(logits[0], x0[0], mask[0]),
        _hand_loss_one_sample(logits[1], x0[1], mask[1]),
    ]
    hand_loss = sum(hand_per_sample) / len(hand_per_sample)

    assert math.isclose(lib_loss, hand_loss, rel_tol=0.0, abs_tol=1e-6), \
        f"mdm_loss disagrees with hand computation: lib={lib_loss:.10f} vs hand={hand_loss:.10f}"

    # Spot-check the per-sample mask counts surfaced for diagnostics
    assert out.n_masked_per_sample.tolist() == [2, 3], \
        f"n_masked_per_sample wrong: {out.n_masked_per_sample.tolist()}"
    assert out.n_masked_total == 5


def test_mdm_loss_zero_mask_sample_is_handled():
    """A sample with zero masked positions must contribute 0 to the loss
    (defensive — the standard sampler always uses n ≥ 1, but the function
    should not divide-by-zero on a malformed batch)."""
    B, L, V = 1, 4, 3
    logits = torch.zeros(B, L, V)
    x0 = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)

    out = mdm_loss(logits, x0, mask)
    assert float(out.loss) == pytest.approx(0.0, abs=1e-12)


def test_mdm_loss_mean_reduction_consistent_under_batch_doubling():
    """Replicating a sample twice in the batch should not change the mean loss."""
    B, L, V = 2, 4, 5
    torch.manual_seed(7)
    logits = torch.randn(B, L, V)
    x0 = torch.randint(0, V, (B, L))
    mask = torch.tensor([
        [False, True, True, False],
        [True, False, False, True],
    ], dtype=torch.bool)

    out1 = mdm_loss(logits, x0, mask)
    # Duplicate the batch
    out2 = mdm_loss(
        torch.cat([logits, logits], dim=0),
        torch.cat([x0, x0], dim=0),
        torch.cat([mask, mask], dim=0),
    )
    assert math.isclose(float(out1.loss), float(out2.loss), abs_tol=1e-6)
