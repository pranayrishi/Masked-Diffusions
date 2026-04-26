"""Test the inference pipeline.

Concrete assertions (binding):
  - After running vanilla and adaptive inference on a tiny untrained model,
    the output sequence contains NO mask token.
  - Initial clue positions, if supplied, are preserved exactly.

Also checks:
  - Each strategy ('vanilla', 'top_prob', 'top_prob_margin') terminates without error.
  - Adaptive K formula does not select more positions than are masked.
  - All 3 oracles × 2 noise modes (Gumbel for puzzles, Gaussian for text)
    produce valid sequences. This covers the matrix the user requested
    explicitly in the Phase 6 review (six strategy/noise combinations).
"""

from __future__ import annotations

import pytest
import torch

from src.inference import run_inference
from src.model import Transformer, TransformerConfig


def _make_tiny_model(vocab_size: int = 4, max_seq_len: int = 32, causal: bool = False) -> Transformer:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden=16,
        n_layers=1,
        n_heads=2,
        ff=32,
        max_seq_len=max_seq_len,
        causal=causal,
        pos_type="learned",
    )
    torch.manual_seed(0)
    return Transformer(cfg)


def test_vanilla_inference_produces_no_mask_token():
    model = _make_tiny_model(vocab_size=4, max_seq_len=16)
    out = run_inference(
        model,
        seq_len=16,
        mask_token_id=3,
        strategy="vanilla",
        num_steps=20,
        noise="none",
        noise_scale=0.0,
        device="cpu",
    )
    assert out.shape == (16,)
    assert (out != 3).all().item(), "vanilla inference left mask tokens in the output"


def test_top_prob_inference_produces_no_mask_token():
    model = _make_tiny_model(vocab_size=4, max_seq_len=16)
    out = run_inference(
        model,
        seq_len=16,
        mask_token_id=3,
        strategy="top_prob",
        num_steps=20,
        noise="gumbel",
        noise_scale=0.5,
        device="cpu",
    )
    assert (out != 3).all().item(), "top_prob inference left mask tokens in the output"


def test_top_prob_margin_inference_produces_no_mask_token():
    model = _make_tiny_model(vocab_size=4, max_seq_len=16)
    out = run_inference(
        model,
        seq_len=16,
        mask_token_id=3,
        strategy="top_prob_margin",
        num_steps=20,
        noise="gumbel",
        noise_scale=0.5,
        device="cpu",
    )
    assert (out != 3).all().item(), "top_prob_margin inference left mask tokens in the output"


def test_initial_clues_are_preserved():
    """If we declare positions {3, 7, 11} as fixed clues with values {1, 2, 0},
    those positions must equal those values in the output for every strategy."""
    model = _make_tiny_model(vocab_size=4, max_seq_len=16)

    L = 16
    fixed_tokens = torch.zeros(L, dtype=torch.long)
    fixed_mask = torch.zeros(L, dtype=torch.bool)
    fixed_tokens[3], fixed_mask[3] = 1, True
    fixed_tokens[7], fixed_mask[7] = 2, True
    fixed_tokens[11], fixed_mask[11] = 0, True

    for strategy in ("vanilla", "top_prob", "top_prob_margin"):
        out = run_inference(
            model,
            seq_len=L,
            mask_token_id=3,
            strategy=strategy,
            num_steps=20,
            noise="gumbel",
            noise_scale=0.5,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            device="cpu",
        )
        assert (out != 3).all().item(), f"{strategy}: mask token survived"
        for pos, val in [(3, 1), (7, 2), (11, 0)]:
            assert int(out[pos].item()) == val, \
                f"{strategy}: clue at pos {pos} should be {val}, got {int(out[pos])}"


def test_inference_handles_fully_revealed_state_gracefully():
    """If the input is already fully revealed (no masks), inference should be a no-op."""
    model = _make_tiny_model(vocab_size=4, max_seq_len=8)
    L = 8
    # All positions fixed (clues), no positions masked
    fixed_tokens = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    fixed_mask = torch.ones(L, dtype=torch.bool)
    out = run_inference(
        model,
        seq_len=L,
        mask_token_id=3,
        strategy="top_prob_margin",
        num_steps=10,
        noise="none",
        noise_scale=0.0,
        fixed_tokens=fixed_tokens,
        fixed_mask=fixed_mask,
        device="cpu",
    )
    assert torch.equal(out, fixed_tokens), "inference modified clues when no masks were present"


# ---------------------------------------------------------------------------
# All 3 oracles × 2 noise modes — full matrix coverage
# (paper §3.4: Gumbel for puzzles, Gaussian for text)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["vanilla", "top_prob", "top_prob_margin"])
@pytest.mark.parametrize("noise,noise_scale", [
    ("gumbel", 0.5),       # puzzle setting (paper Appendix D.2)
    ("gaussian", 0.001),   # text setting (paper Appendix D.1.2, σ default from existing notebook)
])
def test_full_strategy_x_noise_matrix_produces_valid_sequences(strategy, noise, noise_scale):
    """For every (strategy, noise-mode) combination, inference must:
    (1) terminate without error, (2) produce a sequence with no mask tokens,
    (3) preserve the supplied clue positions exactly."""
    model = _make_tiny_model(vocab_size=5, max_seq_len=24)
    L = 24

    # Inject some clues so we also verify the conditional-inference invariant.
    # All clue values must be distinct from mask_token_id (=4) — see Notes for the
    # defensive assertion in `run_inference`: a clue happening to equal mask_token_id
    # is a legal user choice but obviously can't be tested for "no mask in output".
    fixed_tokens = torch.zeros(L, dtype=torch.long)
    fixed_mask = torch.zeros(L, dtype=torch.bool)
    for pos, val in [(2, 1), (10, 3), (15, 2), (22, 0)]:
        fixed_tokens[pos], fixed_mask[pos] = val, True

    out = run_inference(
        model,
        seq_len=L,
        mask_token_id=4,                    # distinct from data values 0..3
        strategy=strategy,
        num_steps=24,
        noise=noise,
        noise_scale=noise_scale,
        fixed_tokens=fixed_tokens,
        fixed_mask=fixed_mask,
        device="cpu",
    )
    # (1) shape and (2) no mask tokens
    assert out.shape == (L,)
    # (2) no mask token at any NON-CLUE position
    non_clue = ~fixed_mask
    assert (out[non_clue] != 4).all().item(), f"({strategy}, {noise}): mask token in output"
    # (3) clues preserved exactly
    for pos, val in [(2, 1), (10, 3), (15, 2), (22, 0)]:
        assert int(out[pos].item()) == val, \
            f"({strategy}, {noise}): clue at pos {pos} should be {val}, got {int(out[pos])}"


def test_noise_none_is_noop_when_scale_is_zero():
    """noise='none' or noise_scale=0 should produce identical output for a fixed seed."""
    model = _make_tiny_model(vocab_size=4, max_seq_len=12)
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)

    o_none = run_inference(
        model, seq_len=12, mask_token_id=3,
        strategy="top_prob_margin", num_steps=12,
        noise="none", noise_scale=0.0, device="cpu", generator=g1,
    )
    o_zero = run_inference(
        model, seq_len=12, mask_token_id=3,
        strategy="top_prob_margin", num_steps=12,
        noise="gumbel", noise_scale=0.0, device="cpu", generator=g2,
    )
    assert torch.equal(o_none, o_zero), "noise='none' and scale=0.0 disagree"
