"""Test the inference pipeline.

Concrete assertions (binding):
  - After running vanilla and adaptive inference on a tiny untrained model,
    the output sequence contains NO mask token.
  - Initial clue positions, if supplied, are preserved exactly.

Also checks:
  - Each strategy ('vanilla', 'top_prob', 'top_prob_margin') terminates without error.
  - Adaptive K formula does not select more positions than are masked.
"""

from __future__ import annotations

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
