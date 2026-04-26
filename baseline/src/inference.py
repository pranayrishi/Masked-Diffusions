"""Inference: vanilla MDM (Bernoulli per-position) + adaptive oracles.

Paper references (paper-notes):
  §2.6  vanilla: each masked position included in S with prob (α_s − α_t)/(1 − α_t).
        We honor this Bernoulli specification (fixing audit P1-1).
  §3.1  Top-Probability oracle:        c_i = max_j p_θ(x^i = j | x_t).
  §3.2  Top-Probability-Margin oracle: c_i = |p_θ(x^i = j_1 | x_t) − p_θ(x^i = j_2 | x_t)|.
  §3.3  K = (# masked) × (α_s − α_t)/(1 − α_t)  (deterministic Top-K for adaptive).
  §3.4  Noise injection on oracle SCORES (not logits):
          puzzles → Gumbel(0,1) × 0.5
          text    → Normal(0, σ²), σ = 0.001 (notebook default; tune)

All inference modes share a single function `run_inference` parameterized by
`strategy ∈ {"vanilla", "top_prob", "top_prob_margin"}` and `noise ∈ {"none", "gumbel", "gaussian"}`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _exclude_mask_token(logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Set the mask-token logit to -inf so it is never sampled. Returns a *new* tensor."""
    out = logits.clone()
    out[..., mask_token_id] = float("-inf")
    return out


def _add_score_noise(scores: torch.Tensor, noise: str, scale: float, generator) -> torch.Tensor:
    """Add Gumbel or Gaussian noise to a tensor of selection scores.

    paper §3.4: noise is added to *selection scores*, not to logits used for sampling.
    """
    if noise == "none" or scale <= 0.0:
        return scores
    if noise == "gumbel":
        u = torch.rand_like(scores)
        # Gumbel(0,1) = -log(-log(U))
        u = u.clamp(min=1e-12)
        g = -torch.log(-torch.log(u) + 1e-12)
        return scores + scale * g
    if noise == "gaussian":
        e = torch.randn(scores.shape, device=scores.device, dtype=scores.dtype, generator=generator)
        return scores + scale * e
    raise ValueError(f"unknown noise type: {noise}")


@torch.no_grad()
def run_inference(
    model,
    *,
    seq_len: int,
    mask_token_id: int,
    strategy: str = "vanilla",
    num_steps: int = 50,
    noise: str = "none",
    noise_scale: float = 0.0,
    fixed_tokens: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the MDM reverse process to produce a single completed sequence.

    Args:
        model:          a callable returning logits of shape (1, L, V) given an input (1, L).
        seq_len:        the sequence length L.
        mask_token_id:  integer id of the mask token.
        strategy:       "vanilla" (Bernoulli per-position), "top_prob", or "top_prob_margin".
        num_steps:      number of reverse diffusion steps. Paper §9.3 puzzles uses 50.
        noise:          "none", "gumbel", or "gaussian" — applied to oracle scores.
        noise_scale:    coefficient (e.g., 0.5 for Gumbel on puzzles, 0.001 for Gaussian on text).
        fixed_tokens:   (L,) LongTensor — values to inject at fixed positions before inference.
                        If supplied with `fixed_mask`, those positions are written into the
                        initial state and never modified during reverse diffusion.
        fixed_mask:     (L,) bool — True where the position is fixed (a clue), False otherwise.
        device:         device on which to run.
        generator:      optional torch.Generator for deterministic random choices.

    Returns:
        x_final: (L,) LongTensor — the completed sequence with no mask tokens.
    """
    device = torch.device(device)

    # Initialize: fully masked, then overlay fixed tokens (clues) at clue positions.
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if fixed_tokens is not None and fixed_mask is not None:
        x[0, fixed_mask] = fixed_tokens.to(device)[fixed_mask]
    elif (fixed_tokens is None) ^ (fixed_mask is None):
        raise ValueError("fixed_tokens and fixed_mask must be supplied together")

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for step in range(num_steps):
        t = float(timesteps[step].item())
        s = float(timesteps[step + 1].item())
        # Linear schedule: α_t = 1 - t, α_s = 1 - s, so the unmask probability is
        # (α_s - α_t) / (1 - α_t) = ((1-s) - (1-t)) / (1 - (1-t)) = (t - s) / t.
        # When t == 0 there are no masks left; we break before then.
        if t <= 0.0:
            break
        unmask_prob = (t - s) / t

        # Currently-masked positions (excluding fixed/clue positions)
        is_masked = (x[0] == mask_token_id)
        if fixed_mask is not None:
            is_masked = is_masked & ~fixed_mask.to(device)
        masked_pos = is_masked.nonzero(as_tuple=True)[0]
        num_masked = int(masked_pos.numel())
        if num_masked == 0:
            break

        # Compute model predictions, exclude mask token, softmax → probs over data tokens
        logits = model(x)                                          # (1, L, V)
        masked_logits = logits[0, masked_pos]                      # (num_masked, V)
        masked_logits = _exclude_mask_token(masked_logits, mask_token_id)
        probs = F.softmax(masked_logits, dim=-1)                   # (num_masked, V)

        if strategy == "vanilla":
            # Bernoulli per masked position (paper §2.6)
            keep = torch.bernoulli(
                torch.full((num_masked,), unmask_prob, device=device),
                generator=generator,
            ).bool()
            sel_local = keep.nonzero(as_tuple=True)[0]
        else:
            # Adaptive: K = num_masked × unmask_prob, then Top-K by oracle score
            K = max(1, int(round(num_masked * unmask_prob)))
            K = min(K, num_masked)

            if strategy == "top_prob":
                # c_i = max_j p_θ(x^i = j | x_t)
                scores = probs.max(dim=-1).values
            elif strategy == "top_prob_margin":
                # c_i = |p_θ(x^i=j1) − p_θ(x^i=j2)|; equivalently top1 − top2 since ≥ 0
                top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
                scores = (top2[:, 0] - top2[:, 1]) if top2.shape[-1] >= 2 else top2[:, 0]
            else:
                raise ValueError(f"unknown strategy: {strategy}")

            scores = _add_score_noise(scores, noise=noise, scale=noise_scale, generator=generator)
            sel_local = scores.topk(K).indices

        if sel_local.numel() == 0:
            continue

        # Sample tokens for the selected positions, batched
        sel_global = masked_pos[sel_local]
        sel_probs = probs[sel_local]                                # (K, V)
        # torch.multinomial doesn't accept generator on MPS but works on CPU/CUDA
        sampled = torch.multinomial(sel_probs, num_samples=1, generator=generator).squeeze(-1)
        x[0, sel_global] = sampled

    # Final cleanup: any remaining mask tokens get sampled greedily (without noise)
    is_masked = (x[0] == mask_token_id)
    if fixed_mask is not None:
        is_masked = is_masked & ~fixed_mask.to(device)
    if is_masked.any():
        masked_pos = is_masked.nonzero(as_tuple=True)[0]
        logits = model(x)
        masked_logits = _exclude_mask_token(logits[0, masked_pos], mask_token_id)
        probs = F.softmax(masked_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        x[0, masked_pos] = sampled

    # Sanity assertions: no mask token in output, all clues preserved
    assert (x[0] != mask_token_id).all(), "inference left mask tokens in the output"
    if fixed_tokens is not None and fixed_mask is not None:
        clue_positions = fixed_mask.to(device)
        assert torch.equal(x[0][clue_positions], fixed_tokens.to(device)[clue_positions]), \
            "inference modified clue positions"

    return x[0]
