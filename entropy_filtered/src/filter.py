"""Entropy filter: score per-sample mask difficulty by mean prediction entropy at masked
positions, then drop samples whose score lies outside a productive band.

Five modes (paper-notes §13.2 / professor's spec / user-confirmed 2026-04-26):

  - "none"             : no filtering. The control. Identical to baseline training.
  - "bottom"           : drop samples with H̄ <= H_low.   Tests "remove wasted gradient
                          on trivial subproblems."
  - "top"              : drop samples with H̄ >= H_high.  Tests the paper's theoretical
                          intractable-region claim (Conjecture B.13).
  - "band"             : drop both ends, ABSOLUTE thresholds [H_low, H_high].
  - "percentile_band"  : drop both ends, BATCH-RELATIVE percentiles. Self-calibrating to
                          the model's current state.

User-confirmed defaults:
  warmup_steps = 500          # filter is OFF for the first 500 steps. The model is too
                                random at step 0 for entropy to be informative; we wait
                                until the loss has dropped below trivial before filtering.
  reduction    = "mean"       # H̄ = (1/|M_b|) Σ_{i in M_b} H(p_θ(x^i | x_t)) per sample.
                                "median" supported as an alternative (more robust to outliers).
  use_softmax_with_mask = True # exclude the mask token from the categorical when computing
                                entropy: a model that places probability on the mask token
                                is usually a model that hasn't separated mask from data yet.

Implementation cost: one extra no-grad forward pass to score masks. The extra forward
pass uses the SAME masked input that's about to be used for the loss step, so we just
compute it once with no_grad, then re-use it (with grad) for the actual loss. In effect
the "extra" cost is just the gradient computation we save by dropping samples — net
performance roughly the same as baseline at fixed wall-clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EntropyFilterConfig:
    mode: str = "none"                  # "none" | "bottom" | "top" | "band" | "percentile_band"
    warmup_steps: int = 500             # filter disabled for steps 1..warmup_steps
    reduction: str = "mean"             # "mean" or "median" over masked positions per sample
    # Absolute thresholds (used by "bottom", "top", "band")
    H_low: float = 0.0                  # in nats; samples with H̄ <= H_low are dropped (when applicable)
    H_high: float = 1e9                 # in nats; samples with H̄ >= H_high are dropped (when applicable)
    # Percentile thresholds (used by "percentile_band")
    pct_low: float = 0.25               # drop the lowest pct_low fraction of the batch
    pct_high: float = 0.75              # drop the highest (1 - pct_high) fraction of the batch
    # Implementation toggles
    use_softmax_excluding_mask_token: bool = True
    eps: float = 1e-12                  # numerical stability in entropy

    def validate(self) -> None:
        if self.mode not in {"none", "bottom", "top", "band", "percentile_band"}:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.reduction not in {"mean", "median"}:
            raise ValueError(f"unknown reduction: {self.reduction}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative: {self.warmup_steps}")
        if self.H_low > self.H_high:
            raise ValueError(f"H_low {self.H_low} > H_high {self.H_high}")
        if not (0.0 <= self.pct_low <= self.pct_high <= 1.0):
            raise ValueError(f"need 0 <= pct_low <= pct_high <= 1, got {self.pct_low}, {self.pct_high}")


# ---------------------------------------------------------------------------
# Per-sample entropy scoring (no-grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_sample_entropy(
    logits: torch.Tensor,        # (B, L, V) — model outputs at all positions
    mask: torch.Tensor,          # (B, L) bool — True at MASKED positions
    *,
    mask_token_id: int,
    reduction: str = "mean",
    use_softmax_excluding_mask_token: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the per-sample mean (or median) Shannon entropy of p_θ(x^i | x_t)
    at the masked positions.

    Returns a (B,) tensor of nats. Samples with zero masked positions get entropy 0.0
    (they cannot contribute to the loss anyway).
    """
    B, L, V = logits.shape

    # Compute the categorical distribution. If we are excluding the mask token from the
    # support, we slice it out *before* the softmax so the remaining V-1 logits get
    # normalized cleanly. This avoids 0 × -inf = nan which poisons subsequent sums.
    if use_softmax_excluding_mask_token:
        # Build the index list of non-mask token ids
        keep_ids = [i for i in range(V) if i != mask_token_id]
        logits_eff = logits[..., keep_ids]                        # (B, L, V-1)
    else:
        logits_eff = logits

    log_probs = F.log_softmax(logits_eff, dim=-1)
    probs = log_probs.exp()
    # Numerically safe entropy: where probs == 0, the contribution to entropy is 0
    # (limit of x log x as x → 0+). xlogy(p, p) does this correctly.
    plogp = torch.special.xlogy(probs, probs)
    H_per_position = -(plogp.sum(dim=-1))                         # (B, L), nats
    H_per_position = torch.nan_to_num(H_per_position, nan=0.0, posinf=0.0, neginf=0.0)

    H_per_position = H_per_position * mask.float()
    n_masked = mask.sum(dim=-1)

    if reduction == "mean":
        safe_n = n_masked.clamp(min=1).float()
        H_per_sample = H_per_position.sum(dim=-1) / safe_n
        H_per_sample = H_per_sample * (n_masked > 0).float()
    elif reduction == "median":
        # For each sample, take the median of entropies at masked positions only.
        # Implemented per-sample because masked-position counts vary.
        H_per_sample = torch.zeros(B, device=logits.device, dtype=H_per_position.dtype)
        for b in range(B):
            n_b = int(n_masked[b].item())
            if n_b == 0:
                continue
            row = H_per_position[b][mask[b]]
            H_per_sample[b] = row.median()
    else:
        raise ValueError(f"unknown reduction: {reduction}")

    return H_per_sample


# ---------------------------------------------------------------------------
# Mask-keeping decision (returns a (B,) bool tensor: True = keep this sample)
# ---------------------------------------------------------------------------

def select_kept_samples(
    H: torch.Tensor,                    # (B,) per-sample entropy scores in nats
    cfg: EntropyFilterConfig,
    *,
    step: int,
) -> torch.Tensor:
    """Return a (B,) bool mask of which samples to keep for the loss step.

    Honors the warmup window: for steps 1..warmup_steps inclusive, ALL samples are kept
    regardless of mode (the filter is "off" until the model has trained for `warmup_steps`).
    """
    cfg.validate()
    B = H.shape[0]

    # Always keep everything during warmup
    if step <= cfg.warmup_steps or cfg.mode == "none":
        return torch.ones(B, dtype=torch.bool, device=H.device)

    if cfg.mode == "bottom":
        return H > cfg.H_low

    if cfg.mode == "top":
        return H < cfg.H_high

    if cfg.mode == "band":
        return (H > cfg.H_low) & (H < cfg.H_high)

    if cfg.mode == "percentile_band":
        # Compute batch-relative percentile thresholds. With small batches these are noisy
        # but still self-calibrating to model state.
        sorted_H, _ = torch.sort(H)
        idx_low = int(cfg.pct_low * (B - 1))
        idx_high = int(cfg.pct_high * (B - 1))
        threshold_low = sorted_H[idx_low].item()
        threshold_high = sorted_H[idx_high].item()
        keep = (H >= threshold_low) & (H <= threshold_high)
        if not keep.any():           # never let the filter wipe the entire batch
            keep = torch.ones(B, dtype=torch.bool, device=H.device)
        return keep

    raise ValueError(f"unhandled mode: {cfg.mode}")


# ---------------------------------------------------------------------------
# Convenience: produce a "kept-samples" mask for downstream loss
# ---------------------------------------------------------------------------

@dataclass
class FilterDecision:
    keep: torch.Tensor                  # (B,) bool — which samples survive
    H: torch.Tensor                     # (B,) per-sample entropy scores
    n_kept: int                         # int(keep.sum())
    n_dropped: int                      # B - n_kept
    # Diagnostics for the JSONL log
    H_min: float
    H_max: float
    H_mean: float


def filter_batch(
    logits: torch.Tensor,
    mask: torch.Tensor,
    cfg: EntropyFilterConfig,
    *,
    mask_token_id: int,
    step: int,
) -> FilterDecision:
    """End-to-end: score the batch, decide which samples to keep, return diagnostics."""
    H = per_sample_entropy(
        logits, mask,
        mask_token_id=mask_token_id,
        reduction=cfg.reduction,
        use_softmax_excluding_mask_token=cfg.use_softmax_excluding_mask_token,
        eps=cfg.eps,
    )
    keep = select_kept_samples(H, cfg, step=step)
    return FilterDecision(
        keep=keep,
        H=H,
        n_kept=int(keep.sum().item()),
        n_dropped=int((~keep).sum().item()),
        H_min=float(H.min().item()),
        H_max=float(H.max().item()),
        H_mean=float(H.mean().item()),
    )
