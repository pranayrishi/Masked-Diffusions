"""Evaluation: L&O-NAE-SAT observation-token accuracy under vanilla and adaptive inference.

Paper §7.1 (Table 1) measures fraction of observation tokens predicted correctly when
the model is given a fully masked sequence and runs the full reverse process.
For now the baseline supports L&O-NAE-SAT; Sudoku evaluation (Tables 2, 5) is added
when the Sudoku data path is wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import LoNaeSatConfig
from .inference import run_inference


@dataclass
class EvalResult:
    strategy: str
    num_eval_samples: int
    obs_correct: int
    obs_total: int
    obs_accuracy: float


@torch.no_grad()
def evaluate_lo_nae_sat(
    model,
    test_sequences: np.ndarray,           # (num_eval, L)
    data_cfg: LoNaeSatConfig,
    *,
    strategy: str,
    num_steps: int = 50,
    noise: str = "none",
    noise_scale: float = 0.0,
    num_eval: int | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> EvalResult:
    """Run inference per test sample and measure observation-token accuracy.

    Padding positions are fixed (the model only generates the L&O part). Latents and
    observations are all initialized as masked.
    """
    model.eval()
    L = data_cfg.L
    L_data = data_cfg.L_data

    if num_eval is None:
        num_eval = test_sequences.shape[0]
    num_eval = min(num_eval, test_sequences.shape[0])

    # Padding positions are fixed and never modified during inference.
    pad_positions = torch.zeros(L, dtype=torch.bool)
    if L_data < L:
        pad_positions[L_data:] = True

    obs_correct = 0
    obs_total = 0
    for i in range(num_eval):
        x_true = test_sequences[i]
        fixed_tokens = torch.as_tensor(x_true, dtype=torch.long)

        generated = run_inference(
            model,
            seq_len=L,
            mask_token_id=data_cfg.mask_token_id,
            strategy=strategy,
            num_steps=num_steps,
            noise=noise,
            noise_scale=noise_scale,
            fixed_tokens=fixed_tokens,
            fixed_mask=pad_positions,
            device=device,
            generator=generator,
        )
        gen_obs = generated[data_cfg.N : L_data].cpu().numpy()
        true_obs = x_true[data_cfg.N : L_data]
        obs_correct += int((gen_obs == true_obs).sum())
        obs_total += len(true_obs)

    return EvalResult(
        strategy=strategy,
        num_eval_samples=num_eval,
        obs_correct=obs_correct,
        obs_total=obs_total,
        obs_accuracy=obs_correct / max(1, obs_total),
    )
