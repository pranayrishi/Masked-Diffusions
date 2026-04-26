"""Entropy-filtered MDM training.

Imports baseline.src.* — the only difference from `baseline.src.train.run_training`
is that, between mask creation and the loss step, we score per-sample mask difficulty
by mean prediction entropy at the masked positions and drop samples whose score lies
outside the configured band (paper-notes §13.2).

To call:

    from entropy_filtered.src.train_filtered import run_filtered_training, FilteredTrainConfig
    cfg = FilteredTrainConfig(...)
    run_filtered_training(cfg)

Or via CLI:

    python -m entropy_filtered.src.train_filtered --config entropy_filtered/configs/lo_nae_sat_25_275_band.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Make `baseline` importable. We add the parent of this file's repo to sys.path so
# `from baseline.src.X import Y` resolves regardless of how this module is invoked.
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent.parent          # parent-of-entropy_filtered → repo root
sys.path.insert(0, str(REPO))

from baseline.src.data import LoNaeSatConfig, generate_dataset            # noqa: E402
from baseline.src.diffusion import apply_mask, mdm_loss, per_sample_mdm_loss, sample_mask_counts  # noqa: E402
from baseline.src.model import Transformer, TransformerConfig             # noqa: E402
from baseline.src.utils import (                                          # noqa: E402
    batch_indices_for_step,
    JsonlLogger, auto_device, capture_rng_states, CheckpointState,
    load_checkpoint, load_config, restore_rng_states, save_checkpoint,
    save_config, set_global_seed,
)

from .filter import EntropyFilterConfig, filter_batch                     # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FilteredTrainConfig:
    # data
    task: str
    data: dict
    # model
    model: dict
    # training
    seed: int
    num_iterations: int
    batch_size: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    eta_min_ratio: float
    log_every: int
    ckpt_every: int
    # entropy filter
    entropy_filter: dict                      # → EntropyFilterConfig fields
    # io
    output_dir: str
    # data sampling
    train_size: int
    train_sample_seed: int
    # early stopping (calibration runs use this; production runs leave disabled)
    early_stop: dict = field(default_factory=dict)


@dataclass
class EarlyStopConfig:
    """Plateau-based early stopping for calibration runs.

    Criterion (single supported flavor — `rolling_mean_relative`):
        rolling_mean(loss[step-window+1 .. step])
            vs. rolling_mean(loss[step-2*window+1 .. step-window])
        plateau iff |last - prev| / max(prev, eps) < tolerance

    Checks every `check_every` training steps starting at `min_step`. The check
    excludes steps where the optimizer was skipped (filter dropped the entire
    batch) — those steps had no gradient update so their "loss=0" is meaningless.
    """
    enabled: bool = False
    criterion: str = "rolling_mean_relative"
    tolerance: float = 0.005       # 0.5% relative
    window: int = 500              # in TRAINING steps (not logged steps)
    check_every: int = 500
    min_step: int = 1000


def _is_plateau(losses_by_step: dict[int, float], step: int, cfg: EarlyStopConfig) -> bool:
    """Return True iff the rolling-mean relative criterion fires at this step."""
    if not cfg.enabled:
        return False
    if cfg.criterion != "rolling_mean_relative":
        raise ValueError(f"unsupported criterion: {cfg.criterion}")
    if step < cfg.min_step:
        return False
    if step % cfg.check_every != 0:
        return False
    W = cfg.window
    last = [losses_by_step[s] for s in range(step - W + 1, step + 1) if s in losses_by_step]
    prev = [losses_by_step[s] for s in range(step - 2 * W + 1, step - W + 1) if s in losses_by_step]
    # Need at least half-coverage in each window to call it valid (skipped-optim
    # steps drop out, so we tolerate some sparsity).
    if len(last) < W // 2 or len(prev) < W // 2:
        return False
    last_mean = sum(last) / len(last)
    prev_mean = sum(prev) / len(prev)
    if prev_mean <= 0:
        return False
    rel = abs(last_mean - prev_mean) / prev_mean
    return rel < cfg.tolerance


def _setup_warnings_log(output_dir: str) -> None:
    """Append all UserWarnings (incl. torch.use_deterministic_algorithms warnings) to
    `output_dir/warnings.log`. Always-on for filtered training; the log is harmless
    when empty and provides the audit trail for eventually flipping
    `warn_only=True` → `warn_only=False`."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fh = open(f"{output_dir}/warnings.log", "a", buffering=1)

    def _show(message, category, filename, lineno, *_args, **_kwargs):
        fh.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")

    warnings.simplefilter("default")  # show each warning once per (filename, lineno)
    warnings.showwarning = _show


def _to_filtered_config(d: dict) -> FilteredTrainConfig:
    return FilteredTrainConfig(
        task=d["task"],
        data=d["data"],
        model=d["model"],
        seed=int(d["seed"]),
        num_iterations=int(d["num_iterations"]),
        batch_size=int(d["batch_size"]),
        lr=float(d["lr"]),
        weight_decay=float(d.get("weight_decay", 0.1)),
        beta1=float(d.get("beta1", 0.9)),
        beta2=float(d.get("beta2", 0.95)),
        grad_clip=float(d.get("grad_clip", 1.0)),
        eta_min_ratio=float(d.get("eta_min_ratio", 0.1)),
        log_every=int(d.get("log_every", 50)),
        ckpt_every=int(d.get("ckpt_every", 1000)),
        entropy_filter=d.get("entropy_filter", {"mode": "none"}),
        output_dir=str(d["output_dir"]),
        train_size=int(d.get("train_size", 100_000)),
        train_sample_seed=int(d.get("train_sample_seed", 123)),
        early_stop=d.get("early_stop", {}),
    )


def _build_model(cfg: FilteredTrainConfig, max_seq_len: int) -> Transformer:
    m = TransformerConfig(
        vocab_size=cfg.model["vocab_size"],
        hidden=cfg.model["hidden"],
        n_layers=cfg.model["n_layers"],
        n_heads=cfg.model["n_heads"],
        ff=cfg.model["ff"],
        max_seq_len=max_seq_len,
        causal=cfg.model.get("causal", False),
        pos_type=cfg.model.get("pos_type", "rope"),
        dropout=cfg.model.get("dropout", 0.0),
        weight_tie=cfg.model.get("weight_tie", False),
    )
    return Transformer(m)


# ---------------------------------------------------------------------------
# The single-step entropy-filtered MDM update
# ---------------------------------------------------------------------------

def filtered_train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    *,
    mask_token_id: int,
    pad_start: int | None,
    grad_clip: float,
    filter_cfg: EntropyFilterConfig,
    step: int,
) -> dict:
    """One filtered training step. Returns scalar metrics for logging.

    Order of operations:
      1. Sample mask counts and apply masks (same as baseline).
      2. Forward the model UNDER torch.no_grad() to compute per-sample entropy.
      3. Decide which samples to keep (filter_batch).
      4. Forward again WITH grad on only the kept samples; compute MDM loss; step.
    """
    model.train()
    B, L = x0.shape
    device = x0.device

    maskable_len = pad_start if pad_start is not None else L

    fixed_mask = None
    if pad_start is not None and pad_start < L:
        fixed_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        fixed_mask[:, pad_start:] = True

    n = sample_mask_counts(B, maskable_len, device)
    x_masked, mask = apply_mask(x0, n, mask_token_id=mask_token_id, fixed_mask=fixed_mask)

    # Step 1: scoring forward pass (no grad)
    with torch.no_grad():
        scoring_logits = model(x_masked)
    decision = filter_batch(
        scoring_logits, mask, filter_cfg,
        mask_token_id=mask_token_id, step=step,
    )

    # Mechanism diagnostic: compute per-sample MDM loss on ALL samples under no-grad
    # using the same scoring forward pass, then split by keep/drop. This gives the
    # apples-to-apples "is the filter targeting harder subproblems?" measurement.
    # Both subsets are evaluated with the SAME model state (the scoring pass), so any
    # difference in mean loss reflects the filter's selection, not parameter updates.
    with torch.no_grad():
        per_sample_losses = per_sample_mdm_loss(scoring_logits, x0, mask)  # (B,)
    keep_bool = decision.keep
    drop_bool = ~keep_bool
    if keep_bool.any():
        filter_loss_kept_mean = float(per_sample_losses[keep_bool].mean().cpu().item())
    else:
        filter_loss_kept_mean = None
    if drop_bool.any():
        filter_loss_dropped_mean = float(per_sample_losses[drop_bool].mean().cpu().item())
    else:
        filter_loss_dropped_mean = None

    # Step 2: discard dropped samples and run the loss-bearing forward pass
    keep_idx = decision.keep.nonzero(as_tuple=True)[0]
    if keep_idx.numel() == 0:
        # Defensive: filter dropped everything (shouldn't happen — `select_kept_samples`
        # falls back to keeping all in percentile_band; but absolute thresholds could
        # land outside the batch's range). Skip the optimizer step but log the event.
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "mean_n": float(n.float().mean().cpu().item()),
            "n_masked_total": int(mask.sum().item()),
            "filter_n_kept": 0,
            "filter_n_dropped": int(B),
            "filter_H_min": decision.H_min,
            "filter_H_max": decision.H_max,
            "filter_H_mean": decision.H_mean,
            "filter_loss_kept_mean": filter_loss_kept_mean,
            "filter_loss_dropped_mean": filter_loss_dropped_mean,
            "skipped_optim_step": 1,
        }

    x0_kept = x0[keep_idx]
    x_masked_kept = x_masked[keep_idx]
    mask_kept = mask[keep_idx]
    n_kept = n[keep_idx]

    logits_kept = model(x_masked_kept)
    out = mdm_loss(logits_kept, x0_kept, mask_kept)

    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(out.loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "mean_n": float(n_kept.float().mean().cpu().item()),
        "n_masked_total": int(out.n_masked_total),
        "filter_n_kept": decision.n_kept,
        "filter_n_dropped": decision.n_dropped,
        "filter_H_min": decision.H_min,
        "filter_H_max": decision.H_max,
        "filter_H_mean": decision.H_mean,
        "filter_loss_kept_mean": filter_loss_kept_mean,
        "filter_loss_dropped_mean": filter_loss_dropped_mean,
        "skipped_optim_step": 0,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _build_loaders_lo_nae_sat(cfg: FilteredTrainConfig) -> tuple[np.ndarray, LoNaeSatConfig]:
    data_cfg_obj = LoNaeSatConfig(**cfg.data)
    train, _ = generate_dataset(data_cfg_obj, num_samples=cfg.train_size, sample_seed=cfg.train_sample_seed)
    return train, data_cfg_obj


def run_filtered_training(cfg: FilteredTrainConfig, *, resume_from: str | None = None) -> None:
    set_global_seed(cfg.seed)
    device = auto_device()

    # Build entropy-filter config
    fcfg = EntropyFilterConfig(**cfg.entropy_filter) if cfg.entropy_filter else EntropyFilterConfig()
    fcfg.validate()

    # Build early-stop config (no-op when not provided)
    es_cfg = EarlyStopConfig(**cfg.early_stop) if cfg.early_stop else EarlyStopConfig()

    # Capture warnings (incl. non-deterministic-op UserWarnings from torch) to a
    # per-run log next to metrics.jsonl. Used during calibration to audit which
    # ops would abort under warn_only=False.
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    _setup_warnings_log(cfg.output_dir)

    print(f"[filtered] device={device}, mode={fcfg.mode}, warmup={fcfg.warmup_steps}, output_dir={cfg.output_dir}")
    if es_cfg.enabled:
        print(f"[filtered] early-stop ENABLED: criterion={es_cfg.criterion} "
              f"tolerance={es_cfg.tolerance} window={es_cfg.window} "
              f"check_every={es_cfg.check_every} min_step={es_cfg.min_step}")

    if cfg.task == "lo_nae_sat":
        train, data_cfg_obj = _build_loaders_lo_nae_sat(cfg)
        max_seq_len = data_cfg_obj.L
        mask_token_id = data_cfg_obj.mask_token_id
        pad_start = data_cfg_obj.L_data if data_cfg_obj.L_data < data_cfg_obj.L else None
    else:
        raise NotImplementedError(f"task={cfg.task} not yet supported")

    model = _build_model(cfg, max_seq_len=max_seq_len).to(device)
    n_params = model.num_parameters()
    print(f"[filtered] model: {n_params/1e6:.2f}M params, max_seq_len={max_seq_len}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_iterations, eta_min=cfg.lr * cfg.eta_min_ratio,
    )

    save_config(cfg.__dict__, f"{cfg.output_dir}/config.yaml")
    logger = JsonlLogger(f"{cfg.output_dir}/metrics.jsonl")

    start_step = 0
    if resume_from is not None:
        state = load_checkpoint(resume_from)
        model.load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            scheduler.load_state_dict(state.scheduler_state)
        restore_rng_states(state)
        start_step = state.step
        print(f"[filtered] resumed from step {start_step}")

    train_size = train.shape[0]
    t0 = time.monotonic()

    # Per-step loss history for plateau detection. Skipped-optim steps are excluded
    # (loss=0 there is not a real loss). Keep an in-memory dict; ~50K floats × 8B = 400KB.
    losses_by_step: dict[int, float] = {}
    final_step = cfg.num_iterations
    early_stopped = False

    for step in range(start_step + 1, cfg.num_iterations + 1):
        idx = batch_indices_for_step(cfg.seed, step, train_size, cfg.batch_size)
        batch = torch.as_tensor(train[idx], dtype=torch.long, device=device)

        metrics = filtered_train_step(
            model, optimizer, batch,
            mask_token_id=mask_token_id, pad_start=pad_start,
            grad_clip=cfg.grad_clip, filter_cfg=fcfg, step=step,
        )
        scheduler.step()

        if metrics.get("skipped_optim_step", 0) == 0:
            losses_by_step[step] = float(metrics["loss"])

        if step % cfg.log_every == 0 or step == 1:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.monotonic() - t0
            logger.log(step=step, lr=lr_now, elapsed=elapsed, **metrics)

        if step % cfg.ckpt_every == 0 or step == cfg.num_iterations:
            rng_states = capture_rng_states()
            ckpt = CheckpointState(
                step=step, seed=cfg.seed,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                rng_state_torch=rng_states["rng_state_torch"],
                rng_state_cuda=rng_states["rng_state_cuda"],
                rng_state_numpy=rng_states["rng_state_numpy"],
                rng_state_python=rng_states["rng_state_python"],
                extra={"task": cfg.task, "n_params": n_params, "filter_mode": fcfg.mode},
            )
            save_checkpoint(ckpt, f"{cfg.output_dir}/ckpt_step{step}.pt")

        if _is_plateau(losses_by_step, step, es_cfg):
            print(f"[filtered] plateau detected at step {step} "
                  f"(rolling {es_cfg.window}-step mean within {es_cfg.tolerance:.3%} relative); "
                  f"stopping early")
            # Save a final checkpoint at the plateau step (idempotent if already saved).
            if step % cfg.ckpt_every != 0:
                rng_states = capture_rng_states()
                ckpt = CheckpointState(
                    step=step, seed=cfg.seed,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    rng_state_torch=rng_states["rng_state_torch"],
                    rng_state_cuda=rng_states["rng_state_cuda"],
                    rng_state_numpy=rng_states["rng_state_numpy"],
                    rng_state_python=rng_states["rng_state_python"],
                    extra={"task": cfg.task, "n_params": n_params,
                           "filter_mode": fcfg.mode, "early_stopped": True},
                )
                save_checkpoint(ckpt, f"{cfg.output_dir}/ckpt_step{step}.pt")
            final_step = step
            early_stopped = True
            break

    logger.close()

    # Always emit plateau_step.txt — single integer, no other characters. The value is
    # the step at which training STOPPED, which is either the detected plateau step
    # (if early_stopped) or the configured cap (if no plateau was detected).
    plateau_path = Path(cfg.output_dir) / "plateau_step.txt"
    plateau_path.write_text(str(final_step))

    elapsed = time.monotonic() - t0
    status = "EARLY-STOPPED at plateau" if early_stopped else "completed full budget"
    print(f"[filtered] done in {elapsed:.1f}s ({final_step} iterations, {status})")
    print(f"[filtered] plateau_step.txt={final_step}")


def _smart_cast(v: str):
    """Best-effort cast for CLI overrides: int → float → bool → str.

    Identical to baseline.src.train._smart_cast — duplicated rather than imported
    so this module's CLI works regardless of import path.
    """
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description="Entropy-filtered MDM training.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--override", action="append", default=[],
                        help="Repeatable 'key=value' override (e.g., num_iterations=250 or "
                             "entropy_filter.warmup_steps=50). One level of nesting via dot "
                             "notation is supported.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    for kv in args.override:
        if "=" not in kv:
            raise SystemExit(f"--override expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if "." in k:
            head, tail = k.split(".", 1)
            cfg_dict[head][tail] = _smart_cast(v)
        else:
            cfg_dict[k] = _smart_cast(v)

    cfg = _to_filtered_config(cfg_dict)
    run_filtered_training(cfg, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
