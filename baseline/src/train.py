"""MDM training loop with seed plumbing, checkpoint/resume, and JSONL logging.

Intentionally thin and library-style: the script entry-point in this file is
`run_training(cfg)`, called from `scripts/run_lo_nae_sat.sh` (or directly via
`python -m src.train --config <path>`).

The Phase 5 entropy filter will subclass / wrap `train_step` without changing
this file, by importing `train_step_factory` (or by composing it).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import LoNaeSatConfig, generate_dataset
from .diffusion import (
    apply_mask,
    mdm_loss,
    sample_mask_counts,
)
from .model import Transformer, TransformerConfig
from .utils import (
    JsonlLogger,
    auto_device,
    capture_rng_states,
    CheckpointState,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # data
    task: str                           # "lo_nae_sat" (Sudoku added in Phase 4B)
    data: dict                          # task-specific data config (e.g., LoNaeSatConfig fields)

    # model
    model: dict                         # TransformerConfig fields

    # training
    seed: int
    num_iterations: int
    batch_size: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    eta_min_ratio: float                # min_lr = lr × eta_min_ratio
    log_every: int
    ckpt_every: int

    # io
    output_dir: str

    # data sampling
    train_size: int                     # how many data sequences to generate up front
    train_sample_seed: int              # seed for data latents (independent of model seed)


def _build_loaders_lo_nae_sat(cfg: TrainConfig) -> tuple[np.ndarray, LoNaeSatConfig]:
    data_cfg_obj = LoNaeSatConfig(**cfg.data)
    train, _ = generate_dataset(data_cfg_obj, num_samples=cfg.train_size, sample_seed=cfg.train_sample_seed)
    return train, data_cfg_obj


def _build_model(cfg: TrainConfig, max_seq_len: int) -> Transformer:
    m_cfg = TransformerConfig(
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
    return Transformer(m_cfg)


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    *,
    mask_token_id: int,
    pad_start: int | None,
    grad_clip: float,
) -> dict:
    """Run a single MDM training step. Returns a dict of scalar metrics for logging."""
    model.train()
    B, L = x0.shape
    device = x0.device

    maskable_len = pad_start if pad_start is not None else L

    # Optional fixed-mask: padding positions never masked
    fixed_mask = None
    if pad_start is not None and pad_start < L:
        fixed_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        fixed_mask[:, pad_start:] = True

    # Sample n ~ Uniform{1, ..., maskable_len} per sample (paper §12.2)
    n = sample_mask_counts(B, maskable_len, device)
    x_masked, mask = apply_mask(x0, n, mask_token_id=mask_token_id, fixed_mask=fixed_mask)

    logits = model(x_masked)
    out = mdm_loss(logits, x0, mask)

    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(out.loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "mean_n": float(n.float().mean().cpu().item()),
        "n_masked_total": int(out.n_masked_total),
    }


def run_training(cfg: TrainConfig, *, resume_from: str | None = None) -> None:
    """Top-level training entry point. Idempotent w.r.t. checkpoints in `output_dir`."""
    set_global_seed(cfg.seed)
    device = auto_device()
    print(f"[train] device={device}, task={cfg.task}, seed={cfg.seed}, output_dir={cfg.output_dir}")

    # --- Build data ---
    if cfg.task == "lo_nae_sat":
        train, data_cfg_obj = _build_loaders_lo_nae_sat(cfg)
        max_seq_len = data_cfg_obj.L
        mask_token_id = data_cfg_obj.mask_token_id
        pad_start = data_cfg_obj.L_data if data_cfg_obj.L_data < data_cfg_obj.L else None
    else:
        raise NotImplementedError(f"task={cfg.task} not yet supported in baseline/. Sudoku coming next.")

    # --- Build model ---
    model = _build_model(cfg, max_seq_len=max_seq_len).to(device)
    n_params = model.num_parameters()
    print(f"[train] model: {n_params/1e6:.2f}M params, max_seq_len={max_seq_len}")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_iterations, eta_min=cfg.lr * cfg.eta_min_ratio
    )

    # --- Logging + checkpointing ---
    save_config(cfg.__dict__, f"{cfg.output_dir}/config.yaml")
    logger = JsonlLogger(f"{cfg.output_dir}/metrics.jsonl")

    start_step = 0
    if resume_from is not None:
        state = load_checkpoint(resume_from)
        model.load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            scheduler.load_state_dict(state.scheduler_state)
        # RNG restoration is best-effort; numpy state is required for data sampling determinism
        try:
            torch.set_rng_state(state.rng_state_torch)
            np.random.set_state(state.rng_state_numpy)
        except Exception as exc:
            print(f"[train] warning: failed to restore RNG state cleanly: {exc}")
        start_step = state.step
        print(f"[train] resumed from step {start_step}")

    # --- Training loop ---
    train_size = train.shape[0]
    rng = np.random.default_rng(cfg.seed + 1)   # for index sampling (independent of model RNG)
    t0 = time.monotonic()

    for step in range(start_step + 1, cfg.num_iterations + 1):
        idx = rng.integers(0, train_size, size=cfg.batch_size)
        batch = torch.as_tensor(train[idx], dtype=torch.long, device=device)

        metrics = train_step(
            model, optimizer, batch,
            mask_token_id=mask_token_id,
            pad_start=pad_start,
            grad_clip=cfg.grad_clip,
        )
        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.monotonic() - t0
            logger.log(step=step, lr=lr_now, elapsed=elapsed, **metrics)

        if step % cfg.ckpt_every == 0 or step == cfg.num_iterations:
            rng_states = capture_rng_states()
            ckpt = CheckpointState(
                step=step,
                seed=cfg.seed,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                rng_state_torch=rng_states["rng_state_torch"],
                rng_state_cuda=rng_states["rng_state_cuda"],
                rng_state_numpy=rng_states["rng_state_numpy"],
                rng_state_python=rng_states["rng_state_python"],
                extra={"task": cfg.task, "n_params": n_params},
            )
            save_checkpoint(ckpt, f"{cfg.output_dir}/ckpt_step{step}.pt")

    logger.close()
    elapsed = time.monotonic() - t0
    print(f"[train] done in {elapsed:.1f}s ({cfg.num_iterations} iterations)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _to_train_config(d: dict) -> TrainConfig:
    """Turn a YAML-loaded dict into a TrainConfig. Fail loudly if keys are missing."""
    return TrainConfig(
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
        output_dir=str(d["output_dir"]),
        train_size=int(d.get("train_size", 100_000)),
        train_sample_seed=int(d.get("train_sample_seed", 123)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an MDM on a configured task.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Optional 'key=value' overrides (e.g., output_dir=runs/test).")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    for kv in args.override:
        if "=" not in kv:
            raise SystemExit(f"--override expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        # Allow nested overrides like data.N=10
        if "." in k:
            head, tail = k.split(".", 1)
            cfg_dict[head][tail] = _smart_cast(v)
        else:
            cfg_dict[k] = _smart_cast(v)

    cfg = _to_train_config(cfg_dict)
    run_training(cfg, resume_from=args.resume)
    return 0


def _smart_cast(v: str):
    """Best-effort cast for CLI overrides: int → float → bool → str."""
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


if __name__ == "__main__":
    raise SystemExit(main())
