"""Utilities: config loading, seeding, JSONL logging, checkpointing.

Design goals (paper-notes §13.1, code_audit P2-3 / P2-4 / P2-6):
  - Every script accepts a `--seed`; we set torch + numpy + python random + cuda.
  - Logs are written as JSON-lines (one row per logging step) so that downstream
    aggregation in Phase 8 can be done with pandas/jq without parsing prints.
  - Checkpoints save model + optimizer + scheduler + RNG state + step count, so a
    Bouchet preemption (paper-notes §scavenge_gpu) can resume cleanly.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed numpy, python random, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

class JsonlLogger:
    """Simple append-only JSONL logger with one row per call.

    Each row is a JSON object with the keys passed in. We always include
    `step`, `wall_time`, and a monotonically increasing `event_idx`.
    """

    def __init__(self, path: str):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.event_idx = 0
        self._t0 = time.monotonic()
        # Open append; create if missing
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def log(self, step: int, **fields) -> None:
        row = {"step": int(step), "wall_time": time.monotonic() - self._t0, "event_idx": self.event_idx, **fields}
        self._fh.write(json.dumps(row) + "\n")
        self.event_idx += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class CheckpointState:
    step: int
    seed: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    rng_state_torch: torch.Tensor
    rng_state_cuda: list | None
    rng_state_numpy: dict
    rng_state_python: tuple
    extra: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(state: CheckpointState, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": state.step,
        "seed": state.seed,
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "rng_state_torch": state.rng_state_torch,
        "rng_state_cuda": state.rng_state_cuda,
        "rng_state_numpy": state.rng_state_numpy,
        "rng_state_python": state.rng_state_python,
        "extra": state.extra,
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic


def load_checkpoint(path: str) -> CheckpointState:
    payload = torch.load(path, map_location="cpu")
    return CheckpointState(
        step=payload["step"],
        seed=payload["seed"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload.get("scheduler_state"),
        rng_state_torch=payload["rng_state_torch"],
        rng_state_cuda=payload.get("rng_state_cuda"),
        rng_state_numpy=payload["rng_state_numpy"],
        rng_state_python=payload["rng_state_python"],
        extra=payload.get("extra", {}),
    )


def capture_rng_states() -> dict[str, Any]:
    return {
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_state_numpy": np.random.get_state(legacy=False),
        "rng_state_python": random.getstate(),
    }


def restore_rng_states(state: CheckpointState) -> None:
    torch.set_rng_state(state.rng_state_torch)
    if state.rng_state_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.rng_state_cuda)
    np.random.set_state(state.rng_state_numpy)
    random.setstate(state.rng_state_python)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def auto_device() -> torch.device:
    """Return cuda if available else cpu. Bouchet jobs override via env if needed."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
