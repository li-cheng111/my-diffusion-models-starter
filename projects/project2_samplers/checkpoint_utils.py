"""Checkpoint compatibility helpers shared by Project 2 command-line tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess

import torch


def repository_commit(start: Path | None = None) -> str | None:
    """Return the enclosing Git commit without changing repository state."""

    cwd = str(start or Path(__file__).resolve().parent)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def model_state_from_checkpoint(
    checkpoint: Mapping,
    use_ema: bool = True,
    ema_decay: float | None = None,
) -> Mapping:
    """Return raw or EMA weights from legacy and multi-EMA Project 1 files."""

    if use_ema and "ema" in checkpoint:
        state = checkpoint["ema"]
        if isinstance(state, Mapping) and "models" in state:
            decays = [float(value) for value in state.get("decays", [])]
            models = state["models"]
            if len(decays) != len(models):
                raise ValueError("EMA decay list and model bank have different lengths")
            requested = max(decays) if ema_decay is None else float(ema_decay)
            matches = [i for i, value in enumerate(decays) if abs(value - requested) <= 1e-8]
            if not matches:
                choices = ", ".join(f"{value:g}" for value in decays) or "none"
                raise ValueError(f"EMA decay {requested:g} is unavailable; choices: {choices}")
            state = models[matches[0]]
        elif ema_decay is not None:
            raise ValueError("This checkpoint does not contain a selectable EMA bank")
        if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
            state = state["model"]
    else:
        if "model" not in checkpoint:
            raise KeyError("Checkpoint contains neither usable EMA nor model weights")
        state = checkpoint["model"]

    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict mapping, got {type(state).__name__}")
    return state


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Mapping,
    use_ema: bool = True,
    ema_decay: float | None = None,
) -> str:
    """Load weights and return the human-readable weight type."""

    model.load_state_dict(
        model_state_from_checkpoint(checkpoint, use_ema=use_ema, ema_decay=ema_decay)
    )
    if not use_ema or "ema" not in checkpoint:
        return "raw"
    ema = checkpoint["ema"]
    if isinstance(ema, Mapping) and "decays" in ema:
        selected = max(float(value) for value in ema["decays"]) if ema_decay is None else float(ema_decay)
        return f"EMA_{selected:g}"
    return "EMA"
