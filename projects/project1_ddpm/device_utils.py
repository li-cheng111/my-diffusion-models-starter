"""Small helpers for selecting CUDA, Moore Threads MUSA, or CPU."""

from __future__ import annotations

from typing import Any, Optional

import torch


def _load_musa_backend() -> Any:
    """Import torch_musa when installed so it registers ``torch.musa``."""

    try:
        import torch_musa  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_musa":
            raise
    return getattr(torch, "musa", None)


def get_device() -> torch.device:
    """Return the first available accelerator, preferring CUDA then MUSA."""

    musa = _load_musa_backend()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if musa is not None and musa.is_available():
        return torch.device("musa")
    return torch.device("cpu")


def accelerator_api(device: Optional[torch.device] = None) -> Any:
    """Return the backend module for a device, or ``None`` for CPU."""

    if device is None:
        device = get_device()
    if device.type == "cuda":
        return torch.cuda
    if device.type == "musa":
        musa = _load_musa_backend()
        if musa is None:
            raise RuntimeError("A MUSA device was selected but torch_musa is not importable")
        return musa
    return None


def seed_accelerator(seed: int, device: Optional[torch.device] = None) -> None:
    api = accelerator_api(device)
    if api is None:
        return
    seed_all = getattr(api, "manual_seed_all", None)
    if seed_all is not None:
        seed_all(seed)
    else:
        api.manual_seed(seed)


def capture_accelerator_rng(device: Optional[torch.device] = None) -> Any:
    api = accelerator_api(device)
    if api is None:
        return None
    get_state = getattr(api, "get_rng_state_all", None)
    return get_state() if get_state is not None else None


def restore_accelerator_rng(
    state: Any, device: Optional[torch.device] = None
) -> None:
    if state is None:
        return
    api = accelerator_api(device)
    if api is None:
        return
    set_state = getattr(api, "set_rng_state_all", None)
    if set_state is not None:
        set_state([item.cpu() for item in state])


def amp_components(device: torch.device) -> tuple[Any, Any]:
    """Return the autocast context and GradScaler classes for the device."""

    if device.type == "musa":
        musa = _load_musa_backend()
        if musa is None or not hasattr(musa, "amp"):
            raise RuntimeError("This torch_musa build does not expose MUSA AMP")
        return musa.amp.autocast, musa.amp.GradScaler
    return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler
