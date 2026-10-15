"""Device helpers for CUDA, MUSA and CPU Project 5 runs."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Prefer an available MUSA device, then CUDA, then CPU."""
    try:
        import torch_musa  # noqa: F401  pylint: disable=unused-import
    except (ImportError, OSError):
        pass
    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        return torch.device("musa")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_device(seed: int) -> None:
    """Seed host and all available accelerator RNGs."""
    try:
        import torch_musa  # noqa: F401  pylint: disable=unused-import
    except (ImportError, OSError):
        pass
    torch.manual_seed(seed)
    for backend_name in ("cuda", "musa"):
        backend = getattr(torch, backend_name, None)
        if backend is not None and backend.is_available():
            seed_all = getattr(backend, "manual_seed_all", None)
            if seed_all is not None:
                seed_all(seed)
            else:
                backend.manual_seed(seed)


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Create a reproducible generator on the requested device."""
    return torch.Generator(device=device).manual_seed(int(seed))
