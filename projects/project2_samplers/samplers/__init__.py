"""Project 2 sampler package."""

from .base import BaseSampler
from .ddim import DDIMSampler
from .ddim_inversion import DDIMInverter, ddim_invert
from .ddpm import DDPMSampler
from .dpm_solver import DPMSolver2Sampler
from .euler import EulerSampler

__all__ = [
    "BaseSampler",
    "DDIMSampler",
    "DDIMInverter",
    "DDPMSampler",
    "DPMSolver2Sampler",
    "EulerSampler",
    "ddim_invert",
    "get_sampler",
]


def get_sampler(name: str, model, schedule, device="cuda", **kwargs):
    normalized = name.lower().replace("_", "-")
    if normalized == "ddpm":
        return DDPMSampler(
            model, schedule, device,
            clip_denoised=kwargs.get("clip_denoised", True),
        )
    if normalized == "ddim":
        return DDIMSampler(
            model,
            schedule,
            device,
            eta=kwargs.get("eta", 0.0),
            timestep_strategy=kwargs.get("timestep_strategy", "linear"),
            clip_denoised=kwargs.get("clip_denoised", True),
        )
    if normalized == "dpm-solver":
        return DPMSolver2Sampler(
            model,
            schedule,
            device,
            timestep_strategy=kwargs.get("timestep_strategy", "lambda"),
            clip_denoised=kwargs.get("clip_denoised", True),
        )
    if normalized == "euler":
        return EulerSampler(model, schedule, device, ancestral=False)
    if normalized in {"euler-a", "eulera"}:
        return EulerSampler(model, schedule, device, ancestral=True)
    raise ValueError(f"Unknown sampler: {name}")
