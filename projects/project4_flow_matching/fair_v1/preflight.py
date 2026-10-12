"""Fast structural, objective, CFG and numerical checks before long training."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from projects.project4_flow_matching.fair_v1.evaluate import integrate_velocity, _velocity
from projects.project4_flow_matching.fair_v1.models import ConditionalProject1UNet, build_model, initial_state_sha256
from projects.project4_flow_matching.fair_v1.train import load_config

BASE = Path(__file__).parent / "configs"
configs = {name: load_config(BASE / name) for name in (
    "ddpm_unet.yaml", "fm_unet.yaml", "fm_logit_normal.yaml", "fm_conditional.yaml", "fm_dit.yaml")}
models = {}
for name in ("ddpm_unet.yaml", "fm_unet.yaml", "fm_logit_normal.yaml"):
    torch.manual_seed(44)
    models[name] = build_model(configs[name]["model"])
hashes = {name: initial_state_sha256(model) for name, model in models.items()}
assert len(set(hashes.values())) == 1, f"paired U-Net initial states differ: {hashes}"
base_state = models["fm_unet.yaml"].state_dict()
torch.manual_seed(44)
conditional = build_model(configs["fm_conditional.yaml"]["model"])
conditional_backbone = {k: v for k, v in conditional.state_dict().items() if not k.startswith("class_embedding.")}
assert base_state.keys() == conditional_backbone.keys()
assert all(torch.equal(base_state[k], conditional_backbone[k]) for k in base_state), "conditional U-Net backbone initialization differs"

for name in ("fm_unet.yaml", "fm_conditional.yaml", "fm_dit.yaml"):
    model = models.get(name)
    if model is None:
        torch.manual_seed(44)
        model = build_model(configs[name]["model"])
    model = model.cuda().eval()
    x = torch.randn(2, 3, 32, 32, device="cuda")
    t = torch.tensor([0.0, 0.437], device="cuda")
    labels = torch.tensor([0, 10 if name == "fm_conditional.yaml" else 3], device="cuda")
    if isinstance(model, ConditionalProject1UNet):
        pred = model(x, t * 999.0, labels)
        cond = _velocity(model, x, t, labels.clamp_max(9), "conditional", 1.0, 10)
        cfg1 = _velocity(model, x, t, labels.clamp_max(9), "cfg", 1.0, 10)
        uncond = _velocity(model, x, t, torch.zeros_like(labels), "unconditional", 0.0, 10)
        cfg0 = _velocity(model, x, t, torch.zeros_like(labels), "cfg", 0.0, 10)
        assert torch.allclose(cond, cfg1, atol=5e-4, rtol=5e-4)
        assert torch.allclose(uncond, cfg0, atol=5e-4, rtol=5e-4)
    else:
        from projects.project4_flow_matching.fair_v1.models import predict
        pred = predict(model, x, t, torch.full((2,), 10, device="cuda", dtype=torch.long))
    assert pred.shape == x.shape and torch.isfinite(pred).all()
    del model, x, t, labels, pred
    torch.cuda.empty_cache()

for solver, budget in (("euler", 512), ("heun", 512)):
    calls = [0]
    def velocity(x, t):
        calls[0] += 1
        return x
    initial = torch.ones(1, 1)
    result = integrate_velocity(initial, velocity, budget, solver)
    assert calls[0] == budget
    assert abs(result.item() - torch.e) < (0.0035 if solver == "euler" else 1e-5), (solver, result.item())
print("preflight passed: paired initialization, model forwards, CFG endpoints, exact NFE, Euler/Heun convergence")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)} peak_mem_gb={torch.cuda.max_memory_allocated()/1024**3:.2f}")
