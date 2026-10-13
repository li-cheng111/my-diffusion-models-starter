"""Rebuild the matched U-Net NFE–FID figure reported in README.md."""
from pathlib import Path
import matplotlib.pyplot as plt

NFE = [2, 4, 8, 10, 16, 20, 32, 50, 64, 100]
SERIES = {
    "DDPM + DDIM": [389.6992, 56.7488, 29.1369, 26.2918, 22.0175, 19.9765, 17.9058, 16.3803, 16.0098, 15.4242],
    "FM-U + Euler": [174.2554, 68.3946, 31.7916, 27.6283, 21.9929, 20.8186, 18.4714, 17.3398, 16.5395, 15.9969],
    "FM-U + Heun": [404.2330, 311.5474, 139.7589, 109.3425, 67.4344, 53.9351, 32.1175, 20.1862, 16.9381, 14.5048],
    "FM continued + Euler": [176.9113, 69.8698, 33.1324, 28.7829, 22.9350, 21.6280, 19.1418, 17.9956, 17.2232, 16.6269],
    "Reflow + Euler": [20.2793, 18.2201, 17.2271, 17.0783, 16.7047, 16.6678, 16.4586, 16.4426, 16.3228, 16.3683],
}
STYLE = ["o-", "s-", "^-", "D-", "P-"]
COLORS = ["#555555", "#2474B5", "#E1812C", "#3A923A", "#C43C39"]

fig, (full, zoom) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
for (label, values), style, color in zip(SERIES.items(), STYLE, COLORS):
    full.plot(NFE, values, style, color=color, linewidth=2, markersize=5, label=label)
    zoom.plot(NFE, values, style, color=color, linewidth=2, markersize=5, label=label)
for ax in (full, zoom):
    ax.set_xscale("log")
    ax.set_xticks(NFE, labels=[str(x) for x in NFE])
    ax.set_xlabel("Actual network evaluations per image (NFE)")
    ax.grid(True, which="both", alpha=.25)
full.set_yscale("log")
full.set_ylabel("FID-2048 (lower is better; log scale)")
full.set_title("Full measured range")
zoom.set_ylim(12, 40)
zoom.set_ylabel("FID-2048 (lower is better)")
zoom.set_title("Detail for FID 12–40")
zoom.legend(frameon=False, fontsize=8)
fig.suptitle("CIFAR-10 matched Project 1 U-Net comparison (seed 44)")
fig.tight_layout()
out = Path(__file__).resolve().parent / "results" / "nfe_fid_curve.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
