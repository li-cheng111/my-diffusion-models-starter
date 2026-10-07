"""Create a compact, inference-only artifact from a resumable training checkpoint."""
import argparse
import os

import torch


def export_checkpoint(source, destination):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("step", "model", "cfg")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"checkpoint is missing required fields: {', '.join(missing)}")
    artifact = {
        key: checkpoint[key]
        for key in ("step", "model", "ema", "cfg", "seed", "precision")
        if key in checkpoint
    }
    destination = os.path.abspath(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".tmp"
    torch.save(artifact, temporary)
    os.replace(temporary, destination)
    print(f"exported step {checkpoint['step']} inference checkpoint to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Full resumable checkpoint")
    parser.add_argument("--output", required=True, help="Compact inference artifact")
    args = parser.parse_args()
    export_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
