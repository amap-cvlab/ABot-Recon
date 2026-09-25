from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Keep the repository script directly runnable before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abot_recon.training.ema import merge_ema_state_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge ABot-Recon trainable-parameter EMA into a complete checkpoint."
    )
    parser.add_argument(
        "--model", required=True, help="Complete online .safetensors checkpoint"
    )
    parser.add_argument("--ema", required=True, help="Training ema.pt checkpoint")
    parser.add_argument("--output", required=True, help="Output complete .safetensors path")
    args = parser.parse_args()

    model_path = Path(args.model)
    ema_path = Path(args.ema)
    output_path = Path(args.output)
    if output_path.resolve() in {model_path.resolve(), ema_path.resolve()}:
        raise ValueError("--output must not overwrite --model or --ema")

    full_state = load_file(str(model_path), device="cpu")
    ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
    merged = merge_ema_state_dict(full_state, ema_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_path))
    print(f"Wrote complete EMA checkpoint: {output_path}")


if __name__ == "__main__":
    main()
