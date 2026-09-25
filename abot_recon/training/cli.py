from __future__ import annotations

import argparse

from .config import load_config
from .trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the released ABot-Recon model")
    parser.add_argument("--config", required=True, help="Path to a training YAML file")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
