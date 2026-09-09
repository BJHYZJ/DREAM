"""Download the models and scene assets required by the simulation profiles."""
import argparse
import subprocess
import sys

from dream_sim.sources import engine_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=("assets", "models"))
    args, forwarded = parser.parse_known_args()
    helper = {"assets": "prepare_instruction_assets.py", "models": "prepare_learned_models.py"}[args.resource]
    # Preserve the user's cwd so relative output/cache paths in the tutorial
    # resolve against the checkout, not the extracted compatibility workspace.
    raise SystemExit(subprocess.call([sys.executable, str(engine_root() / "experiments" / helper), *forwarded]))


if __name__ == "__main__":
    main()
