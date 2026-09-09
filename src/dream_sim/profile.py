"""Launch simulation profiles or the dynamic/static memory comparison."""
import sys
from dream_sim.sources import engine_root


def main():
    sys.path.insert(0, str(engine_root() / "experiments"))
    from run_instruction_profile import main as original_main
    original_main()


if __name__ == "__main__":
    main()
