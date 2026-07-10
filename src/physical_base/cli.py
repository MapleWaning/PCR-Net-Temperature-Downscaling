from __future__ import annotations

import argparse

from physical_base.config import PhysicalConfig
from physical_base.pipeline import plan, run_all, run_factors_stage, run_tbase_stage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="physical_base")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run-factors", "run-tbase", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = PhysicalConfig.load(args.config)
    if args.command == "plan":
        output = plan(config)
        print(f"plan: {output}")
    elif args.command == "run-factors":
        outputs = run_factors_stage(config)
        print(f"run-factors: {len(outputs)} outputs")
    elif args.command == "run-tbase":
        outputs = run_tbase_stage(config)
        print(f"run-tbase: {len(outputs)} outputs")
    else:
        outputs = run_all(config)
        print(f"run: {len(outputs)} outputs")


if __name__ == "__main__":
    main()
