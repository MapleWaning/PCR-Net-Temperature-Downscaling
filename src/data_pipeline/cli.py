from __future__ import annotations

import argparse

from data_pipeline.config import PipelineConfig
from data_pipeline.pipeline import plan_pipeline, run_pipeline


def parse_module_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="data_pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--modules", default=None, help="Comma-separated module names")
    args = parser.parse_args(argv)

    config = PipelineConfig.load(args.config)
    modules = parse_module_list(args.modules)

    if args.command == "plan":
        results = plan_pipeline(config, modules)
    else:
        results = run_pipeline(config, modules)

    for result in results:
        print(f"{result.name}: outputs={len(result.outputs)} reports={len(result.reports)} details={result.details}")


if __name__ == "__main__":
    main()
