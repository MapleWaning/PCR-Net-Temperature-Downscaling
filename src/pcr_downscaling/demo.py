from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def repo_root_from(path: str | Path) -> Path:
    return Path(path).resolve().parents[2]


def ensure_src_on_path(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if not module_available(name)]


def write_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_python(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    conda_exe = env.get("PCR_CONDA_EXE")
    conda_env = env.get("PCR_CONDA_ENV")
    command = [sys.executable, *args]
    if conda_exe and conda_env:
        command = [conda_exe, "run", "-n", conda_env, "python", *args]
    return subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def ensure_release_artifact(root: Path, artifact_name: str) -> subprocess.CompletedProcess[str]:
    command = ["scripts/download_release_artifact.py", artifact_name]
    artifact_manifest = os.environ.get("PCR_ARTIFACT_MANIFEST")
    if artifact_manifest:
        command.extend(["--manifest", artifact_manifest])
    return run_python(root, command)


def completed_process_summary(result: subprocess.CompletedProcess[str] | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
    }


def ensure_demo_dataset(root: Path, dataset: str, data_root: str | None) -> tuple[str, subprocess.CompletedProcess[str] | None]:
    if data_root is not None or dataset != "mini_case":
        return dataset, None

    result = ensure_release_artifact(root, "mini_case")
    from pcr_downscaling.demo_cases import is_prepared_case_available

    mini_case_root = root / "assets" / "demo_data" / "mini_case"
    if result.returncode == 0 and is_prepared_case_available(mini_case_root):
        return dataset, result
    return "smoke_case", result


def ensure_pretrained_checkpoint(
    root: Path,
    split_mode: str,
    explicit_model_path: str | None,
) -> subprocess.CompletedProcess[str] | None:
    if explicit_model_path:
        return None
    artifact_name = "pcr_time" if split_mode == "temporal" else "pcr_spatial"
    return ensure_release_artifact(root, artifact_name)


def command_line(args: list[str]) -> str:
    return " ".join(args)
