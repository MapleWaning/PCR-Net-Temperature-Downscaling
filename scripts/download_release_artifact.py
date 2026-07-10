from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "artifacts" / "artifact_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Download and install Release-managed PCR-Net artifacts.")
    parser.add_argument("artifact", help="Artifact key, for example mini_case, pcr_time, pcr_spatial, or all.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def flatten_artifacts(manifest: dict) -> dict[str, dict]:
    flattened = {}
    for group_name, group in manifest.get("artifacts", {}).items():
        for artifact_name, artifact in group.items():
            flattened[artifact_name] = artifact
            flattened[f"{group_name}.{artifact_name}"] = artifact
    return flattened


def artifact_url(manifest: dict, artifact: dict) -> str:
    if artifact.get("download_url"):
        return artifact["download_url"]
    release = manifest["release"]
    return release["url_template"].format(tag=release["tag"], asset_name=artifact["release_asset_name"])


def resolve(path: str | Path) -> Path:
    return (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: dict) -> None:
    expected_size = artifact.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}")

    expected_hash = artifact.get("sha256")
    if expected_hash:
        actual_hash = sha256(path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(f"SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}")


def artifact_ready(artifact: dict) -> bool:
    install = artifact.get("install", {})
    if install.get("action") == "extract_tar_gz":
        return all(resolve(path).exists() for path in artifact.get("required_after_install", []))

    runtime_path = artifact.get("runtime_path") or install.get("target_path")
    if not runtime_path:
        return False
    path = resolve(runtime_path)
    if not path.exists():
        return False
    try:
        verify_file(path, artifact)
    except ValueError:
        return False
    return True


def ensure_downloaded(manifest: dict, artifact: dict, retries: int, timeout: int) -> Path:
    destination = resolve(artifact["download_path"])
    if destination.exists():
        verify_file(destination, artifact)
        print(f"Found local artifact: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = artifact_url(manifest, artifact)
    last_error = None
    for attempt in range(1, retries + 1):
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            if partial.exists():
                try:
                    verify_file(partial, artifact)
                    print(f"Found verified partial artifact: {partial}")
                    return partial
                except ValueError:
                    partial.unlink()
            print(f"Downloading {url} -> {destination} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as f:
                shutil.copyfileobj(response, f)
            verify_file(partial, artifact)
            try:
                partial.replace(destination)
                verify_file(destination, artifact)
                return destination
            except OSError as exc:
                print(f"Could not rename temporary download into place; using verified partial file: {exc}", file=sys.stderr)
                return partial
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            print(f"Download attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {artifact['release_asset_name']} after {retries} attempts: {last_error}")


def safe_extract_tar_gz(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    extract_root = extract_to.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (extract_root / member.name).resolve()
            if extract_root != target and extract_root not in target.parents:
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member: {member.name}")
                with source, target.open("wb") as f:
                    shutil.copyfileobj(source, f)


def install_artifact(artifact: dict, downloaded_path: Path) -> None:
    install = artifact.get("install", {})
    action = install.get("action")
    if action == "extract_tar_gz":
        safe_extract_tar_gz(downloaded_path, resolve(install["extract_to"]))
        if install.get("remove_archive_after_extract"):
            downloaded_path.unlink()
        return

    if action == "rename":
        source = resolve(install.get("source_path", artifact["download_path"]))
        target = resolve(install["target_path"])
        if source != downloaded_path and not source.exists():
            source = downloaded_path
        verify_file(source, artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return

    runtime_path = artifact.get("runtime_path")
    if runtime_path:
        target = resolve(runtime_path)
        if target != downloaded_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, target)
        return

    raise ValueError(f"Unsupported install action: {action}")


def ensure_artifact(manifest: dict, artifact: dict, retries: int, timeout: int) -> None:
    if artifact_ready(artifact):
        print(f"Artifact ready: {artifact['release_asset_name']}")
        return
    downloaded_path = ensure_downloaded(manifest, artifact, retries=retries, timeout=timeout)
    install_artifact(artifact, downloaded_path)
    if not artifact_ready(artifact):
        raise RuntimeError(f"Artifact install did not produce the expected runtime files: {artifact['release_asset_name']}")
    print(f"Artifact installed: {artifact['release_asset_name']}")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    artifacts = flatten_artifacts(manifest)
    if args.artifact == "all":
        selected = []
        seen = set()
        for artifact in artifacts.values():
            marker = artifact["release_asset_name"]
            if marker in seen:
                continue
            seen.add(marker)
            selected.append(artifact)
    else:
        if args.artifact not in artifacts:
            valid = ", ".join(sorted(artifacts))
            raise KeyError(f"Unknown artifact '{args.artifact}'. Valid keys: {valid}")
        selected = [artifacts[args.artifact]]

    for artifact in selected:
        ensure_artifact(manifest, artifact, retries=args.retries, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
