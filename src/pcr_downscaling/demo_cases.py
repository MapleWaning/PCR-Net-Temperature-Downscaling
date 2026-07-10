from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pcr_downscaling.adapters.layout import DataLayout


DATASET_CHOICES = ("smoke_case", "mini_case")
SPLIT_MODE_CHOICES = ("auto", "temporal", "spatial")
VERSION_CHOICES = ("auto", "time", "temporal", "spatial")
DEMO1_FALLBACK_NAME = "demo1_model_ready"
DEMO1_FALLBACK_RELATIVE_ROOT = Path("outputs") / "demos" / "01_data_fetch" / "model_ready"


@dataclass(frozen=True)
class DemoCase:
    name: str
    root: Path
    layout: DataLayout
    years: list[int]
    split_mode: str
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    train_sample_csv: Path
    val_sample_csv: Path
    test_sample_csv: Path
    all_sample_csv: Path
    train_meta_csv: Path
    val_meta_csv: Path
    test_meta_csv: Path

    @property
    def years_arg(self) -> str:
        return ",".join(str(year) for year in self.years)

    @property
    def train_years_arg(self) -> str:
        return ",".join(str(year) for year in self.train_years)

    @property
    def val_years_arg(self) -> str:
        return ",".join(str(year) for year in self.val_years)

    @property
    def test_years_arg(self) -> str:
        return ",".join(str(year) for year in self.test_years)

    @property
    def feature_split(self) -> str:
        return "year" if self.split_mode == "temporal" else "station"

    @property
    def generalization(self) -> str:
        return "year" if self.split_mode == "temporal" else "station"


def add_demo_case_args(parser) -> None:
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="smoke_case")
    parser.add_argument("--data-root", default=None, help="Override the prepared demo dataset root.")
    parser.add_argument("--split-mode", choices=SPLIT_MODE_CHOICES, default="auto")
    parser.add_argument(
        "--version",
        choices=VERSION_CHOICES,
        default="auto",
        help="Alias for the temporal/time or spatial demo version.",
    )


def resolve_demo_case(root: Path, dataset: str, data_root: str | None, split_mode: str, version: str = "auto") -> DemoCase:
    if data_root:
        case_root = Path(data_root).expanduser().resolve()
        name = Path(data_root).name
        if not case_root.exists():
            raise FileNotFoundError(f"Demo dataset was not found: {case_root}")
    else:
        case_root, name = resolve_packaged_or_demo1_case(root, dataset)

    manifest = read_json(case_root / "manifest.json")
    split_dir = case_root / "model_inputs" / "splits"
    years = [int(year) for year in manifest.get("years", infer_years(case_root))]
    chosen_split = choose_split_mode(dataset, split_mode, version)

    if chosen_split == "temporal":
        temporal = read_json(split_dir / "temporal_years.json")
        train_years = [int(year) for year in temporal["train"]]
        val_years = [int(year) for year in temporal["val"]]
        test_years = [int(year) for year in temporal["test"]]
        sample_prefix = "temporal"
    else:
        train_years = years
        val_years = years
        test_years = years
        sample_prefix = "spatial"

    return DemoCase(
        name=name,
        root=case_root,
        layout=DataLayout.from_root(case_root),
        years=years,
        split_mode=chosen_split,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        train_sample_csv=split_dir / f"{sample_prefix}_train_samples.csv",
        val_sample_csv=split_dir / f"{sample_prefix}_val_samples.csv",
        test_sample_csv=split_dir / f"{sample_prefix}_test_samples.csv",
        all_sample_csv=split_dir / "all_selected_samples.csv",
        train_meta_csv=split_dir / "train_meta.csv",
        val_meta_csv=split_dir / "val_meta.csv",
        test_meta_csv=split_dir / "test_meta.csv",
    )


def resolve_packaged_or_demo1_case(root: Path, dataset: str) -> tuple[Path, str]:
    prepared_roots = {name: (root / "assets" / "demo_data" / name).resolve() for name in DATASET_CHOICES}
    selected_root = prepared_roots[dataset]
    if is_prepared_case_available(selected_root):
        return selected_root, dataset

    available = [name for name, path in prepared_roots.items() if is_prepared_case_available(path)]
    if not available:
        demo1_root = (root / DEMO1_FALLBACK_RELATIVE_ROOT).resolve()
        if is_demo1_model_ready_available(demo1_root):
            return demo1_root, DEMO1_FALLBACK_NAME
        raise FileNotFoundError(
            "Neither smoke_case nor mini_case is available, and Demo 01 model_ready data was not found. "
            f"Expected Demo 01 fallback at: {demo1_root}"
        )

    available_text = ", ".join(available)
    raise FileNotFoundError(
        f"Requested demo dataset is not available: {selected_root}. "
        f"Available prepared dataset(s): {available_text}. Use --dataset to select one of them."
    )


def is_prepared_case_available(case_root: Path) -> bool:
    split_dir = case_root / "model_inputs" / "splits"
    required = [
        case_root / "manifest.json",
        case_root / "model_inputs" / "static.npy",
        case_root / "model_inputs" / "truth.npy",
        split_dir / "all_selected_samples.csv",
        split_dir / "train_meta.csv",
        split_dir / "val_meta.csv",
        split_dir / "test_meta.csv",
    ]
    return all(path.exists() for path in required)


def is_demo1_model_ready_available(case_root: Path) -> bool:
    if not is_prepared_case_available(case_root):
        return False
    required = [
        case_root / "standard" / "era5",
        case_root / "standard" / "lst",
        case_root / "physical" / "t_base",
    ]
    years = infer_years(case_root)
    return (
        all(path.exists() for path in required)
        and bool(years)
        and all((case_root / "standard" / "lst" / f"lst_{year}.h5").exists() for year in years)
        and all(has_tbase_file(case_root, year) for year in years)
    )


def has_tbase_file(case_root: Path, year: int) -> bool:
    tbase_dir = case_root / "physical" / "t_base"
    return any((tbase_dir / name).exists() for name in (f"t_base_advanced_{year}.h5", f"t_base_{year}.h5"))


def is_demo1_model_ready_case(case: DemoCase) -> bool:
    manifest_path = case.root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        return read_json(manifest_path).get("dataset") == DEMO1_FALLBACK_NAME
    except (OSError, json.JSONDecodeError):
        return False


def choose_split_mode(dataset: str, split_mode: str, version: str = "auto") -> str:
    version_split = version_to_split_mode(version)
    if split_mode != "auto" and version_split != "auto" and split_mode != version_split:
        raise ValueError(f"Conflicting split selectors: --split-mode {split_mode} vs --version {version}")
    if version_split != "auto":
        return version_split
    if split_mode != "auto":
        return split_mode
    if dataset == "mini_case":
        return "temporal"
    return "spatial"


def version_to_split_mode(version: str) -> str:
    if version in ("auto", None):
        return "auto"
    if version in ("time", "temporal"):
        return "temporal"
    if version == "spatial":
        return "spatial"
    raise ValueError(f"Unsupported version: {version}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_years(case_root: Path) -> list[int]:
    era5_dir = case_root / "standard" / "era5"
    years = []
    for path in sorted(era5_dir.glob("era5_t2m_*.h5")):
        years.append(int(path.stem.rsplit("_", 1)[-1]))
    return years
