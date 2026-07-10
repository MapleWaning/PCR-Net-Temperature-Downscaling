from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    root: Path
    standard: Path
    physical: Path
    catboost_inference: Path
    model_inputs: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DataLayout":
        base = Path(root).expanduser().resolve()
        return cls(
            root=base,
            standard=base / "standard",
            physical=base / "physical",
            catboost_inference=base / "catboost_inference",
            model_inputs=base / "model_inputs",
        )

    def expected_paths(self) -> dict[str, Path]:
        return {
            "metadata": self.standard / "metadata" / "high-quality-meta.csv",
            "weather": self.standard / "gsod" / "weather_data.npy",
            "tbase": self.physical / "t_base",
            "catboost_guidance": self.catboost_inference / "catboost_lst",
            "static": self.model_inputs / "static.npy",
            "truth": self.model_inputs / "truth.npy",
        }

    def existence_report(self) -> dict[str, bool]:
        return {name: path.exists() for name, path in self.expected_paths().items()}
