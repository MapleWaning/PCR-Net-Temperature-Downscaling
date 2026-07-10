from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ModuleResult:
    name: str
    outputs: list[Path] = field(default_factory=list)
    reports: list[Path] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class DataModule(Protocol):
    name: str

    def plan(self, context) -> ModuleResult:
        ...

    def run(self, context) -> ModuleResult:
        ...
