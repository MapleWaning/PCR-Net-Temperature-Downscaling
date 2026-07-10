from __future__ import annotations

from data_pipeline.modules.albedo import AlbedoModule
from data_pipeline.modules.cci import CciModule
from data_pipeline.modules.dem import DemModule
from data_pipeline.modules.era5 import Era5Module
from data_pipeline.modules.gsod import GsodModule
from data_pipeline.modules.lst import LstModule


MODULES = {
    "gsod": GsodModule(),
    "albedo": AlbedoModule(),
    "era5": Era5Module(),
    "lst": LstModule(),
    "dem": DemModule(),
    "cci": CciModule(),
}

DEFAULT_ORDER = ["gsod", "albedo", "era5", "lst", "dem", "cci"]


def get_modules(names: list[str] | None = None):
    selected = names or DEFAULT_ORDER
    missing = [name for name in selected if name not in MODULES]
    if missing:
        raise KeyError(f"Unknown data modules: {missing}")
    return [MODULES[name] for name in DEFAULT_ORDER if name in selected]
