from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(fn: Callable[[], T], attempts: int, base_sleep_s: float = 2.0) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_sleep_s * (attempt + 1))
    assert last_error is not None
    raise last_error
