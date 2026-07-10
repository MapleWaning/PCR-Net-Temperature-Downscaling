from __future__ import annotations

import calendar
from datetime import date

import pandas as pd


def is_leap_year(year: int) -> bool:
    return calendar.isleap(year)


def days_in_year(year: int) -> int:
    return 366 if is_leap_year(year) else 365


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def daily_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="D")


def year_date_range(year: int, days: int | None = None) -> pd.DatetimeIndex:
    periods = days if days is not None else days_in_year(year)
    return pd.date_range(start=date(year, 1, 1), periods=periods, freq="D")
