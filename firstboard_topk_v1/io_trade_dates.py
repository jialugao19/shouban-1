"""Load and slice the trade calendar for strategy runs."""

from __future__ import annotations

import pandas as pd


def load_open_trade_dates(trade_cal_path: str, start_date: str, end_date: str) -> list[str]:
    """Load open trade dates in [start_date, end_date]."""
    # Load the trade calendar from /data.
    trade_cal = pd.read_feather(trade_cal_path)

    # Filter open days in the inclusive range.
    mask = (trade_cal["is_open"] == 1) & (trade_cal["cal_date"] >= start_date) & (trade_cal["cal_date"] <= end_date)
    open_days = trade_cal.loc[mask, ["cal_date"]]

    # Return sorted date strings.
    return sorted(open_days["cal_date"].astype(str).tolist())


def load_open_trade_dates_upto(trade_cal_path: str, end_date: str) -> list[str]:
    """Load open trade dates up to end_date (inclusive)."""
    # Load the trade calendar from /data.
    trade_cal = pd.read_feather(trade_cal_path)

    # Filter open days up to end_date.
    mask = (trade_cal["is_open"] == 1) & (trade_cal["cal_date"] <= end_date)
    open_days = trade_cal.loc[mask, ["cal_date"]]

    # Return sorted date strings.
    return sorted(open_days["cal_date"].astype(str).tolist())

