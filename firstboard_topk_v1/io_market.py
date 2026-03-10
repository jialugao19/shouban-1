"""Provide typed loaders for stock1d/stock1m files under /data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


def stock1d_path(stock1d_root: str, trade_date: str) -> str:
    """Build a stock1d file path for one trade_date."""
    # Build a YYYY/YYYYMMDD.feather path.
    return str(Path(stock1d_root) / trade_date[:4] / f"{trade_date}.feather")


def stock1m_path(stock1m_root: str, trade_date: str) -> str:
    """Build a stock1m file path for one trade_date."""
    # Build a YYYY/YYYYMMDD.feather path.
    return str(Path(stock1m_root) / trade_date[:4] / f"{trade_date}.feather")


def read_stock1d(trade_date: str, stock1d_root: str) -> pd.DataFrame:
    """Read one day of stock1d with required columns."""
    # Declare the minimal column set used by the strategy.
    cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "up_limit",
        "down_limit",
    ]

    # Read the feather file into pandas.
    path = stock1d_path(stock1d_root, trade_date)
    return pd.read_feather(path, columns=cols)


def read_stock1m_table(trade_date: str, stock1m_root: str, columns: list[str]):
    """Read one day of stock1m into a pyarrow Table (memory-mapped)."""
    # Read the feather file as a memory-mapped arrow table.
    path = stock1m_path(stock1m_root, trade_date)
    return feather.read_table(path, columns=columns, memory_map=True)


def read_stock1m_subset(trade_date: str, stock1m_root: str, stock_codes: list[int], columns: list[str]) -> pd.DataFrame:
    """Read one day of stock1m and keep only the requested StockCode subset."""
    # Load the arrow table for the requested columns.
    table = read_stock1m_table(trade_date=trade_date, stock1m_root=stock1m_root, columns=columns)

    # Return empty outputs for empty subsets to keep call sites simple.
    if len(stock_codes) == 0:
        return table.slice(0, 0).to_pandas(self_destruct=True)

    # Filter the arrow table before converting to pandas for performance.
    value_set = pa.array(stock_codes, type=pa.int64())
    mask = pc.is_in(table["StockCode"], value_set=value_set)
    filtered = table.filter(mask)
    return filtered.to_pandas(self_destruct=True).reset_index(drop=True)
