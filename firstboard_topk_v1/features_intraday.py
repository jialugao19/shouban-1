"""Compute intraday features from stock1m required by scoring and backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shouban2.firstboard_topk_v1.config import LimitQualityParams
from shouban2.firstboard_topk_v1.io_market import read_stock1m_subset


@dataclass(frozen=True)
class LimitQualityResult:
    """Store per-stock limit-quality components derived from stock1m."""

    StockCode: int
    first_seal_minute: int
    open_board_count: int
    open_board_vol_ratio: float
    score_limit_time: float
    score_limit_hardness: float
    score_limit_quality: float


def compute_limit_quality(
    stock1m_root: str,
    trade_date_prev: str,
    stock_codes: list[int],
    up_limit_by_stock: dict[int, float],
    params: LimitQualityParams,
) -> pd.DataFrame:
    """Compute limit-quality scores for a stock subset on trade_date_prev using stock1m."""
    # Load minute bars for the needed stock subset.
    cols = ["StockCode", "MinuteIndex", "Close", "Vol"]
    df = read_stock1m_subset(trade_date=trade_date_prev, stock1m_root=stock1m_root, stock_codes=stock_codes, columns=cols)
    df["StockCode"] = df["StockCode"].astype(int)
    df["MinuteIndex"] = df["MinuteIndex"].astype(int)

    # Build per-stock results by scanning minute sequences.
    results: list[LimitQualityResult] = []
    for stock_code, g in df.groupby("StockCode", sort=False):
        # Resolve up_limit for this stock on d-1.
        up_limit = float(up_limit_by_stock[int(stock_code)])

        # Find first seal minute where Close reaches up_limit.
        close = g["Close"].to_numpy(dtype=float)
        minute = g["MinuteIndex"].to_numpy(dtype=int)
        vol = g["Vol"].to_numpy(dtype=float)
        seal_mask = close == up_limit
        if bool(np.any(seal_mask)):
            first_seal_minute = int(minute[seal_mask].min())
        else:
            first_seal_minute = int(params.seal_time_norm)

        # Count open-board minutes and open-board volume after first seal.
        after = minute > first_seal_minute
        open_board = after & (close < up_limit)
        open_board_count = int(open_board.sum())
        vol_total = float(vol.sum())
        vol_open_board = float(vol[open_board].sum()) if vol_total > 0.0 else 0.0
        open_board_vol_ratio = float(vol_open_board / vol_total) if vol_total > 0.0 else 0.0

        # Map components into [0,1] scores.
        score_limit_time = float(1.0 - np.clip(first_seal_minute / float(params.seal_time_norm), 0.0, 1.0))
        score_limit_hardness = float(1.0 - np.clip(open_board_vol_ratio / float(params.open_board_vol_norm), 0.0, 1.0))
        score_limit_quality = float(0.6 * score_limit_time + 0.4 * score_limit_hardness)

        results.append(
            LimitQualityResult(
                StockCode=int(stock_code),
                first_seal_minute=first_seal_minute,
                open_board_count=open_board_count,
                open_board_vol_ratio=open_board_vol_ratio,
                score_limit_time=score_limit_time,
                score_limit_hardness=score_limit_hardness,
                score_limit_quality=score_limit_quality,
            )
        )

    # Return a stable, typed DataFrame for joins.
    out = pd.DataFrame([r.__dict__ for r in results])
    return out.sort_values(["StockCode"]).reset_index(drop=True)
