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
    sealed_minutes_total: int
    sealed_ratio: float
    last_seal_minute: int
    longest_seal_span: int
    open_board_count: int
    open_board_count_post_first_seal: int
    open_board_minutes_total: int
    reopen_density: float
    tail_seal_flag: int
    open_board_vol_ratio: float
    score_limit_time: float
    score_total_seal_duration: float
    score_reopen_count: float
    score_tail_stability: float
    score_open_board_vol_penalty: float
    score_limit_hardness: float
    score_limit_quality: float


def _longest_true_span(mask: np.ndarray) -> int:
    """Compute the longest consecutive True span in a boolean mask."""
    # Track the current and best span lengths.
    best = 0
    current = 0
    for value in mask.tolist():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _count_reopen_events(mask_after_first: np.ndarray) -> int:
    """Count reopen events after the first seal using state transitions."""
    # Count transitions from sealed to open-board.
    count = 0
    prev = bool(mask_after_first[0])
    for value in mask_after_first[1:].tolist():
        current = bool(value)
        if prev and (not current):
            count += 1
        prev = current
    return int(count)


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
            last_seal_minute = int(minute[seal_mask].max())
        else:
            first_seal_minute = int(params.seal_time_norm)
            last_seal_minute = int(params.seal_time_norm)

        # Count sealing and reopen features across the whole day.
        sealed_minutes_total = int(seal_mask.sum())
        sealed_ratio = float(sealed_minutes_total / max(1, len(minute)))
        longest_seal_span = _longest_true_span(seal_mask)

        # Count open-board minutes and open-board volume after first seal.
        after = minute > first_seal_minute
        seal_after_first = after & seal_mask
        open_board = after & (close < up_limit)
        open_board_count_post_first_seal = _count_reopen_events(seal_after_first) if bool(np.any(seal_mask)) else 0
        open_board_minutes_total = int(open_board.sum())
        reopen_density = float(open_board_count_post_first_seal / max(1, sealed_minutes_total))
        vol_total = float(vol.sum())
        vol_open_board = float(vol[open_board].sum()) if vol_total > 0.0 else 0.0
        open_board_vol_ratio = float(vol_open_board / vol_total) if vol_total > 0.0 else 0.0
        tail_window = min(int(params.tail_minutes), int(len(seal_mask)))
        tail_seal_flag = int(bool(np.all(seal_mask[-tail_window:]))) if tail_window > 0 else 0

        # Map components into [0,1] scores.
        score_limit_time = float(1.0 - np.clip(first_seal_minute / float(params.seal_time_norm), 0.0, 1.0))
        score_total_seal_duration = float(np.clip(sealed_minutes_total / float(params.seal_duration_norm), 0.0, 1.0))
        score_reopen_count = float(1.0 - np.clip(open_board_count_post_first_seal / float(params.reopen_count_norm), 0.0, 1.0))
        score_tail_stability = float(tail_seal_flag)
        score_open_board_vol_penalty = float(1.0 - np.clip(open_board_vol_ratio / float(params.open_board_vol_norm), 0.0, 1.0))
        score_limit_hardness = float(1.0 - np.clip(open_board_vol_ratio / float(params.open_board_vol_norm), 0.0, 1.0))
        score_limit_quality = float(
            0.20 * score_limit_time
            + 0.30 * score_total_seal_duration
            + 0.20 * score_reopen_count
            + 0.15 * score_tail_stability
            + 0.15 * score_open_board_vol_penalty
        )

        results.append(
            LimitQualityResult(
                StockCode=int(stock_code),
                first_seal_minute=first_seal_minute,
                sealed_minutes_total=sealed_minutes_total,
                sealed_ratio=sealed_ratio,
                last_seal_minute=last_seal_minute,
                longest_seal_span=longest_seal_span,
                open_board_count=open_board_count_post_first_seal,
                open_board_count_post_first_seal=open_board_count_post_first_seal,
                open_board_minutes_total=open_board_minutes_total,
                reopen_density=reopen_density,
                tail_seal_flag=tail_seal_flag,
                open_board_vol_ratio=open_board_vol_ratio,
                score_limit_time=score_limit_time,
                score_total_seal_duration=score_total_seal_duration,
                score_reopen_count=score_reopen_count,
                score_tail_stability=score_tail_stability,
                score_open_board_vol_penalty=score_open_board_vol_penalty,
                score_limit_hardness=score_limit_hardness,
                score_limit_quality=score_limit_quality,
            )
        )

    # Return a stable, typed DataFrame for joins.
    out = pd.DataFrame([r.__dict__ for r in results])
    return out.sort_values(["StockCode"]).reset_index(drop=True)
