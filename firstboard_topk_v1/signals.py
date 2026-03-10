"""Generate daily candidates, continuous scores, and Top-K picks for firstboard_topk_v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from shouban2.firstboard_topk_v1.config import StrategyConfig
from shouban2.firstboard_topk_v1.features_intraday import compute_limit_quality
from shouban2.firstboard_topk_v1.io_market import read_stock1m_subset


@dataclass(frozen=True)
class DaySignals:
    """Hold per-day signals outputs for persistence and backtest."""

    candidates: pd.DataFrame
    topk: pd.DataFrame
    filtered_limit_up_at_open: int


def _clip01(x: np.ndarray) -> np.ndarray:
    """Clip an array into [0,1]."""
    # Clamp values into the unit interval.
    return np.clip(x, 0.0, 1.0)


def _score_triangle(open_ratio: np.ndarray, low: float, high: float) -> np.ndarray:
    """Score open_ratio by triangular distance to the interval center."""
    # Compute the interval center and half-width.
    center = 0.5 * (low + high)
    half = 0.5 * (high - low)

    # Map distance to [0,1] with peak at the center.
    score = 1.0 - np.abs(open_ratio - center) / half
    return _clip01(score)


def _score_auction_sbgk(x: np.ndarray, t_min: float, t_hot: float, t_max: float) -> np.ndarray:
    """Score auction_vol_ratio for sbgk with an overheat penalty."""
    # Map the cold region linearly up to 1.
    score = np.ones_like(x, dtype=float)
    cold = x < t_min
    score[cold] = x[cold] / t_min

    # Penalize the overheat region beyond t_hot.
    hot = x > t_hot
    score[hot] = 1.0 - (x[hot] - t_hot) / (t_max - t_hot)
    return _clip01(score)


def _score_auction_sbdk(x: np.ndarray, t_best: float, t_max: float) -> np.ndarray:
    """Score auction_vol_ratio for sbdk by preferring low-volume stabilization."""
    # Map distance to t_best into a triangle that decays to 0 at t_max.
    score = 1.0 - np.abs(x - t_best) / (t_max - t_best)
    return _clip01(score)


def _score_liquidity(amount_cny: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """Score liquidity by a log-saturated mapping on amount."""
    # Map log(amount) into [0,1] with saturation.
    score = (np.log(amount_cny) - np.log(a0)) / (np.log(a1) - np.log(a0))
    return _clip01(score)


def _score_linear(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Score a value linearly from x0->0 to x1->1."""
    # Map x into [0,1] by a linear transform.
    score = (x - x0) / (x1 - x0)
    return _clip01(score)


def _score_rp_penalty(rp60: np.ndarray) -> np.ndarray:
    """Score rp by penalizing high relative-position names."""
    # Convert rp into a penalty then invert into a score.
    penalty = (rp60 - 0.5) / 0.3
    return 1.0 - _clip01(penalty)


def _score_upper_shadow(upper_shadow: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """Score upper shadow ratio by a linear mapping."""
    # Map upper_shadow into [0,1] within [s0,s1].
    return _score_linear(upper_shadow, s0, s1)


def _score_vol_shrinkage(vol_shrinkage: np.ndarray, width: float) -> np.ndarray:
    """Score vol_shrinkage by closeness to 1.0."""
    # Convert closeness-to-1 into a triangle.
    score = 1.0 - np.abs(vol_shrinkage - 1.0) / width
    return _clip01(score)


def _choose_topk(
    ranked: pd.DataFrame,
    topk_k: int,
    filter_limit_up_at_open: bool,
    score_threshold: float,
    dynamic_k_by_score: bool,
) -> tuple[pd.DataFrame, int]:
    """Select Top-K rows with an optional limit-up-at-open filter."""
    # Walk ranked rows and pick the first K that pass the selection constraints.
    picked_rows: list[int] = []
    filtered = 0
    for row_i, row in ranked.iterrows():
        # Apply score threshold only when dynamic-k mode is enabled.
        if dynamic_k_by_score and float(row["score_total"]) < float(score_threshold):
            break
        if filter_limit_up_at_open and bool(row["limit_up_at_open"]):
            filtered += 1
            continue
        picked_rows.append(int(row_i))
        if len(picked_rows) >= topk_k:
            break

    # Return the selected Top-K table.
    out = ranked.loc[picked_rows].copy()
    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, int(out.shape[0]) + 1, dtype=int)
    return out, int(filtered)


def build_day_signals(
    config: StrategyConfig,
    daily_panel: pl.DataFrame,
    valid_by_date: dict[int, set[int]],
    trade_date: str,
    prev_date: str,
) -> DaySignals:
    """Build daily candidates with scores and the final Top-K list."""
    # Prepare d and d-1 integers for joins and filters.
    d = int(trade_date)
    dprev = int(prev_date)

    # Join today's open and up_limit with yesterday's features.
    today = daily_panel.filter(pl.col("trade_date") == d).select(
        ["StockCode", "ts_code", "trade_date", "open", "close", "up_limit", "ma5_close"]
    )
    prev = daily_panel.filter(pl.col("trade_date") == dprev).select(
        [
            "StockCode",
            pl.col("close").alias("close_prev"),
            pl.col("open").alias("open_prev"),
            pl.col("high").alias("high_prev"),
            pl.col("low").alias("low_prev"),
            pl.col("amount").alias("amount_prev"),
            pl.col("vol").alias("vol_prev"),
            pl.col("vwap").alias("vwap_prev"),
            pl.col("rp").alias("rp_prev"),
            pl.col("first_straight_board").alias("first_straight_prev"),
            pl.col("first_touch_not_close").alias("first_touch_not_close_prev"),
            pl.col("close_lag3").alias("close_d4"),
            pl.col("vol_lag1").alias("vol_d2"),
            pl.col("up_limit").alias("up_limit_prev"),
        ]
    )
    base = today.join(prev, on="StockCode", how="inner").with_columns((pl.col("open") / pl.col("close_prev")).alias("open_ratio"))

    # Filter the universe by valid_stocks for this trade_date.
    base_pd = base.to_pandas()
    valid_codes = valid_by_date[int(d)]
    base_pd = base_pd.loc[base_pd["StockCode"].astype(int).isin(valid_codes)].reset_index(drop=True)

    # Build candidate masks for three paths using the plan constraints.
    amount_cny_prev = base_pd["amount_prev"].astype(float) * 1000.0
    rp_prev = base_pd["rp_prev"].astype(float)

    sbdk_mask = (base_pd["first_straight_prev"].astype(bool)) & (amount_cny_prev >= 1e8) & (rp_prev <= 0.6)
    sbgk_mask = (base_pd["first_straight_prev"].astype(bool)) & (amount_cny_prev >= 5.5e8) & (amount_cny_prev <= 2.0e9)
    rzg_mask = (base_pd["first_touch_not_close_prev"].astype(bool)) & (
        base_pd["close_prev"].astype(float) / base_pd["close_d4"].astype(float) - 1.0 <= 0.28
    )

    base_pd["is_sbdk"] = sbdk_mask
    base_pd["is_sbgk"] = sbgk_mask
    base_pd["is_rzg"] = rzg_mask

    # Restrict to union candidates and prepare the code list for intraday reads.
    union_mask = sbdk_mask | sbgk_mask | rzg_mask
    cand = base_pd.loc[union_mask].copy()
    cand = cand.reset_index(drop=True)
    cand_codes = cand["StockCode"].astype(int).tolist()

    # Short-circuit empty days so downstream IO is avoided.
    if int(cand.shape[0]) == 0:
        # Ensure a stable schema so downstream aggregation and feather writers don't break.
        cand["trade_date"] = int(d)
        cand["auction_vol_ratio"] = np.array([], dtype=float)
        cand["amount_cny_prev"] = np.array([], dtype=float)
        cand["limit_up_at_open"] = np.array([], dtype=bool)
        cand["score_total"] = np.array([], dtype=float)
        cand["score_path"] = np.array([], dtype=object)
        topk = cand.head(0).copy()
        topk["trade_date"] = int(d)
        topk["rank"] = np.array([], dtype=int)
        return DaySignals(candidates=cand, topk=topk, filtered_limit_up_at_open=0)

    # Load auction volume ratio for candidates using minute index 0 on trade_date.
    m_cols = ["StockCode", "MinuteIndex", "Vol"]
    auction_df = read_stock1m_subset(trade_date=trade_date, stock1m_root=config.stock1m_root, stock_codes=cand_codes, columns=m_cols)
    auction_df["StockCode"] = auction_df["StockCode"].astype(int)
    auction_df["MinuteIndex"] = auction_df["MinuteIndex"].astype(int)
    auction_df = auction_df.loc[auction_df["MinuteIndex"] == 0, ["StockCode", "Vol"]].rename(columns={"Vol": "auction_vol"})

    # Join auction volume and compute auction_vol_ratio on a shares basis.
    cand = cand.merge(auction_df, on="StockCode", how="left")
    missing_auction = cand["auction_vol"].isna().to_numpy()
    if bool(np.any(missing_auction)):
        # Drop candidates with missing auction bars so long runs can complete.
        missing_n = int(np.sum(missing_auction))
        print(f"[warn] {trade_date} drop_missing_auction_vol n={missing_n}")
        cand = cand.loc[~missing_auction].reset_index(drop=True)
        cand_codes = cand["StockCode"].astype(int).tolist()
        if int(cand.shape[0]) == 0:
            cand["trade_date"] = int(d)
            cand["auction_vol_ratio"] = np.array([], dtype=float)
            cand["amount_cny_prev"] = np.array([], dtype=float)
            cand["limit_up_at_open"] = np.array([], dtype=bool)
            cand["score_total"] = np.array([], dtype=float)
            cand["score_path"] = np.array([], dtype=object)
            topk = cand.head(0).copy()
            topk["trade_date"] = int(d)
            topk["rank"] = np.array([], dtype=int)
            return DaySignals(candidates=cand, topk=topk, filtered_limit_up_at_open=0)
    cand["auction_vol"] = cand["auction_vol"].astype(float)
    cand["auction_vol_ratio"] = cand["auction_vol"].astype(float) / (cand["vol_prev"].astype(float) * 100.0)

    # Compute limit_up_at_open tradability flag for filtering.
    cand["limit_up_at_open"] = cand["open"].astype(float) == cand["up_limit"].astype(float)

    # Compute limit-quality features from stock1m on d-1 for candidates.
    up_limit_by = {int(k): float(v) for k, v in zip(cand["StockCode"].astype(int), cand["up_limit_prev"].astype(float))}
    limit_quality = compute_limit_quality(
        stock1m_root=config.stock1m_root,
        trade_date_prev=prev_date,
        stock_codes=cand_codes,
        up_limit_by_stock=up_limit_by,
        params=config.limit_quality_params,
    )
    cand = cand.merge(limit_quality, on="StockCode", how="left")
    missing_lq = cand["score_limit_quality"].isna().to_numpy()
    if bool(np.any(missing_lq)):
        # Drop candidates with missing limit-quality scores so downstream scoring stays consistent.
        missing_n = int(np.sum(missing_lq))
        print(f"[warn] {trade_date} drop_missing_limit_quality n={missing_n}")
        cand = cand.loc[~missing_lq].reset_index(drop=True)
        cand_codes = cand["StockCode"].astype(int).tolist()
        if int(cand.shape[0]) == 0:
            cand["trade_date"] = int(d)
            cand["auction_vol_ratio"] = np.array([], dtype=float)
            cand["amount_cny_prev"] = np.array([], dtype=float)
            cand["limit_up_at_open"] = np.array([], dtype=bool)
            cand["score_total"] = np.array([], dtype=float)
            cand["score_path"] = np.array([], dtype=object)
            topk = cand.head(0).copy()
            topk["trade_date"] = int(d)
            topk["rank"] = np.array([], dtype=int)
            return DaySignals(candidates=cand, topk=topk, filtered_limit_up_at_open=0)

    # Compute shared features and shared scores.
    cand["amount_cny_prev"] = cand["amount_prev"].astype(float) * 1000.0
    cand["avg_ret_prev"] = cand["vwap_prev"].astype(float) / cand["close_prev"].astype(float) - 1.0
    cand["avg_ret_sbgk"] = cand["vwap_prev"].astype(float) / cand["close_prev"].astype(float) * 1.1 - 1.0
    cand["rp60_prev"] = cand["rp_prev"].astype(float)
    cand["upper_shadow_prev"] = (cand["high_prev"].astype(float) - np.maximum(cand["open_prev"].astype(float), cand["close_prev"].astype(float))) / (
        cand["high_prev"].astype(float) - cand["low_prev"].astype(float)
    )
    cand["vol_shrinkage_prev"] = cand["vol_prev"].astype(float) / cand["vol_d2"].astype(float)

    liq_score = _score_liquidity(cand["amount_cny_prev"].to_numpy(dtype=float), a0=1e8, a1=1e9)
    rp_score = _score_rp_penalty(cand["rp60_prev"].to_numpy(dtype=float))
    limit_quality_score = cand["score_limit_quality"].astype(float).to_numpy(dtype=float)

    cand["score_liquidity"] = liq_score
    cand["score_rp_penalty"] = rp_score
    cand["score_limit_quality"] = limit_quality_score

    # Compute path-specific score components.
    open_ratio = cand["open_ratio"].astype(float).to_numpy(dtype=float)
    auction_x = cand["auction_vol_ratio"].astype(float).to_numpy(dtype=float)

    cand["score_open_sbdk"] = _score_triangle(open_ratio, config.open_band_sbdk.low, config.open_band_sbdk.high)
    cand["score_open_sbgk"] = _score_triangle(open_ratio, config.open_band_sbgk.low, config.open_band_sbgk.high)
    cand["score_open_rzg"] = _score_triangle(open_ratio, config.open_band_rzg.low, config.open_band_rzg.high)

    cand["score_auction_sbgk"] = _score_auction_sbgk(
        auction_x,
        t_min=config.auction_shape_sbgk.t_min,
        t_hot=config.auction_shape_sbgk.t_hot,
        t_max=config.auction_shape_sbgk.t_max,
    )
    cand["score_auction_sbdk"] = _score_auction_sbdk(
        auction_x,
        t_best=config.auction_shape_sbdk.t_best,
        t_max=config.auction_shape_sbdk.t_max,
    )

    strength_sbgk = _score_linear(
        cand["avg_ret_sbgk"].to_numpy(dtype=float),
        x0=config.strength_sbgk.r0,
        x1=config.strength_sbgk.r1,
    )
    strength_rzg = _score_linear(
        cand["avg_ret_prev"].to_numpy(dtype=float),
        x0=config.strength_rzg.r0,
        x1=config.strength_rzg.r1,
    )
    cand["score_strength_sbgk"] = strength_sbgk
    cand["score_strength_rzg"] = strength_rzg

    upper_shadow_score = _score_upper_shadow(
        cand["upper_shadow_prev"].to_numpy(dtype=float),
        s0=config.upper_shadow_range.s0,
        s1=config.upper_shadow_range.s1,
    )
    vol_shrinkage_score = _score_vol_shrinkage(
        cand["vol_shrinkage_prev"].to_numpy(dtype=float),
        width=config.vol_shrinkage_range.width,
    )
    cand["score_upper_shadow"] = upper_shadow_score
    cand["score_vol_shrinkage"] = vol_shrinkage_score
    cand["score_rzg_reversal"] = 0.6 * upper_shadow_score + 0.4 * vol_shrinkage_score

    # Aggregate path scores into score_sbdk/score_sbgk/score_rzg.
    w_sbdk = config.weights_sbdk
    w_sbgk = config.weights_sbgk
    w_rzg = config.weights_rzg

    cand["score_sbdk"] = (
        w_sbdk.w_open_band * cand["score_open_sbdk"].to_numpy(dtype=float)
        + w_sbdk.w_auction * cand["score_auction_sbdk"].to_numpy(dtype=float)
        + w_sbdk.w_liquidity * liq_score
        + w_sbdk.w_rp * rp_score
        + w_sbdk.w_limit_quality * limit_quality_score
    )
    cand["score_sbgk"] = (
        w_sbgk.w_open_band * cand["score_open_sbgk"].to_numpy(dtype=float)
        + w_sbgk.w_auction * cand["score_auction_sbgk"].to_numpy(dtype=float)
        + w_sbgk.w_liquidity * liq_score
        + w_sbgk.w_strength * strength_sbgk
        + w_sbgk.w_limit_quality * limit_quality_score
    )
    cand["score_rzg"] = (
        w_rzg.w_open_band * cand["score_open_rzg"].to_numpy(dtype=float)
        + w_rzg.w_auction * cand["score_auction_sbgk"].to_numpy(dtype=float)
        + w_rzg.w_liquidity * liq_score
        + w_rzg.w_strength * strength_rzg
        + w_rzg.w_reversal * cand["score_rzg_reversal"].to_numpy(dtype=float)
        + w_rzg.w_limit_quality * limit_quality_score
    )

    # Choose score_total and a score_path label for postmortem explainability.
    score_sbdk = cand["score_sbdk"].to_numpy(dtype=float)
    score_sbgk = cand["score_sbgk"].to_numpy(dtype=float)
    score_rzg = cand["score_rzg"].to_numpy(dtype=float)
    stack = np.stack([score_sbdk, score_sbgk, score_rzg], axis=1)
    best = stack.argmax(axis=1)
    cand["score_total"] = stack.max(axis=1)
    cand["score_path"] = np.where(best == 0, "sbdk", np.where(best == 1, "sbgk", "rzg"))

    # Rank candidates and select the final Top-K list.
    ranked = cand.sort_values(["score_total", "auction_vol_ratio", "amount_cny_prev"], ascending=[False, False, False], kind="stable")
    topk, filtered = _choose_topk(
        ranked,
        topk_k=config.topk_k,
        filter_limit_up_at_open=config.filter_limit_up_at_open,
        score_threshold=config.score_threshold,
        dynamic_k_by_score=config.dynamic_k_by_score,
    )

    # Add bookkeeping columns for report-level attribution.
    cand["trade_date"] = int(d)
    topk["trade_date"] = int(d)

    return DaySignals(candidates=cand.reset_index(drop=True), topk=topk.reset_index(drop=True), filtered_limit_up_at_open=int(filtered))


def select_topk_from_candidates(
    candidates: pd.DataFrame,
    topk_k: int,
    filter_limit_up_at_open: bool,
    score_threshold: float,
    dynamic_k_by_score: bool,
) -> tuple[pd.DataFrame, int]:
    """Select Top-K rows from a candidates table with the same ranking rule as build_day_signals."""
    # Rank candidates by the stable sorting rule.
    ranked = candidates.sort_values(["score_total", "auction_vol_ratio", "amount_cny_prev"], ascending=[False, False, False], kind="stable")

    # Select Top-K after applying threshold and tradability filters.
    return _choose_topk(
        ranked,
        topk_k=topk_k,
        filter_limit_up_at_open=filter_limit_up_at_open,
        score_threshold=score_threshold,
        dynamic_k_by_score=dynamic_k_by_score,
    )
