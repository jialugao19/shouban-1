"""Build daily rolling features and event flags from stock1d."""

from __future__ import annotations

import polars as pl

from shouban2.firstboard_topk_v1.io_market import read_stock1d


AMOUNT_TO_CNY = 1000.0
VOL_TO_SHARES = 100.0
VWAP_MULTIPLIER = AMOUNT_TO_CNY / VOL_TO_SHARES


def prepare_daily_panel(
    trade_dates: list[str],
    stock1d_root: str,
    first_board_lookback: int,
    rp_lookback: int,
    ma5_lookback: int,
) -> pl.DataFrame:
    """Build a daily panel with rolling features and event flags."""
    # Load all daily files and concatenate into one long panel.
    daily_frames: list[pl.DataFrame] = []
    for d in trade_dates:
        df = read_stock1d(trade_date=d, stock1d_root=stock1d_root)
        pl_df = pl.from_pandas(df).select(
            [
                "ts_code",
                pl.col("trade_date").cast(pl.Int32),
                pl.col("ts_code").str.slice(0, 6).cast(pl.Int32).alias("StockCode"),
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
        )
        daily_frames.append(pl_df)
    daily = pl.concat(daily_frames, how="vertical").sort(["StockCode", "trade_date"])

    # Compute event flags from the daily OHLC and limit prices.
    daily = daily.with_columns(
        [
            (pl.col("high") == pl.col("up_limit")).alias("touch_limit"),
            (pl.col("close") == pl.col("up_limit")).alias("limit_close"),
        ]
    ).with_columns(((pl.col("touch_limit") & (pl.col("close") < pl.col("up_limit"))).alias("touch_not_close")))

    # Compute first-board flags by checking no touch_limit in the past N days.
    prev_touch_sum = (
        pl.col("touch_limit").cast(pl.Int32).shift(1).rolling_sum(first_board_lookback).over("StockCode").alias("_prev_touch_sum")
    )
    daily = daily.with_columns(prev_touch_sum).with_columns(
        [
            (pl.col("touch_limit") & (pl.col("_prev_touch_sum") == 0)).alias("first_board"),
            (pl.col("limit_close") & (pl.col("_prev_touch_sum") == 0)).alias("first_straight_board"),
            (pl.col("touch_not_close") & (pl.col("_prev_touch_sum") == 0)).alias("first_touch_not_close"),
        ]
    )

    # Compute VWAP proxy, MA5, and lagged fields used by rzg features.
    daily = daily.with_columns(((pl.col("amount") / pl.col("vol")) * VWAP_MULTIPLIER).alias("vwap"))
    daily = daily.with_columns(pl.col("close").shift(1).rolling_mean(ma5_lookback).over("StockCode").alias("ma5_close"))
    daily = daily.with_columns(pl.col("close").shift(3).over("StockCode").alias("close_lag3"))
    daily = daily.with_columns(pl.col("vol").shift(1).over("StockCode").alias("vol_lag1"))

    # Compute rp using rolling highs/lows.
    hi = pl.col("high").rolling_max(rp_lookback).over("StockCode").alias("_hi_rp")
    lo = pl.col("low").rolling_min(rp_lookback).over("StockCode").alias("_lo_rp")
    daily = daily.with_columns([hi, lo]).with_columns(
        ((pl.col("close") - pl.col("_lo_rp")) / (pl.col("_hi_rp") - pl.col("_lo_rp"))).alias("rp")
    )

    # Drop intermediate columns to keep the panel compact.
    return daily.drop(["_prev_touch_sum", "_hi_rp", "_lo_rp"])

