"""Simulate a portfolio with minute-level exits and daily Top-K entries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from shouban2.firstboard_topk_v1.config import StrategyConfig
from shouban2.firstboard_topk_v1.io_market import read_stock1m_subset


@dataclass(frozen=True)
class Position:
    """Represent one open position in the portfolio."""

    StockCode: int
    ts_code: str
    shares: float
    cost: float


def _bps_to_rate(bps: float) -> float:
    """Convert bps into a multiplicative rate."""
    # Convert basis points into a decimal.
    return float(bps) / 10000.0


def _sell_fee_rate(config: StrategyConfig) -> float:
    """Compute total sell fee rate from commission and stamp duty."""
    # Aggregate sell fees into a single decimal rate.
    return _bps_to_rate(config.commission_bps + config.stamp_duty_bps)


def _buy_fee_rate(config: StrategyConfig) -> float:
    """Compute total buy fee rate from commission."""
    # Aggregate buy fees into a single decimal rate.
    return _bps_to_rate(config.commission_bps)


def _slip_rate(config: StrategyConfig) -> float:
    """Compute slippage rate from bps."""
    # Convert slippage bps to a decimal rate.
    return _bps_to_rate(config.slippage_bps)


def _load_day_minute_close(stock1m_root: str, trade_date: str, stock_codes: list[int]) -> pd.DataFrame:
    """Load minute Close/MinuteIndex for the needed StockCode subset on one day."""
    # Read only the minimal columns for exit checks.
    cols = ["StockCode", "MinuteIndex", "Close"]
    df = read_stock1m_subset(trade_date=trade_date, stock1m_root=stock1m_root, stock_codes=stock_codes, columns=cols)
    df["StockCode"] = df["StockCode"].astype(int)
    df["MinuteIndex"] = df["MinuteIndex"].astype(int)
    return df.sort_values(["StockCode", "MinuteIndex"], kind="stable").reset_index(drop=True)


def simulate_backtest(
    config: StrategyConfig,
    daily_panel: pl.DataFrame,
    trade_dates: list[str],
    topk_by_date: dict[str, pd.DataFrame],
    out_trades_path: str,
    out_nav_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a backtest with minute-level exits and daily Top-K entries."""
    # Prepare daily lookup tables for fast access to close/up_limit/ma5 per day.
    daily_pd = daily_panel.select(["StockCode", "trade_date", "close", "open", "up_limit", "ma5_close", "ts_code"]).to_pandas()
    daily_pd["StockCode"] = daily_pd["StockCode"].astype(int)
    daily_pd["trade_date"] = daily_pd["trade_date"].astype(int)
    daily_pd = daily_pd.sort_values(["trade_date", "StockCode"], kind="stable").reset_index(drop=True)

    # Initialize portfolio state and output buffers.
    cash = float(config.initial_cash)
    holdings: dict[int, Position] = {}
    trade_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []

    buy_fee = _buy_fee_rate(config)
    sell_fee = _sell_fee_rate(config)
    slip = _slip_rate(config)

    for d in trade_dates:
        # Resolve day-level fields for existing holdings.
        d_int = int(d)
        day_slice = daily_pd.loc[daily_pd["trade_date"] == d_int]
        day_by_code = {int(k): r for k, r in day_slice.set_index("StockCode").iterrows()}

        # Load minute closes for held stocks to evaluate exits.
        held_codes = list(holdings.keys())
        if len(held_codes) > 0:
            minute_df = _load_day_minute_close(config.stock1m_root, d, held_codes)
            by_code = {int(k): g for k, g in minute_df.groupby("StockCode", sort=False)}

            # Apply exit rules and generate sell trades.
            for stock_code in held_codes:
                # Resolve per-stock day limits and MA5 threshold.
                day_row = day_by_code[int(stock_code)]
                up_limit = float(day_row["up_limit"])
                ma5 = float(day_row["ma5_close"])

                # Scan minute closes to find the earliest exit trigger.
                g = by_code[int(stock_code)]
                close = g["Close"].to_numpy(dtype=float)
                minute = g["MinuteIndex"].to_numpy(dtype=int)

                pos = holdings[int(stock_code)]
                take_profit = (close < up_limit) & (close > pos.cost)
                stop_loss = close < ma5
                trigger = take_profit | stop_loss

                if bool(np.any(trigger)):
                    first_i = int(np.argmax(trigger))
                    sell_px = float(close[first_i]) * (1.0 - slip)
                    proceeds = float(pos.shares) * sell_px
                    fee = proceeds * sell_fee
                    cash += proceeds - fee

                    trade_rows.append(
                        {
                            "trade_date": d_int,
                            "StockCode": int(stock_code),
                            "ts_code": pos.ts_code,
                            "side": "SELL",
                            "minute_index": int(minute[first_i]),
                            "px": float(sell_px),
                            "shares": float(pos.shares),
                            "gross": float(proceeds),
                            "fee": float(fee),
                            "reason": "TP" if bool(take_profit[first_i]) else "SL",
                        }
                    )
                    del holdings[int(stock_code)]

        # Execute daily Top-K buys at open for non-held names.
        day_topk = topk_by_date.get(d)
        if day_topk is not None and int(day_topk.shape[0]) > 0:
            # Build a list of buy candidates that are not already held.
            buy_rows = []
            for _, row in day_topk.iterrows():
                stock_code = int(row["StockCode"])
                if stock_code in holdings:
                    continue
                buy_rows.append(row)

            # Allocate cash equally across new buys.
            if len(buy_rows) > 0:
                cash_per = cash / float(len(buy_rows))
                for row in buy_rows:
                    stock_code = int(row["StockCode"])
                    day_row = day_by_code[int(stock_code)]
                    buy_px = float(day_row["open"]) * (1.0 + slip)
                    shares = cash_per / (buy_px * (1.0 + buy_fee))
                    gross = float(shares) * buy_px
                    fee = gross * buy_fee
                    cash -= gross + fee

                    holdings[int(stock_code)] = Position(
                        StockCode=int(stock_code),
                        ts_code=str(day_row["ts_code"]),
                        shares=float(shares),
                        cost=float(buy_px),
                    )
                    trade_rows.append(
                        {
                            "trade_date": d_int,
                            "StockCode": int(stock_code),
                            "ts_code": str(day_row["ts_code"]),
                            "side": "BUY",
                            "minute_index": 0,
                            "px": float(buy_px),
                            "shares": float(shares),
                            "gross": float(gross),
                            "fee": float(fee),
                            "reason": "TOPK",
                        }
                    )

        # Mark-to-market NAV at close for reporting.
        holdings_value = 0.0
        for stock_code, pos in holdings.items():
            day_row = day_by_code[int(stock_code)]
            holdings_value += float(pos.shares) * float(day_row["close"])

        nav = cash + holdings_value
        nav_rows.append(
            {
                "trade_date": d_int,
                "nav": float(nav),
                "cash": float(cash),
                "holdings": float(holdings_value),
                "n_pos": float(len(holdings)),
            }
        )

    # Write output tables to disk for webapp and audit.
    trades_df = pd.DataFrame(trade_rows)
    nav_df = pd.DataFrame(nav_rows)
    trades_df.reset_index(drop=True).to_feather(out_trades_path)
    nav_df.reset_index(drop=True).to_feather(out_nav_path)

    return nav_df, trades_df


def simulate_backtest_t1_open(
    config: StrategyConfig,
    daily_panel: pl.DataFrame,
    trade_dates: list[str],
    topk_by_date: dict[str, pd.DataFrame],
    out_trades_path: str,
    out_nav_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a baseline backtest that buys at open and sells at next-day open."""
    # Prepare daily lookup tables for open/close and identifiers.
    daily_pd = daily_panel.select(["StockCode", "trade_date", "open", "close", "ts_code"]).to_pandas()
    daily_pd["StockCode"] = daily_pd["StockCode"].astype(int)
    daily_pd["trade_date"] = daily_pd["trade_date"].astype(int)
    daily_pd = daily_pd.sort_values(["trade_date", "StockCode"], kind="stable").reset_index(drop=True)

    # Initialize portfolio state and output buffers.
    cash = float(config.initial_cash)
    holdings: dict[int, Position] = {}
    trade_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []

    # Pre-compute fee and slippage rates for consistent application.
    buy_fee = _buy_fee_rate(config)
    sell_fee = _sell_fee_rate(config)
    slip = _slip_rate(config)

    for i, d in enumerate(trade_dates):
        # Resolve day-level fields for selling/buying and NAV marking.
        d_int = int(d)
        day_slice = daily_pd.loc[daily_pd["trade_date"] == d_int]
        day_by_code = {int(k): r for k, r in day_slice.set_index("StockCode").iterrows()}

        # Sell all holdings at today's open (these were bought on the previous day).
        if len(holdings) > 0:
            for stock_code, pos in list(holdings.items()):
                # Resolve today's open price for selling.
                day_row = day_by_code[int(stock_code)]
                sell_px = float(day_row["open"]) * (1.0 - slip)
                proceeds = float(pos.shares) * sell_px
                fee = proceeds * sell_fee
                cash += proceeds - fee

                trade_rows.append(
                    {
                        "trade_date": d_int,
                        "StockCode": int(stock_code),
                        "ts_code": pos.ts_code,
                        "side": "SELL",
                        "minute_index": 0,
                        "px": float(sell_px),
                        "shares": float(pos.shares),
                        "gross": float(proceeds),
                        "fee": float(fee),
                        "reason": "T1_OPEN",
                    }
                )
                del holdings[int(stock_code)]

        # Skip opening new positions on the last trade date to keep liquidation well-defined.
        if i < len(trade_dates) - 1:
            # Execute daily Top-K buys at open for non-held names.
            day_topk = topk_by_date.get(d)
            if day_topk is not None and int(day_topk.shape[0]) > 0:
                # Build a list of buy candidates that are not already held.
                buy_rows = []
                for _, row in day_topk.iterrows():
                    stock_code = int(row["StockCode"])
                    if stock_code in holdings:
                        continue
                    buy_rows.append(row)

                # Allocate cash equally across new buys.
                if len(buy_rows) > 0:
                    cash_per = cash / float(len(buy_rows))
                    for row in buy_rows:
                        # Resolve today's open price for buying.
                        stock_code = int(row["StockCode"])
                        day_row = day_by_code[int(stock_code)]
                        buy_px = float(day_row["open"]) * (1.0 + slip)
                        shares = cash_per / (buy_px * (1.0 + buy_fee))
                        gross = float(shares) * buy_px
                        fee = gross * buy_fee
                        cash -= gross + fee

                        holdings[int(stock_code)] = Position(
                            StockCode=int(stock_code),
                            ts_code=str(day_row["ts_code"]),
                            shares=float(shares),
                            cost=float(buy_px),
                        )
                        trade_rows.append(
                            {
                                "trade_date": d_int,
                                "StockCode": int(stock_code),
                                "ts_code": str(day_row["ts_code"]),
                                "side": "BUY",
                                "minute_index": 0,
                                "px": float(buy_px),
                                "shares": float(shares),
                                "gross": float(gross),
                                "fee": float(fee),
                                "reason": "TOPK",
                            }
                        )

        # Mark-to-market NAV at close for reporting.
        holdings_value = 0.0
        for stock_code, pos in holdings.items():
            # Resolve today's close price for marking.
            day_row = day_by_code[int(stock_code)]
            holdings_value += float(pos.shares) * float(day_row["close"])

        nav = cash + holdings_value
        nav_rows.append(
            {
                "trade_date": d_int,
                "nav": float(nav),
                "cash": float(cash),
                "holdings": float(holdings_value),
                "n_pos": float(len(holdings)),
            }
        )

    # Write output tables to disk for webapp and audit.
    trades_df = pd.DataFrame(trade_rows)
    nav_df = pd.DataFrame(nav_rows)
    trades_df.reset_index(drop=True).to_feather(out_trades_path)
    nav_df.reset_index(drop=True).to_feather(out_nav_path)

    return nav_df, trades_df
