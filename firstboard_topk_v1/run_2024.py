"""Run firstboard_topk_v1 for the full year 2024 and write outputs to /data-cache."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from shouban2.firstboard_topk_v1.backtest import simulate_backtest_t1_open
from shouban2.firstboard_topk_v1.config import (
    AuctionShapeSbdk,
    AuctionShapeSbgk,
    LimitQualityParams,
    OpenBand,
    RzgWeights,
    SbdkWeights,
    SbgkWeights,
    StrengthRange,
    StrategyConfig,
    UpperShadowRange,
    VolShrinkageRange,
)
from shouban2.firstboard_topk_v1.evaluation import (
    evaluate_nav,
    plot_assets,
    summarize_picks,
    top10_contribution,
    write_report,
    write_structured_evaluation,
)
from shouban2.firstboard_topk_v1.features_daily import prepare_daily_panel
from shouban2.firstboard_topk_v1.io_trade_dates import load_open_trade_dates, load_open_trade_dates_upto
from shouban2.firstboard_topk_v1.signals import build_day_signals, select_topk_from_candidates


def _load_valid_by_date(valid_stocks_path: str, trade_dates: list[str]) -> dict[int, set[int]]:
    """Load valid_stocks into a per-date StockCode set mapping."""
    # Load only the needed date range and convert code->StockCode.
    start = int(trade_dates[0])
    end = int(trade_dates[-1])
    df = pl.scan_ipc(valid_stocks_path).filter((pl.col("date") >= start) & (pl.col("date") <= end)).collect()

    # Aggregate codes into per-date lists and build Python sets.
    grouped = df.group_by("date").agg((pl.col("code") - 1_000_000).cast(pl.Int32).alias("StockCodes"))
    out: dict[int, set[int]] = {}
    for row in grouped.iter_rows(named=True):
        out[int(row["date"])] = set(int(x) for x in row["StockCodes"])
    return out


def _write_yaml(path: str, obj: object) -> None:
    """Write an object as a YAML file with unicode enabled."""
    # Serialize using safe_dump to keep output audit-friendly.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run() -> None:
    """Run a full-year 2024 backtest and write all artifacts under /data-cache."""
    # Declare fixed run range for 2021-2024.
    start_date = "20210104"
    end_date = "20241231"

    # Declare data roots under /data.
    trade_cal_path = "/data/ashare/market/trade_cal.feather"
    stock1d_root = "/data/ashare/market/stock1d"
    stock1m_root = "/data/ashare/market/stock1m"
    valid_stocks_path = "/data/ashare/feature/valid_stocks.feather"

    # Declare output root under /data-cache.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_id = "031001"
    run_id = f"firstboard_topk_v1_2021_2024_{stamp}"
    out_root = f"/data-cache/shouban2/{test_id}/firstboard_topk_v1/runs/{run_id}"

    # Declare strategy configuration values (no implicit defaults).
    config = StrategyConfig(
        start_date=start_date,
        end_date=end_date,
        trade_cal_path=trade_cal_path,
        stock1d_root=stock1d_root,
        stock1m_root=stock1m_root,
        valid_stocks_path=valid_stocks_path,
        out_root=out_root,
        first_board_lookback=60,
        rp_lookback=60,
        ma5_lookback=5,
        topk_k=5,
        score_threshold=0.55,
        dynamic_k_by_score=False,
        filter_limit_up_at_open=True,
        commission_bps=3.0,
        stamp_duty_bps=10.0,
        slippage_bps=5.0,
        initial_cash=10_000_000.0,
        open_band_sbdk=OpenBand(low=0.955, high=0.97),
        open_band_sbgk=OpenBand(low=1.00, high=1.06),
        open_band_rzg=OpenBand(low=0.98, high=1.09),
        auction_shape_sbgk=AuctionShapeSbgk(t_min=0.01, t_hot=0.15, t_max=0.30),
        auction_shape_sbdk=AuctionShapeSbdk(t_best=0.01, t_max=0.08),
        strength_sbgk=StrengthRange(r0=0.07, r1=0.15),
        strength_rzg=StrengthRange(r0=-0.04, r1=0.06),
        upper_shadow_range=UpperShadowRange(s0=0.2, s1=0.6),
        vol_shrinkage_range=VolShrinkageRange(width=0.4),
        limit_quality_params=LimitQualityParams(seal_time_norm=60, open_board_vol_norm=0.2),
        weights_sbdk=SbdkWeights(w_open_band=0.25, w_auction=0.25, w_liquidity=0.15, w_rp=0.15, w_limit_quality=0.20),
        weights_sbgk=SbgkWeights(w_open_band=0.20, w_auction=0.25, w_liquidity=0.15, w_strength=0.20, w_limit_quality=0.20),
        weights_rzg=RzgWeights(w_open_band=0.15, w_auction=0.20, w_liquidity=0.10, w_strength=0.15, w_reversal=0.20, w_limit_quality=0.20),
    )

    # Prepare trade dates with warmup for rolling windows.
    warmup = max(config.first_board_lookback, config.rp_lookback, config.ma5_lookback) + 2
    all_dates_upto_end = load_open_trade_dates_upto(config.trade_cal_path, config.end_date)
    start_pos = all_dates_upto_end.index(config.start_date)
    end_pos = all_dates_upto_end.index(config.end_date)
    begin_pos = max(0, start_pos - warmup)

    trade_dates_all = all_dates_upto_end[begin_pos : end_pos + 1]
    trade_dates_eval = load_open_trade_dates(config.trade_cal_path, config.start_date, config.end_date)

    # Build the daily panel once for the whole run range.
    daily_panel = prepare_daily_panel(
        trade_dates=trade_dates_all,
        stock1d_root=config.stock1d_root,
        first_board_lookback=config.first_board_lookback,
        rp_lookback=config.rp_lookback,
        ma5_lookback=config.ma5_lookback,
    )

    # Load valid stock universe mapping once for the evaluation range.
    valid_by_date = _load_valid_by_date(config.valid_stocks_path, trade_dates_eval)

    # Build signals day by day and persist candidate/topk panels.
    signals_root = Path(config.out_root) / "signals"
    cand_root = signals_root / "daily_candidates"
    topk_root = signals_root / "daily_topk"
    cand_root.mkdir(parents=True, exist_ok=True)
    topk_root.mkdir(parents=True, exist_ok=True)

    topk_pd_by_date: dict[str, object] = {}
    topk_threshold_pd_by_date: dict[str, object] = {}
    filtered_limit_up_count = 0
    all_candidates: list[pl.DataFrame] = []

    pos_by_date = {d: i for i, d in enumerate(trade_dates_all)}
    for i, d in enumerate(trade_dates_eval):
        # Resolve prev_date from the full trade date list to ensure calendar consistency.
        pos = pos_by_date[d]
        prev_date = trade_dates_all[pos - 1]

        # Build signals and write per-day panels.
        out = build_day_signals(config, daily_panel, valid_by_date, trade_date=d, prev_date=prev_date)
        (cand_root / d[:4]).mkdir(parents=True, exist_ok=True)
        (topk_root / d[:4]).mkdir(parents=True, exist_ok=True)
        out.candidates.to_feather(str(cand_root / d[:4] / f"{d}.feather"))
        out.topk.to_feather(str(topk_root / d[:4] / f"{d}.feather"))

        # Maintain aggregated objects for evaluation.
        filtered_limit_up_count += int(out.filtered_limit_up_at_open)
        all_candidates.append(pl.from_pandas(out.candidates[["score_total"]]))

        # Keep topk mapping for backtest.
        topk_pd_by_date[d] = out.topk

        # Build a thresholded Top-K for report-level comparisons (do not persist).
        topk_threshold, _ = select_topk_from_candidates(
            candidates=out.candidates,
            topk_k=config.topk_k,
            filter_limit_up_at_open=config.filter_limit_up_at_open,
            score_threshold=config.score_threshold,
            dynamic_k_by_score=True,
        )
        topk_threshold_pd_by_date[d] = topk_threshold

        if (i + 1) % 20 == 0:
            print(f"[signals] built {i+1}/{len(trade_dates_eval)} days")

    # Persist config for reproducibility.
    _write_yaml(f"{config.out_root}/config.yaml", asdict(config))

    # Run the backtest and persist nav/trades.
    bt_root = Path(config.out_root) / "backtest"
    bt_root.mkdir(parents=True, exist_ok=True)
    nav_path = str(bt_root / "nav.feather")
    trades_path = str(bt_root / "trades.feather")
    nav_df, _ = simulate_backtest_t1_open(
        config=config,
        daily_panel=daily_panel,
        trade_dates=trade_dates_eval,
        topk_by_date={d: topk_pd_by_date[d] for d in trade_dates_eval if d in topk_pd_by_date},
        out_trades_path=trades_path,
        out_nav_path=nav_path,
    )

    # Compute evaluation summaries and write report/plots.
    metrics = evaluate_nav(nav_df)
    pick_summary = summarize_picks({d: topk_pd_by_date.get(d) for d in trade_dates_eval}, trade_dates_eval)

    # Compute a T0 open->close diagnostic NAV series using daily topk tables.
    buy_fee = float(config.commission_bps) / 10000.0
    sell_fee = float(config.commission_bps + config.stamp_duty_bps) / 10000.0
    slip = float(config.slippage_bps) / 10000.0
    nav = float(config.initial_cash)
    t0_rows: list[dict[str, object]] = []
    for d in trade_dates_eval:
        # Compute equal-weight per-day return from open to close with costs.
        topk = topk_pd_by_date.get(d)
        if topk is None or int(topk.shape[0]) == 0:
            day_ret = 0.0
        else:
            open_px = topk["open"].astype(float).to_numpy()
            close_px = topk["close"].astype(float).to_numpy()
            buy_px = open_px * (1.0 + slip)
            sell_px = close_px * (1.0 - slip)
            ret = (sell_px * (1.0 - sell_fee)) / (buy_px * (1.0 + buy_fee)) - 1.0
            day_ret = float(ret.mean())

        # Compound the diagnostic NAV and persist in a nav-like table.
        nav = nav * (1.0 + day_ret)
        t0_rows.append({"trade_date": int(d), "nav": float(nav)})
    t0_nav_df = pd.DataFrame(t0_rows)
    t0_metrics = evaluate_nav(t0_nav_df)

    # Compute threshold Top-K metrics for comparisons.
    threshold_nav_df, _ = simulate_backtest_t1_open(
        config=config,
        daily_panel=daily_panel,
        trade_dates=trade_dates_eval,
        topk_by_date={d: topk_threshold_pd_by_date[d] for d in trade_dates_eval if d in topk_threshold_pd_by_date},
        out_trades_path=str(bt_root / "trades_threshold_tmp.feather"),
        out_nav_path=str(bt_root / "nav_threshold_tmp.feather"),
    )
    threshold_metrics = evaluate_nav(threshold_nav_df)
    threshold_pick_summary = summarize_picks({d: topk_threshold_pd_by_date.get(d) for d in trade_dates_eval}, trade_dates_eval)

    # Remove temp backtest files to keep the run contract stable.
    Path(bt_root / "trades_threshold_tmp.feather").unlink()
    Path(bt_root / "nav_threshold_tmp.feather").unlink()
    top10_df = top10_contribution(nav_df)
    max_day = str(pick_summary["max_day"])
    max_day_topk = topk_pd_by_date[max_day][
        ["StockCode", "ts_code", "score_total", "score_path", "auction_vol_ratio", "amount_cny_prev", "limit_up_at_open"]
    ].copy()

    write_report(
        out_root=config.out_root,
        metrics=metrics,
        pick_summary=pick_summary,
        threshold_metrics=threshold_metrics,
        threshold_pick_summary=threshold_pick_summary,
        t0_metrics=t0_metrics,
        top10_df=top10_df,
        max_day_topk=max_day_topk,
        filtered_limit_up_count=filtered_limit_up_count,
    )
    write_structured_evaluation(
        out_root=config.out_root,
        eval_kind="backtest",
        run_meta={
            "test_id": test_id,
            "strategy": "firstboard_topk_v1",
            "date_range": {"start_date": start_date, "end_date": end_date},
        },
        metrics=metrics,
        pick_summary=pick_summary,
        threshold_metrics=threshold_metrics,
        threshold_pick_summary=threshold_pick_summary,
        t0_metrics=t0_metrics,
    )

    candidates_df = pd.concat([x.to_pandas() for x in all_candidates], ignore_index=True) if len(all_candidates) > 0 else pd.DataFrame()
    plot_assets(config.out_root, nav_df, pick_summary, candidates_df)

    print(f"[done] out_root={config.out_root}")


if __name__ == "__main__":
    run()
