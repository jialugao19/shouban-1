"""Run experiments defined in a fixed spec.yaml under /data-cache and write summary artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
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
from shouban2.firstboard_topk_v1.evaluation import evaluate_nav, summarize_picks, top10_contribution, write_report, write_structured_evaluation
from shouban2.firstboard_topk_v1.evaluation import summarize_limit_quality_groups
from shouban2.firstboard_topk_v1.features_daily import prepare_daily_panel
from shouban2.firstboard_topk_v1.io_trade_dates import load_open_trade_dates, load_open_trade_dates_upto
from shouban2.firstboard_topk_v1.run_2024 import _load_valid_by_date, _write_yaml
from shouban2.firstboard_topk_v1.signals import build_day_signals, select_topk_from_candidates


BATCH_ID = "example_batch_id"
TEST_ID = "031001"


def _load_yaml(path: Path) -> object:
    """Load a YAML file as a Python object."""
    # Read YAML text and parse with safe_load.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _deep_update(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Recursively merge overrides into base and return a new dict."""
    # Copy base so the caller keeps immutability.
    out: dict[str, object] = {k: v for k, v in base.items()}

    # Merge nested dicts key by key.
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _strategy_config_from_dict(data: dict[str, object]) -> StrategyConfig:
    """Build StrategyConfig from a nested dict that matches `dataclasses.asdict` output."""
    # Construct nested dataclasses explicitly to keep types stable.
    open_band_sbdk = OpenBand(**data["open_band_sbdk"])
    open_band_sbgk = OpenBand(**data["open_band_sbgk"])
    open_band_rzg = OpenBand(**data["open_band_rzg"])

    auction_shape_sbgk = AuctionShapeSbgk(**data["auction_shape_sbgk"])
    auction_shape_sbdk = AuctionShapeSbdk(**data["auction_shape_sbdk"])

    strength_sbgk = StrengthRange(**data["strength_sbgk"])
    strength_rzg = StrengthRange(**data["strength_rzg"])

    upper_shadow_range = UpperShadowRange(**data["upper_shadow_range"])
    vol_shrinkage_range = VolShrinkageRange(**data["vol_shrinkage_range"])

    limit_quality_params = LimitQualityParams(**data["limit_quality_params"])

    weights_sbdk = SbdkWeights(**data["weights_sbdk"])
    weights_sbgk = SbgkWeights(**data["weights_sbgk"])
    weights_rzg = RzgWeights(**data["weights_rzg"])

    # Build the final StrategyConfig.
    return StrategyConfig(
        start_date=str(data["start_date"]),
        end_date=str(data["end_date"]),
        trade_cal_path=str(data["trade_cal_path"]),
        stock1d_root=str(data["stock1d_root"]),
        stock1m_root=str(data["stock1m_root"]),
        valid_stocks_path=str(data["valid_stocks_path"]),
        out_root=str(data["out_root"]),
        first_board_lookback=int(data["first_board_lookback"]),
        rp_lookback=int(data["rp_lookback"]),
        ma5_lookback=int(data["ma5_lookback"]),
        topk_k=int(data["topk_k"]),
        score_threshold=float(data["score_threshold"]),
        dynamic_k_by_score=bool(data["dynamic_k_by_score"]),
        filter_limit_up_at_open=bool(data["filter_limit_up_at_open"]),
        path_cap_sbdk=int(data["path_cap_sbdk"]),
        path_cap_sbgk=int(data["path_cap_sbgk"]),
        path_cap_rzg=int(data["path_cap_rzg"]),
        position_size_by_score=bool(data["position_size_by_score"]),
        daily_gross_exposure_cap=float(data["daily_gross_exposure_cap"]),
        commission_bps=float(data["commission_bps"]),
        stamp_duty_bps=float(data["stamp_duty_bps"]),
        slippage_bps=float(data["slippage_bps"]),
        initial_cash=float(data["initial_cash"]),
        open_band_sbdk=open_band_sbdk,
        open_band_sbgk=open_band_sbgk,
        open_band_rzg=open_band_rzg,
        auction_shape_sbgk=auction_shape_sbgk,
        auction_shape_sbdk=auction_shape_sbdk,
        strength_sbgk=strength_sbgk,
        strength_rzg=strength_rzg,
        upper_shadow_range=upper_shadow_range,
        vol_shrinkage_range=vol_shrinkage_range,
        limit_quality_params=limit_quality_params,
        weights_sbdk=weights_sbdk,
        weights_sbgk=weights_sbgk,
        weights_rzg=weights_rzg,
    )


def run() -> None:
    """Run a batch of experiments from a fixed spec.yaml path."""
    # Locate the batch root and spec.yaml under /data-cache.
    batch_root = Path("/data-cache/shouban2") / TEST_ID / "firstboard_topk_v1" / "experiments" / BATCH_ID
    spec_path = batch_root / "spec.yaml"
    spec = _load_yaml(spec_path)

    # Load base config and experiment list.
    base_dict = spec["base_config"]
    experiments = spec["experiments"]

    # Prepare a base StrategyConfig for shared preprocessing.
    base = _strategy_config_from_dict(base_dict)

    # Prepare trade dates with warmup for rolling windows.
    warmup = max(base.first_board_lookback, base.rp_lookback, base.ma5_lookback) + 2
    all_dates_upto_end = load_open_trade_dates_upto(base.trade_cal_path, base.end_date)
    start_pos = all_dates_upto_end.index(base.start_date)
    end_pos = all_dates_upto_end.index(base.end_date)
    begin_pos = max(0, start_pos - warmup)

    trade_dates_all = all_dates_upto_end[begin_pos : end_pos + 1]
    trade_dates_eval = load_open_trade_dates(base.trade_cal_path, base.start_date, base.end_date)
    pos_by_date = {d: i for i, d in enumerate(trade_dates_all)}

    # Build the daily panel once for the whole batch.
    daily_panel = prepare_daily_panel(
        trade_dates=trade_dates_all,
        stock1d_root=base.stock1d_root,
        first_board_lookback=base.first_board_lookback,
        rp_lookback=base.rp_lookback,
        ma5_lookback=base.ma5_lookback,
    )

    # Load valid stock universe mapping once for the evaluation range.
    valid_by_date = _load_valid_by_date(base.valid_stocks_path, trade_dates_eval)

    # Run each experiment and collect summary rows.
    summary_rows: list[dict[str, object]] = []
    for exp in experiments:
        # Build the experiment config by deep-merging overrides.
        exp_id = str(exp["exp_id"])
        overrides = exp["overrides"]
        merged = _deep_update(asdict(base), overrides)
        out_root = str(batch_root / "runs" / exp_id)
        merged["out_root"] = out_root
        cfg = _strategy_config_from_dict(merged)

        # Persist config.yaml for reproducibility.
        _write_yaml(f"{out_root}/config.yaml", asdict(cfg))

        # Build signals and keep a topk mapping for backtest.
        signals_root = Path(out_root) / "signals"
        cand_root = signals_root / "daily_candidates"
        topk_root = signals_root / "daily_topk"
        cand_root.mkdir(parents=True, exist_ok=True)
        topk_root.mkdir(parents=True, exist_ok=True)

        topk_pd_by_date: dict[str, pd.DataFrame] = {}
        topk_threshold_pd_by_date: dict[str, pd.DataFrame] = {}
        all_candidates: list[pd.DataFrame] = []
        filtered_limit_up_count = 0
        for d in trade_dates_eval:
            # Resolve prev_date from the full trade date list.
            prev_date = trade_dates_all[pos_by_date[d] - 1]

            # Build signals and persist per-day panels.
            out = build_day_signals(cfg, daily_panel, valid_by_date, trade_date=d, prev_date=prev_date)
            (cand_root / d[:4]).mkdir(parents=True, exist_ok=True)
            (topk_root / d[:4]).mkdir(parents=True, exist_ok=True)
            out.candidates.to_feather(str(cand_root / d[:4] / f"{d}.feather"))
            out.topk.to_feather(str(topk_root / d[:4] / f"{d}.feather"))

            filtered_limit_up_count += int(out.filtered_limit_up_at_open)
            topk_pd_by_date[d] = out.topk
            all_candidates.append(out.candidates[["trade_date", "StockCode", "open", "score_total", "score_limit_quality"]].copy())

            # Build a thresholded Top-K for report-level comparisons (do not persist).
            topk_threshold, _ = select_topk_from_candidates(
                candidates=out.candidates,
                topk_k=cfg.topk_k,
                filter_limit_up_at_open=cfg.filter_limit_up_at_open,
                score_threshold=cfg.score_threshold,
                dynamic_k_by_score=True,
                path_cap_sbdk=cfg.path_cap_sbdk,
                path_cap_sbgk=cfg.path_cap_sbgk,
                path_cap_rzg=cfg.path_cap_rzg,
                position_size_by_score=cfg.position_size_by_score,
                daily_gross_exposure_cap=cfg.daily_gross_exposure_cap,
            )
            topk_threshold_pd_by_date[d] = topk_threshold

        # Run backtest and write nav/trades.
        bt_root = Path(out_root) / "backtest"
        bt_root.mkdir(parents=True, exist_ok=True)
        nav_path = str(bt_root / "nav.feather")
        trades_path = str(bt_root / "trades.feather")
        nav_df, _ = simulate_backtest_t1_open(
            config=cfg,
            daily_panel=daily_panel,
            trade_dates=trade_dates_eval,
            topk_by_date=topk_pd_by_date,
            out_trades_path=trades_path,
            out_nav_path=nav_path,
        )

        # Evaluate and write report.md for the experiment run.
        metrics = evaluate_nav(nav_df)
        pick_summary = summarize_picks(topk_pd_by_date, trade_dates_eval)

        # Compute a T0 open->close diagnostic NAV series using daily topk tables.
        buy_fee = float(cfg.commission_bps) / 10000.0
        sell_fee = float(cfg.commission_bps + cfg.stamp_duty_bps) / 10000.0
        slip = float(cfg.slippage_bps) / 10000.0
        nav = float(cfg.initial_cash)
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

        # Build day-open lookup tables for cross-sectional diagnostics.
        daily_open_pd = daily_panel.select(["StockCode", "trade_date", "open"]).to_pandas()
        daily_open_pd["StockCode"] = daily_open_pd["StockCode"].astype(int)
        daily_open_pd["trade_date"] = daily_open_pd["trade_date"].astype(int)
        daily_open_pd = daily_open_pd.sort_values(["trade_date", "StockCode"], kind="stable").reset_index(drop=True)
        day_open_by_date = {int(d): g.set_index("StockCode")["open"] for d, g in daily_open_pd.groupby("trade_date", sort=False)}
        candidates_eval_df = pd.concat(all_candidates, ignore_index=True)
        limit_quality_groups = summarize_limit_quality_groups(
            candidates_df=candidates_eval_df,
            day_open_by_date=day_open_by_date,
            trade_dates=trade_dates_eval,
            commission_bps=cfg.commission_bps,
            stamp_duty_bps=cfg.stamp_duty_bps,
            slippage_bps=cfg.slippage_bps,
        )

        # Compute threshold Top-K metrics for comparisons.
        threshold_bt_root = Path("/tmp") / "shouban2_firstboard_topk_v1_threshold_tmp"
        threshold_bt_root.mkdir(parents=True, exist_ok=True)
        threshold_nav_path = str(threshold_bt_root / "nav.feather")
        threshold_trades_path = str(threshold_bt_root / "trades.feather")
        threshold_nav_df, _ = simulate_backtest_t1_open(
            config=cfg,
            daily_panel=daily_panel,
            trade_dates=trade_dates_eval,
            topk_by_date=topk_threshold_pd_by_date,
            out_trades_path=threshold_trades_path,
            out_nav_path=threshold_nav_path,
        )
        threshold_metrics = evaluate_nav(threshold_nav_df)
        threshold_pick_summary = summarize_picks(topk_threshold_pd_by_date, trade_dates_eval)
        Path(threshold_nav_path).unlink()
        Path(threshold_trades_path).unlink()
        threshold_bt_root.rmdir()

        top10_df = top10_contribution(nav_df)
        max_day = str(pick_summary["max_day"])
        max_day_topk = topk_pd_by_date[max_day][
            [
                "StockCode",
                "ts_code",
                "score_total",
                "score_path",
                "position_weight",
                "score_limit_quality",
                "sealed_minutes_total",
                "open_board_count_post_first_seal",
                "tail_seal_flag",
                "auction_vol_ratio",
                "amount_cny_prev",
                "limit_up_at_open",
            ]
        ].copy()
        write_report(
            out_root=out_root,
            metrics=metrics,
            pick_summary=pick_summary,
            threshold_metrics=threshold_metrics,
            threshold_pick_summary=threshold_pick_summary,
            t0_metrics=t0_metrics,
            limit_quality_groups=limit_quality_groups,
            top10_df=top10_df,
            max_day_topk=max_day_topk,
            filtered_limit_up_count=filtered_limit_up_count,
        )
        write_structured_evaluation(
            out_root=out_root,
            eval_kind="backtest",
            run_meta={
                "test_id": TEST_ID,
                "strategy": "firstboard_topk_v1",
                "batch_id": BATCH_ID,
                "exp_id": exp_id,
                "date_range": {"start_date": cfg.start_date, "end_date": cfg.end_date},
            },
            metrics=metrics,
            pick_summary=pick_summary,
            threshold_metrics=threshold_metrics,
            threshold_pick_summary=threshold_pick_summary,
            t0_metrics=t0_metrics,
            limit_quality_groups=limit_quality_groups,
        )

        # Compute score quantile returns for the baseline Top-K (T+1 open, gross of allocation effects).
        t1_rows: list[dict[str, object]] = []
        daily_pd = daily_panel.select(["StockCode", "trade_date", "open"]).to_pandas()
        daily_pd["StockCode"] = daily_pd["StockCode"].astype(int)
        daily_pd["trade_date"] = daily_pd["trade_date"].astype(int)
        daily_pd = daily_pd.sort_values(["trade_date", "StockCode"], kind="stable").reset_index(drop=True)
        day_open_by_date = {int(d): g.set_index("StockCode") for d, g in daily_pd.groupby("trade_date", sort=False)}
        for i, d in enumerate(trade_dates_eval):
            # Skip the last day because there is no next-day open for liquidation.
            if i >= len(trade_dates_eval) - 1:
                continue
            topk = topk_pd_by_date.get(d)
            if topk is None or int(topk.shape[0]) == 0:
                continue
            next_d = int(trade_dates_eval[i + 1])
            next_open = day_open_by_date[next_d]["open"]
            for _, r in topk.iterrows():
                # Compute per-pick open->next open return with costs.
                stock_code = int(r["StockCode"])
                buy_open = float(r["open"])
                sell_open = float(next_open.loc[stock_code])
                buy_px = buy_open * (1.0 + slip)
                sell_px = sell_open * (1.0 - slip)
                ret = (sell_px * (1.0 - sell_fee)) / (buy_px * (1.0 + buy_fee)) - 1.0
                t1_rows.append({"score_total": float(r["score_total"]), "ret_t1_open": float(ret)})
        t1_df = pd.DataFrame(t1_rows)
        q = pd.qcut(t1_df["score_total"], 5, duplicates="drop")
        qtab = (
            t1_df.assign(bucket=q.astype(str))
            .groupby("bucket", sort=True)
            .agg(count=("ret_t1_open", "size"), mean_ret=("ret_t1_open", "mean"))
            .reset_index()
        )
        score_quantile_ret = [{"bucket": str(r["bucket"]), "count": int(r["count"]), "mean_ret": float(r["mean_ret"])} for r in qtab.to_dict(orient="records")]

        # Collect summary row for batch-level report.
        summary_rows.append(
            {
                "exp_id": exp_id,
                "ann_return": float(metrics["ann_return"]),
                "ann_vol": float(metrics["ann_vol"]),
                "sharpe": float(metrics["sharpe"]),
                "mdd": float(metrics["mdd"]),
                "calmar": float(metrics["calmar"]),
                "coverage": float(pick_summary["coverage"]),
                "nonempty_days": int(pick_summary["nonempty_days"]),
                "avg_picks": float(pick_summary["avg_picks"]),
                "avg_gross_exposure": float(pick_summary["avg_gross_exposure"]),
                "p99_day_picks": int(pick_summary["p99_day_picks"]),
                "max_day_picks": int(pick_summary["max_day_picks"]),
                "max_day": str(pick_summary["max_day"]),
                "filtered_limit_up_at_open": int(filtered_limit_up_count),
                "path_mix": dict(pick_summary["path_mix"]),
                "limit_quality_groups": limit_quality_groups,
                "score_quantile_ret": score_quantile_ret,
            }
        )

    # Persist batch summary.yaml and a compact report.md.
    batch_root.mkdir(parents=True, exist_ok=True)
    _write_yaml(f"{batch_root}/summary.yaml", {"batch_id": BATCH_ID, "summary": summary_rows})

    df = pd.DataFrame(summary_rows).sort_values(["sharpe", "ann_return"], ascending=[False, False])
    report_lines = [
        f"# firstboard_topk_v1 experiments report: {BATCH_ID}",
        "",
        df.to_markdown(index=False),
        "",
        "## notes",
        "",
        "- Each experiment run writes its own `runs/<exp_id>/evaluation/report.md` for drill-down.",
        "- Use the webapp to compare experiments and inspect extreme days.",
        "",
    ]
    (batch_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[done] batch_root={batch_root}")


if __name__ == "__main__":
    run()
