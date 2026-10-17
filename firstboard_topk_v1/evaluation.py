"""Summarize strategy results and generate report assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _compute_drawdown(nav: pd.Series) -> pd.Series:
    """Compute drawdown series from a NAV series."""
    # Compute running peak and convert into drawdown.
    peak = nav.cummax()
    return nav / peak - 1.0


def evaluate_nav(nav_df: pd.DataFrame) -> dict[str, float]:
    """Compute annualized metrics for a daily NAV series."""
    # Compute daily returns and basic statistics.
    nav = nav_df.sort_values("trade_date")["nav"].astype(float)
    rets = nav.pct_change().dropna()

    # Compute annualized return and volatility.
    ann_return = float((nav.iloc[-1] / nav.iloc[0]) ** (252.0 / float(len(rets))) - 1.0)
    ann_vol = float(rets.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0.0 else float("nan")

    # Compute drawdown-based metrics.
    dd = _compute_drawdown(nav)
    mdd = float(dd.min())
    calmar = float(ann_return / abs(mdd)) if mdd < 0.0 else float("nan")

    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd, "calmar": calmar}


def summarize_picks(topk_by_date: dict[str, pd.DataFrame], trade_dates: list[str]) -> dict[str, object]:
    """Compute coverage and concentration statistics for daily Top-K picks."""
    # Count daily picks and locate the maximum pick day.
    counts: list[int] = []
    gross_exposures: list[float] = []
    max_day = ""
    max_picks = -1
    for d in trade_dates:
        df = topk_by_date.get(d)
        n = int(df.shape[0]) if df is not None else 0
        counts.append(n)
        gross = float(df["position_weight"].astype(float).sum()) if df is not None and int(df.shape[0]) > 0 else 0.0
        gross_exposures.append(gross)
        if n > max_picks:
            max_picks = n
            max_day = d

    # Compute summary statistics for coverage and concentration.
    nonempty_days = int(sum(1 for n in counts if n > 0))
    total_picks = int(sum(counts))
    p99 = int(pd.Series(counts).quantile(0.99))
    avg_picks = float(total_picks / max(1, len(trade_dates)))

    # Compute the path mix for explainability when score_path exists.
    path_counts: dict[str, int] = {}
    for d in trade_dates:
        df = topk_by_date.get(d)
        if df is None or int(df.shape[0]) == 0:
            continue
        if "score_path" not in df.columns:
            continue
        for v in df["score_path"].astype(str).tolist():
            path_counts[str(v)] = int(path_counts.get(str(v), 0)) + 1
    path_mix = {k: float(v / max(1, total_picks)) for k, v in sorted(path_counts.items(), key=lambda x: x[0])}
    return {
        "nonempty_days": nonempty_days,
        "coverage": float(nonempty_days / max(1, len(trade_dates))),
        "total_picks": total_picks,
        "avg_picks": avg_picks,
        "avg_gross_exposure": float(pd.Series(gross_exposures, dtype=float).mean()),
        "p99_gross_exposure": float(pd.Series(gross_exposures, dtype=float).quantile(0.99)),
        "p99_day_picks": p99,
        "max_day_picks": int(max_picks),
        "max_day": str(max_day),
        "path_mix": path_mix,
        "per_day_picks": counts,
        "per_day_gross_exposure": gross_exposures,
    }


def summarize_limit_quality_groups(
    candidates_df: pd.DataFrame,
    day_open_by_date: dict[int, pd.Series],
    trade_dates: list[str],
    commission_bps: float,
    stamp_duty_bps: float,
    slippage_bps: float,
) -> list[dict[str, object]]:
    """Summarize T+1 open returns by limit-quality score buckets."""
    # Convert trading costs into decimal rates once.
    buy_fee = float(commission_bps) / 10000.0
    sell_fee = float(commission_bps + stamp_duty_bps) / 10000.0
    slip = float(slippage_bps) / 10000.0

    # Build per-candidate T+1 open returns across the evaluation window.
    rows: list[dict[str, float]] = []
    grouped = {int(d): g for d, g in candidates_df.groupby("trade_date", sort=False)}
    for i, d in enumerate(trade_dates[:-1]):
        trade_date = int(d)
        next_date = int(trade_dates[i + 1])
        day_candidates = grouped.get(trade_date)
        if day_candidates is None or int(day_candidates.shape[0]) == 0:
            continue
        next_open = day_open_by_date[next_date]
        for _, row in day_candidates.iterrows():
            stock_code = int(row["StockCode"])
            if stock_code not in next_open.index:
                continue
            buy_px = float(row["open"]) * (1.0 + slip)
            sell_px = float(next_open.loc[stock_code]) * (1.0 - slip)
            ret = (sell_px * (1.0 - sell_fee)) / (buy_px * (1.0 + buy_fee)) - 1.0
            rows.append({"score_limit_quality": float(row["score_limit_quality"]), "ret_t1_open": float(ret)})

    # Aggregate the returns by quantile bucket for diagnostics.
    df = pd.DataFrame(rows)
    buckets = pd.qcut(df["score_limit_quality"], 3, duplicates="drop")
    qtab = (
        df.assign(bucket=buckets.astype(str))
        .groupby("bucket", sort=True)
        .agg(
            count=("ret_t1_open", "size"),
            mean_ret=("ret_t1_open", "mean"),
            mean_score=("score_limit_quality", "mean"),
            min_score=("score_limit_quality", "min"),
            max_score=("score_limit_quality", "max"),
        )
        .reset_index()
    )
    return [
        {
            "bucket": str(row["bucket"]),
            "count": int(row["count"]),
            "mean_ret": float(row["mean_ret"]),
            "mean_score": float(row["mean_score"]),
            "min_score": float(row["min_score"]),
            "max_score": float(row["max_score"]),
        }
        for row in qtab.to_dict(orient="records")
    ]


def top10_contribution(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Compute top-10 daily PnL contribution days from NAV."""
    # Compute daily pnl contributions as nav diff.
    df = nav_df.sort_values("trade_date").copy()
    df["pnl"] = df["nav"].astype(float).diff()
    out = df.dropna().sort_values("pnl", ascending=False).head(10)[["trade_date", "pnl"]]
    out["trade_date"] = out["trade_date"].astype(int)
    return out.reset_index(drop=True)


def write_report(
    out_root: str,
    metrics: dict[str, float],
    pick_summary: dict[str, object],
    threshold_metrics: dict[str, float],
    threshold_pick_summary: dict[str, object],
    t0_metrics: dict[str, float],
    limit_quality_groups: list[dict[str, object]],
    top10_df: pd.DataFrame,
    max_day_topk: pd.DataFrame,
    filtered_limit_up_count: int,
) -> None:
    """Write evaluation/report.md with key metrics and postmortem scaffolding."""
    # Build report lines with a stable structure for webapp parsing.
    lines = [
        "# firstboard_topk_v1 evaluation",
        "",
        "## metrics_baseline_t1_open",
        "",
        f"- ann_return: {metrics['ann_return']:.6f}",
        f"- ann_vol: {metrics['ann_vol']:.6f}",
        f"- sharpe: {metrics['sharpe']:.6f}",
        f"- mdd: {metrics['mdd']:.6f}",
        f"- calmar: {metrics['calmar']:.6f}",
        "",
        "## picks_baseline",
        "",
        f"- nonempty_days: {int(pick_summary['nonempty_days'])}",
        f"- coverage: {float(pick_summary['coverage']):.6f}",
        f"- total_picks: {int(pick_summary['total_picks'])}",
        f"- avg_picks: {float(pick_summary['avg_picks']):.6f}",
        f"- avg_gross_exposure: {float(pick_summary['avg_gross_exposure']):.6f}",
        f"- p99_gross_exposure: {float(pick_summary['p99_gross_exposure']):.6f}",
        f"- p99_day_picks: {int(pick_summary['p99_day_picks'])}",
        f"- max_day_picks: {int(pick_summary['max_day_picks'])}",
        f"- max_day: {str(pick_summary['max_day'])}",
        f"- filtered_limit_up_at_open: {int(filtered_limit_up_count)}",
        f"- path_mix: {pick_summary['path_mix']}",
        "",
        "## metrics_score_threshold_t1_open",
        "",
        f"- ann_return: {threshold_metrics['ann_return']:.6f}",
        f"- ann_vol: {threshold_metrics['ann_vol']:.6f}",
        f"- sharpe: {threshold_metrics['sharpe']:.6f}",
        f"- mdd: {threshold_metrics['mdd']:.6f}",
        f"- calmar: {threshold_metrics['calmar']:.6f}",
        "",
        "## picks_score_threshold",
        "",
        f"- nonempty_days: {int(threshold_pick_summary['nonempty_days'])}",
        f"- coverage: {float(threshold_pick_summary['coverage']):.6f}",
        f"- total_picks: {int(threshold_pick_summary['total_picks'])}",
        f"- avg_picks: {float(threshold_pick_summary['avg_picks']):.6f}",
        f"- avg_gross_exposure: {float(threshold_pick_summary['avg_gross_exposure']):.6f}",
        f"- p99_gross_exposure: {float(threshold_pick_summary['p99_gross_exposure']):.6f}",
        f"- p99_day_picks: {int(threshold_pick_summary['p99_day_picks'])}",
        f"- max_day_picks: {int(threshold_pick_summary['max_day_picks'])}",
        f"- max_day: {str(threshold_pick_summary['max_day'])}",
        f"- path_mix: {threshold_pick_summary['path_mix']}",
        "",
        "## limit_quality_groups_t1_open",
        "",
        pd.DataFrame(limit_quality_groups).to_markdown(index=False),
        "",
        "## metrics_baseline_t0_open_close",
        "",
        f"- ann_return: {t0_metrics['ann_return']:.6f}",
        f"- ann_vol: {t0_metrics['ann_vol']:.6f}",
        f"- sharpe: {t0_metrics['sharpe']:.6f}",
        f"- mdd: {t0_metrics['mdd']:.6f}",
        f"- calmar: {t0_metrics['calmar']:.6f}",
        "",
        "## top10_pnl_days",
        "",
        top10_df.to_markdown(index=False),
        "",
        "## max_day_topk_detail",
        "",
        max_day_topk.to_markdown(index=False),
        "",
        "## next_steps",
        "",
        "- Check whether performance is dominated by a few days (see top10_pnl_days).",
        "- Compare score distributions between extreme days and normal days.",
        "- In experiments, vary one scoring component at a time and preserve full candidate panels for diffing.",
        "",
    ]

    # Write report.md under evaluation/.
    report_dir = Path(out_root) / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_structured_evaluation(
    out_root: str,
    eval_kind: str,
    run_meta: dict[str, object],
    metrics: dict[str, float],
    pick_summary: dict[str, object],
    threshold_metrics: dict[str, float],
    threshold_pick_summary: dict[str, object],
    t0_metrics: dict[str, float],
    limit_quality_groups: list[dict[str, object]],
) -> None:
    """Write structured YAML files under evaluation/ for regression checks."""
    # Create the evaluation directory for structured outputs.
    eval_dir = Path(out_root) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics.yaml for machine-readable comparisons.
    metrics_obj = {
        "baseline_t1_open": metrics,
        "score_threshold_t1_open": threshold_metrics,
        "baseline_t0_open_close": t0_metrics,
        "limit_quality_groups_t1_open": limit_quality_groups,
    }
    (eval_dir / "metrics.yaml").write_text(yaml.safe_dump(metrics_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Write picks.yaml for coverage, concentration, and path mix checks.
    picks_obj = {
        "baseline": pick_summary,
        "score_threshold": threshold_pick_summary,
    }
    (eval_dir / "picks.yaml").write_text(yaml.safe_dump(picks_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Write status.yaml for quick pass/fail gating and basic metadata.
    status_obj = {
        "status": "ok",
        "schema_version": 1,
        "eval_kind": str(eval_kind),
        "run_meta": dict(run_meta),
    }
    (eval_dir / "status.yaml").write_text(yaml.safe_dump(status_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def plot_assets(out_root: str, nav_df: pd.DataFrame, pick_summary: dict[str, object], candidates_df: pd.DataFrame) -> None:
    """Plot NAV, drawdown, pick hist, and score hist into assets/."""
    # Create the assets directory for plots.
    assets_dir = Path(out_root) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot normalized NAV curve.
    df = nav_df.sort_values("trade_date").copy()
    nav = df["nav"].astype(float).to_numpy()
    nav_norm = nav / float(nav[0])
    plt.figure(figsize=(11, 6))
    plt.plot(df["trade_date"].astype(int).to_numpy(), nav_norm, linewidth=1.6)
    plt.title("NAV (normalized)")
    plt.xlabel("trade_date")
    plt.ylabel("nav")
    plt.tight_layout()
    plt.savefig(assets_dir / "nav.png", dpi=160)
    plt.close()

    # Plot drawdown curve.
    dd = _compute_drawdown(pd.Series(nav_norm))
    plt.figure(figsize=(11, 5.5))
    plt.plot(df["trade_date"].astype(int).to_numpy(), dd.to_numpy(), linewidth=1.6)
    plt.title("Drawdown")
    plt.xlabel("trade_date")
    plt.ylabel("drawdown")
    plt.tight_layout()
    plt.savefig(assets_dir / "drawdown.png", dpi=160)
    plt.close()

    # Plot histogram of daily picks.
    counts = pd.Series(pick_summary["per_day_picks"], dtype=float)
    plt.figure(figsize=(10.5, 6))
    plt.hist(counts.values, bins=40, alpha=0.8)
    plt.title("Top-K picks per day (hist)")
    plt.xlabel("picks")
    plt.ylabel("days")
    plt.tight_layout()
    plt.savefig(assets_dir / "picks_hist.png", dpi=160)
    plt.close()

    # Plot score_total distribution for candidates.
    plt.figure(figsize=(10.5, 6))
    plt.hist(candidates_df["score_total"].astype(float).to_numpy(), bins=60, alpha=0.8)
    plt.title("score_total distribution (candidates)")
    plt.xlabel("score_total")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(assets_dir / "score_total_hist.png", dpi=160)
    plt.close()
