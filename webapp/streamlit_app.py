"""Streamlit app to browse runs/experiments under /data-cache for firstboard_topk_v1."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml


ROOT = Path("/data-cache/shouban2")


def _load_yaml(path: Path) -> object:
    """Load a YAML file as a Python object."""
    # Read YAML text and parse with safe_load.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _list_test_ids() -> list[str]:
    """List available test_id directories under /data-cache/shouban2."""
    # Enumerate test folders that contain firstboard_topk_v1 outputs.
    out: list[str] = []
    if not ROOT.exists():
        return out
    for p in ROOT.iterdir():
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        if (p / "firstboard_topk_v1").exists():
            out.append(p.name)
    return sorted(out, reverse=True)


def _strategy_root(test_id: str) -> Path:
    """Build a strategy root path under one test_id."""
    # Resolve the strategy root for firstboard_topk_v1.
    return ROOT / test_id / "firstboard_topk_v1"


def _list_runs(strategy_root: Path) -> list[Path]:
    """List available run directories."""
    # Enumerate run folders under <strategy_root>/runs.
    runs_dir = strategy_root / "runs"
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def _list_batches(strategy_root: Path) -> list[Path]:
    """List available experiment batch directories."""
    # Enumerate batch folders under <strategy_root>/experiments.
    batches_dir = strategy_root / "experiments"
    if not batches_dir.exists():
        return []
    return sorted([p for p in batches_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def _read_batch_summary(batch_dir: Path) -> pd.DataFrame:
    """Read summary.yaml for one experiment batch."""
    # Load summary.yaml and normalize into a DataFrame.
    obj = _load_yaml(batch_dir / "summary.yaml")
    rows = obj["summary"]
    return pd.DataFrame(rows)


def _read_nav(run_dir: Path) -> pd.DataFrame:
    """Read nav.feather for a run."""
    # Load nav series and sort by trade_date.
    nav_path = run_dir / "backtest" / "nav.feather"
    df = pd.read_feather(nav_path)
    return df.sort_values("trade_date").reset_index(drop=True)


def _read_report(run_dir: Path) -> str:
    """Read evaluation/report.md for a run."""
    # Load report markdown as text.
    report_path = run_dir / "evaluation" / "report.md"
    return report_path.read_text(encoding="utf-8")


def _read_status(run_dir: Path) -> dict[str, object]:
    """Read evaluation/status.yaml for a run."""
    # Load status yaml as a dict for display.
    status_path = run_dir / "evaluation" / "status.yaml"
    obj = _load_yaml(status_path)
    return dict(obj)


def _to_trade_date_dt(trade_date: pd.Series) -> pd.Series:
    """Convert trade_date int YYYYMMDD into pandas datetime."""
    # Parse YYYYMMDD into datetime64 for plotting.
    return pd.to_datetime(trade_date.astype(int).astype(str), format="%Y%m%d")


def main() -> None:
    """Render the Streamlit UI for browsing backtest outputs."""
    # Build the sidebar for navigation.
    st.set_page_config(page_title="shouban2 firstboard_topk_v1", layout="wide")
    st.sidebar.title("firstboard_topk_v1")

    # Choose the test_id root under /data-cache/shouban2.
    test_ids = _list_test_ids()
    selected_test = st.sidebar.selectbox("Test ID", options=test_ids, index=0) if len(test_ids) > 0 else None
    if selected_test is None:
        st.write("No test outputs found under `/data-cache/shouban2/<test_id>/firstboard_topk_v1`.")
        return
    strategy_root = _strategy_root(selected_test)

    # Allow choosing between single runs and experiment batches.
    mode = st.sidebar.radio("Mode", options=["Run", "Experiments"], index=0)

    if mode == "Experiments":
        batches = _list_batches(strategy_root)
        batch_names = [p.name for p in batches]
        selected_batch = st.sidebar.selectbox("Batch", options=batch_names, index=0) if len(batch_names) > 0 else None
        if selected_batch is None:
            st.write("No experiment batches found under `/data-cache/shouban2/<test_id>/firstboard_topk_v1/experiments`.")
            return

        batch_dir = strategy_root / "experiments" / selected_batch
        st.header(f"Test: {selected_test} | Batch: {selected_batch}")
        if (batch_dir / "report.md").exists():
            st.subheader("Batch Report")
            st.markdown((batch_dir / "report.md").read_text(encoding="utf-8"))

        if (batch_dir / "summary.yaml").exists():
            st.subheader("Summary")
            df = _read_batch_summary(batch_dir).sort_values(["sharpe", "ann_return"], ascending=[False, False])
            st.dataframe(df, use_container_width=True)

        return

    # Load runs and allow the user to pick one.
    runs = _list_runs(strategy_root)
    run_names = [p.name for p in runs]
    selected = st.sidebar.selectbox("Run", options=run_names, index=0) if len(run_names) > 0 else None

    if selected is None:
        st.write("No runs found under `/data-cache/shouban2/<test_id>/firstboard_topk_v1/runs`.")
        return

    run_dir = strategy_root / "runs" / selected
    st.header(f"Test: {selected_test} | Run: {selected}")

    # Show evaluation kind metadata for clarity.
    if (run_dir / "evaluation" / "status.yaml").exists():
        status = _read_status(run_dir)
        st.caption(f"Eval kind: {status.get('eval_kind', 'unknown')}")

    # Show config.yaml for reproducibility.
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        st.subheader("Config")
        st.json(cfg)

    # Show NAV and drawdown chart.
    nav_df = _read_nav(run_dir)
    nav_df["trade_date_dt"] = _to_trade_date_dt(nav_df["trade_date"])
    nav_df["nav_norm"] = nav_df["nav"].astype(float) / float(nav_df["nav"].iloc[0])
    nav_df["peak"] = nav_df["nav_norm"].cummax()
    nav_df["drawdown"] = nav_df["nav_norm"] / nav_df["peak"] - 1.0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("NAV (normalized)")
        chart = (
            alt.Chart(nav_df)
            .mark_line()
            .encode(
                x=alt.X("trade_date_dt:T", axis=alt.Axis(title="date", format="%Y-%m", tickCount=12)),
                y=alt.Y("nav_norm:Q", axis=alt.Axis(title="nav_norm")),
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.subheader("Drawdown")
        chart = (
            alt.Chart(nav_df)
            .mark_line()
            .encode(
                x=alt.X("trade_date_dt:T", axis=alt.Axis(title="date", format="%Y-%m", tickCount=12)),
                y=alt.Y("drawdown:Q", axis=alt.Axis(title="drawdown")),
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # Show report markdown for postmortem.
    st.subheader("Report")
    st.markdown(_read_report(run_dir))


if __name__ == "__main__":
    main()
