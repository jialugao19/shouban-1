"""Streamlit app to browse runs/experiments under /data-cache for firstboard_topk_v1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


ROOT = Path("/data-cache/shouban2/firstboard_topk_v1")


def _load_yaml(path: Path) -> object:
    """Load a YAML file as a Python object."""
    # Read YAML text and parse with safe_load.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _list_runs() -> list[Path]:
    """List available run directories."""
    # Enumerate run folders under ROOT/runs.
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def _list_batches() -> list[Path]:
    """List available experiment batch directories."""
    # Enumerate batch folders under ROOT/experiments.
    batches_dir = ROOT / "experiments"
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


def main() -> None:
    """Render the Streamlit UI for browsing backtest outputs."""
    # Build the sidebar for navigation.
    st.set_page_config(page_title="shouban2 firstboard_topk_v1", layout="wide")
    st.sidebar.title("firstboard_topk_v1")

    # Allow choosing between single runs and experiment batches.
    mode = st.sidebar.radio("Mode", options=["Run", "Experiments"], index=0)

    if mode == "Experiments":
        batches = _list_batches()
        batch_names = [p.name for p in batches]
        selected_batch = st.sidebar.selectbox("Batch", options=batch_names, index=0) if len(batch_names) > 0 else None
        if selected_batch is None:
            st.write("No experiment batches found under `/data-cache/shouban2/firstboard_topk_v1/experiments`.")
            return

        batch_dir = ROOT / "experiments" / selected_batch
        st.header(f"Batch: {selected_batch}")
        if (batch_dir / "report.md").exists():
            st.subheader("Batch Report")
            st.markdown((batch_dir / "report.md").read_text(encoding="utf-8"))

        if (batch_dir / "summary.yaml").exists():
            st.subheader("Summary")
            df = _read_batch_summary(batch_dir).sort_values(["sharpe", "ann_return"], ascending=[False, False])
            st.dataframe(df, use_container_width=True)

        return

    # Load runs and allow the user to pick one.
    runs = _list_runs()
    run_names = [p.name for p in runs]
    selected = st.sidebar.selectbox("Run", options=run_names, index=0) if len(run_names) > 0 else None

    if selected is None:
        st.write("No runs found under `/data-cache/shouban2/firstboard_topk_v1/runs`.")
        return

    run_dir = ROOT / "runs" / selected
    st.header(f"Run: {selected}")

    # Show config.yaml for reproducibility.
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        st.subheader("Config")
        st.json(cfg)

    # Show NAV and drawdown chart.
    nav_df = _read_nav(run_dir)
    nav_df["nav_norm"] = nav_df["nav"].astype(float) / float(nav_df["nav"].iloc[0])
    nav_df["peak"] = nav_df["nav_norm"].cummax()
    nav_df["drawdown"] = nav_df["nav_norm"] / nav_df["peak"] - 1.0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("NAV (normalized)")
        st.line_chart(nav_df.set_index("trade_date")["nav_norm"])
    with c2:
        st.subheader("Drawdown")
        st.line_chart(nav_df.set_index("trade_date")["drawdown"])

    # Show report markdown for postmortem.
    st.subheader("Report")
    st.markdown(_read_report(run_dir))


if __name__ == "__main__":
    main()
