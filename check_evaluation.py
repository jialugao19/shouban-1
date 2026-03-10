"""Check evaluation artifacts for one run under /data-cache."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: Path) -> object:
    """Load a YAML file as a Python object."""
    # Read YAML text and parse with safe_load.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_run(out_root: str) -> None:
    """Check that evaluation YAML artifacts exist and satisfy basic invariants."""
    # Resolve the evaluation directory from the out_root.
    eval_dir = Path(out_root) / "evaluation"
    if not eval_dir.exists():
        raise FileNotFoundError(f"missing evaluation dir: {eval_dir}")

    # Load required YAML artifacts.
    metrics = _load_yaml(eval_dir / "metrics.yaml")
    picks = _load_yaml(eval_dir / "picks.yaml")
    status = _load_yaml(eval_dir / "status.yaml")

    # Validate status metadata.
    if status.get("status") != "ok":
        raise ValueError(f"status not ok: {status.get('status')}")

    # Validate that required metric blocks exist and are numeric.
    for k in ["baseline_t1_open", "score_threshold_t1_open", "baseline_t0_open_close"]:
        block = metrics[k]
        for mk in ["ann_return", "ann_vol", "sharpe", "mdd", "calmar"]:
            float(block[mk])

    # Validate pick summary invariants for baseline and score_threshold.
    for k in ["baseline", "score_threshold"]:
        block = picks[k]
        max_day_picks = int(block["max_day_picks"])
        p99_day_picks = int(block["p99_day_picks"])
        int(block["total_picks"])
        float(block["coverage"])
        float(block["avg_picks"])

        if max_day_picks < 0 or p99_day_picks < 0:
            raise ValueError(f"invalid picks counts for {k}: max={max_day_picks}, p99={p99_day_picks}")


def run() -> None:
    """Check the latest 2024 run under /data-cache by default."""
    # Locate the latest run folder that contains structured evaluation artifacts.
    base = Path("/data-cache/shouban2")
    run_dirs: list[Path] = []
    for test_dir in base.iterdir():
        if not test_dir.is_dir():
            continue
        if not test_dir.name.isdigit():
            continue
        runs_dir = test_dir / "firstboard_topk_v1" / "runs"
        if not runs_dir.exists():
            continue
        for p in runs_dir.iterdir():
            if p.is_dir():
                run_dirs.append(p)
    run_dirs = sorted(run_dirs, key=lambda p: p.name, reverse=True)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"no runs found under: {base}")

    # Pick the newest run that already has evaluation/metrics.yaml.
    out_root = ""
    for p in run_dirs:
        if (p / "evaluation" / "metrics.yaml").exists():
            out_root = str(p)
            break
    if out_root == "":
        raise FileNotFoundError(f"no runs with evaluation/metrics.yaml found under: {base}")

    # Check the selected run.
    check_run(out_root)
    print(f"[ok] {out_root}")


if __name__ == "__main__":
    run()
