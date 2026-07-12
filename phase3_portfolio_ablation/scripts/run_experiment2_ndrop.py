#!/usr/bin/env python3
"""Phase 3 Experiment 2: n_drop ablation (fixed topk=20, fixed pred.pkl)."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import qlib
import yaml
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.utils import fill_placeholder
from qlib.backtest import backtest as normal_backtest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "experiment2_ndrop"
PRED_PATH = ROOT / "inputs" / "baseline_pred.pkl"
HASH_FILE = ROOT / "inputs" / "input_hashes.txt"
RUNNER = ROOT / "scripts" / "run_portfolio_ablation.py"
PYTHON = "/opt/anaconda3/envs/rdagent4qlib/bin/python"

NDROP_CONFIGS = {
    1: ROOT / "configs" / "ndrop1.yaml",
    2: ROOT / "configs" / "baseline.yaml",
    5: ROOT / "configs" / "ndrop5.yaml",
    10: ROOT / "configs" / "ndrop10.yaml",
    20: ROOT / "configs" / "ndrop20.yaml",
}

PERIODS = {
    "2020Q1_COVID": ("2020-01-01", "2020-03-31"),
    "2020Q2_2021_recovery": ("2020-04-01", "2021-12-31"),
    "2022_2023_post_covid": ("2022-01-01", "2023-12-31"),
}

EXECUTOR = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> dict:
    manifest = {}
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                manifest[parts[1]] = parts[0]

    rels = ["inputs/baseline_pred.pkl"] + [f"configs/ndrop{n}.yaml" if n != 2 else "configs/baseline.yaml" for n in NDROP_CONFIGS]
    checks = []
    for rel in rels:
        path = ROOT / rel
        actual = sha256_file(path)
        expected = manifest.get(rel)
        checks.append({"relative": rel, "match": actual == expected, "sha256": actual})
    return {"all_match": all(c["match"] for c in checks), "checks": checks}


def run_full_sample(ndrop: int, skip_if_exists: bool = False) -> dict:
    cfg = NDROP_CONFIGS[ndrop]
    out_dir = OUT / f"ndrop{ndrop}"
    summary_path = out_dir / "summary.json"

    if ndrop == 2 and not summary_path.exists():
        baseline_summary = ROOT / "results" / "baseline" / "summary.json"
        if baseline_summary.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_summary, summary_path)
            for extra in ["effective_config.yaml"]:
                src = ROOT / "results" / "baseline" / extra
                if src.exists():
                    shutil.copy2(src, out_dir / extra)

    if skip_if_exists and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if ndrop != 2 or not summary_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, str(RUNNER),
            "--config", str(cfg),
            "--pred", str(PRED_PATH),
            "--output", str(out_dir),
            "--verify-hash",
        ]
        print(f"\n=== Running n_drop={ndrop} ===")
        subprocess.run(cmd, check=True)

    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_period(ndrop: int, pred: pd.DataFrame, conf_qlib: dict) -> pd.DataFrame:
    with NDROP_CONFIGS[ndrop].open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    port_cfg = copy.deepcopy(conf["port_analysis_config"])
    port_cfg = fill_placeholder(port_cfg, {"<PRED>": pred})

    rows = []
    for period, (start, end) in PERIODS.items():
        metrics = _backtest_period(port_cfg, pred, start, end)
        rows.append({
            "n_drop": ndrop,
            "topk": 20,
            "period": period,
            "start": start,
            "end": end,
            "pred_hash": sha256_file(PRED_PATH),
            **metrics,
        })
    return pd.DataFrame(rows)


def _backtest_period(port_cfg: dict, pred: pd.DataFrame, start: str, end: str) -> dict:
    cfg = copy.deepcopy(port_cfg)
    cfg = fill_placeholder(cfg, {"<PRED>": pred})
    cfg["backtest"]["start_time"] = start
    cfg["backtest"]["end_time"] = end
    pmd, _ = normal_backtest(
        executor=copy.deepcopy(EXECUTOR),
        strategy=cfg["strategy"],
        **cfg["backtest"],
    )
    report, positions = pmd["1day"]
    ex_w = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
    ex_wo = risk_analysis(report["return"] - report["bench"], freq="day")
    arr_w = float(ex_w.loc["annualized_return", "risk"])
    arr_wo = float(ex_wo.loc["annualized_return", "risk"])
    return {
        "annualized_return_with_cost": arr_w,
        "annualized_return_without_cost": arr_wo,
        "annualized_cost_drag": arr_wo - arr_w,
        "information_ratio_with_cost": float(ex_w.loc["information_ratio", "risk"]),
        "maximum_drawdown_with_cost": float(ex_w.loc["max_drawdown", "risk"]),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "average_number_of_positions": _avg_positions(positions),
        "n_backtest_days": int(len(report)),
    }


def _avg_positions(positions: dict) -> float:
    counts = [len(p.get_stock_list()) for p in positions.values() if hasattr(p, "get_stock_list")]
    return float(sum(counts) / len(counts)) if counts else float("nan")


def summary_row(ndrop: int, s: dict) -> dict:
    m = s["metrics"]
    strat = s.get("strategy", {})
    return {
        "n_drop": ndrop,
        "topk": strat.get("kwargs", {}).get("topk", 20),
        "pred_hash": s["pred_hash"],
        "config_hash": s["config_hash"],
        "annualized_return_with_cost": m["annualized_return_with_cost"],
        "annualized_return_without_cost": m["annualized_return_without_cost"],
        "annualized_cost_drag": m["annualized_cost_drag"],
        "information_ratio_with_cost": m["information_ratio_with_cost"],
        "maximum_drawdown_with_cost": m["maximum_drawdown_with_cost"],
        "mean_daily_turnover": m["mean_daily_turnover"],
        "average_number_of_positions": m.get("average_number_of_positions"),
    }


def analyze_relationships(full_df: pd.DataFrame, period_df: pd.DataFrame) -> dict:
    df = full_df.sort_values("n_drop")
    cost_mono = bool((df["annualized_cost_drag"].diff().dropna() <= 1e-9).all())
    turnover_mono_inc = bool((df["mean_daily_turnover"].diff().dropna() >= -1e-9).all())

    best_arr = df.loc[df["annualized_return_with_cost"].idxmax()]
    worst_arr = df.loc[df["annualized_return_with_cost"].idxmin()]

    period_best = (
        period_df.groupby("period")
        .apply(lambda g: g.loc[g["annualized_return_with_cost"].idxmax(), ["n_drop", "annualized_return_with_cost", "mean_daily_turnover"]].to_dict())
        .to_dict()
    )

    return {
        "cost_drag_monotone_decreasing_with_lower_ndrop": cost_mono,
        "turnover_monotone_increasing_with_higher_ndrop": turnover_mono_inc,
        "full_sample_best_ndrop": int(best_arr["n_drop"]),
        "full_sample_best_arr": float(best_arr["annualized_return_with_cost"]),
        "full_sample_worst_ndrop": int(worst_arr["n_drop"]),
        "full_sample_worst_arr": float(worst_arr["annualized_return_with_cost"]),
        "extreme_ndrop1": df[df.n_drop == 1].iloc[0].to_dict() if 1 in df.n_drop.values else {},
        "extreme_ndrop20": df[df.n_drop == 20].iloc[0].to_dict() if 20 in df.n_drop.values else {},
        "period_best_ndrop": period_best,
        "turnover_arr_correlation_full_sample": float(df["mean_daily_turnover"].corr(df["annualized_return_with_cost"])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Manifest verification ===")
    manifest = verify_manifest()
    (OUT / "manifest_verification.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["all_match"]:
        raise SystemExit("Manifest verification failed")
    print(json.dumps(manifest, indent=2))

    full_summaries = {}
    for ndrop in sorted(NDROP_CONFIGS):
        full_summaries[ndrop] = run_full_sample(ndrop, skip_if_exists=(ndrop == 2))

    full_df = pd.DataFrame([summary_row(k, v) for k, v in full_summaries.items()])
    full_df = full_df.sort_values("n_drop")
    full_df.to_csv(OUT / "task1_full_sample.csv", index=False)
    (OUT / "task1_full_sample.json").write_text(full_df.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n=== Task 1: Full sample ===")
    print(full_df.to_string(index=False))

    with NDROP_CONFIGS[2].open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(provider_uri=conf["qlib_init"]["provider_uri"], region=REG_US, kernels=1)
    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)

    period_parts = [run_period(ndrop, pred, conf["qlib_init"]) for ndrop in sorted(NDROP_CONFIGS)]
    period_df = pd.concat(period_parts, ignore_index=True)
    period_df.to_csv(OUT / "task2_by_period.csv", index=False)

    print("\n=== Task 2: By period ===")
    for period in PERIODS:
        print(f"\n-- {period} --")
        sub = period_df[period_df.period == period].sort_values("n_drop")
        print(sub[["n_drop", "annualized_return_with_cost", "information_ratio_with_cost", "maximum_drawdown_with_cost", "mean_daily_turnover", "average_number_of_positions"]].to_string(index=False))

    analysis = analyze_relationships(full_df, period_df)
    (OUT / "task3_relationship_analysis.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")

    print("\n=== Task 3: Relationship analysis ===")
    print(json.dumps(analysis, indent=2, default=str))
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
