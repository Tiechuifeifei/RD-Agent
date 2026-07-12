#!/usr/bin/env python3
"""Period-sliced IC + TopK portfolio analysis (fixed pred.pkl, no retrain)."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.constant import REG_US
from qlib.contrib.data.handler import DataHandlerLP
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.utils import fill_placeholder, init_instance_by_config
from qlib.backtest import backtest as normal_backtest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "period_analysis"
PRED_PATH = ROOT / "inputs" / "baseline_pred.pkl"
HASH_FILE = ROOT / "inputs" / "input_hashes.txt"
BASELINE_CONFIG = ROOT / "configs" / "baseline.yaml"

PERIODS = {
    "2020Q1_COVID": ("2020-01-01", "2020-03-31"),
    "2020Q2_2021_recovery": ("2020-04-01", "2021-12-31"),
    "2022_2023_post_covid": ("2022-01-01", "2023-12-31"),
}

TOPK_CONFIGS = {
    10: ROOT / "configs" / "topk10.yaml",
    20: ROOT / "configs" / "baseline.yaml",
    50: ROOT / "configs" / "topk50.yaml",
    100: ROOT / "configs" / "topk100.yaml",
}

SEGMENTS = {
    "rank_1_20": (1, 20),
    "rank_21_50": (21, 50),
    "rank_51_100": (51, 100),
}

EXECUTOR_CONFIG = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}

BENCHMARK = "P10104"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                manifest[parts[1]] = parts[0]
    return manifest


def verify_manifest() -> dict:
    manifest = load_manifest()
    checks = []
    for rel in [
        "inputs/baseline_pred.pkl",
        "configs/baseline.yaml",
        "configs/topk10.yaml",
        "configs/topk50.yaml",
        "configs/topk100.yaml",
    ]:
        path = ROOT / rel
        actual = sha256_file(path)
        expected = manifest.get(rel)
        checks.append({
            "path": str(path),
            "relative": rel,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        })
    return {
        "manifest_path": str(HASH_FILE),
        "all_match": all(c["match"] for c in checks),
        "checks": checks,
    }


def init_qlib_from_config() -> dict:
    with BASELINE_CONFIG.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(
        provider_uri=conf["qlib_init"]["provider_uri"],
        region=REG_US,
        kernels=1,
    )
    return conf


def load_pred_and_label(conf: dict) -> tuple[pd.DataFrame, pd.Series]:
    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)
    dataset = init_instance_by_config(conf["task"]["dataset"])
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    label_s = label.iloc[:, 0]
    return pred, label_s


def slice_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = df.index.get_level_values("datetime")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask]


def task1_ic_by_period(pred: pd.DataFrame, label_s: pd.Series) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        pred_p = slice_by_period(pred, start, end)
        label_p = slice_by_period(label_s.to_frame("label"), start, end)
        common_idx = pred_p.index.intersection(label_p.index)
        pred_aligned = pred_p.loc[common_idx, "score"]
        label_aligned = label_p.loc[common_idx, "label"]
        ic, ric = calc_ic(pred_aligned, label_aligned)
        rows.append({
            "period": period,
            "start": start,
            "end": end,
            "mean_ic": float(ic.mean()),
            "mean_rank_ic": float(ric.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
            "rank_icir": float(ric.mean() / ric.std()) if ric.std() else float("nan"),
            "n_trading_days": int(len(ic)),
            "n_observations": int(len(pred_aligned)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task1_ic_by_period.csv", index=False)
    (OUT / "task1_ic_by_period.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def run_portfolio_period(
    port_cfg: dict,
    pred: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    cfg = copy.deepcopy(port_cfg)
    cfg = fill_placeholder(cfg, {"<PRED>": pred})
    cfg["backtest"]["start_time"] = start
    cfg["backtest"]["end_time"] = end
    pmd, _ = normal_backtest(
        executor=copy.deepcopy(EXECUTOR_CONFIG),
        strategy=cfg["strategy"],
        **cfg["backtest"],
    )
    report, _ = pmd["1day"]
    excess_w = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
    return {
        "annualized_return_with_cost": float(excess_w.loc["annualized_return", "risk"]),
        "information_ratio_with_cost": float(excess_w.loc["information_ratio", "risk"]),
        "maximum_drawdown_with_cost": float(excess_w.loc["max_drawdown", "risk"]),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "n_backtest_days": int(len(report)),
    }


def task2_topk_by_period(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topk, cfg_path in TOPK_CONFIGS.items():
        with cfg_path.open(encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        port_cfg = conf["port_analysis_config"]
        for period, (start, end) in PERIODS.items():
            metrics = run_portfolio_period(port_cfg, pred, start, end)
            rows.append({
                "period": period,
                "start": start,
                "end": end,
                "topk": topk,
                "config": str(cfg_path),
                "pred_hash": sha256_file(PRED_PATH),
                **metrics,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task2_topk_by_period.csv", index=False)

    # U-shape check per period: is 50 worst among 10/20/50/100?
    u_checks = []
    for period in PERIODS:
        sub = df[df["period"] == period].set_index("topk")["annualized_return_with_cost"]
        u_shape = (
            sub.loc[50] < sub.loc[20]
            and sub.loc[50] < sub.loc[100]
            and sub.loc[20] > sub.loc[50]
            and sub.loc[100] > sub.loc[50]
        ) if all(k in sub.index for k in [10, 20, 50, 100]) else False
        best_topk = int(sub.idxmax())
        worst_topk = int(sub.idxmin())
        u_checks.append({
            "period": period,
            "arr_by_topk": {int(k): float(v) for k, v in sub.items()},
            "u_shape_50_worst_between_20_and_100": bool(u_shape),
            "best_topk": best_topk,
            "worst_topk": worst_topk,
        })
    # exclude 2020Q1 combined periods
    no_q1 = df[df["period"] != "2020Q1_COVID"]
    combined_no_q1 = (
        no_q1.groupby("topk")["annualized_return_with_cost"].mean().to_dict()
        if len(no_q1) else {}
    )
    u_no_q1 = False
    if all(k in combined_no_q1 for k in [20, 50, 100]):
        u_no_q1 = combined_no_q1[50] < combined_no_q1[20] and combined_no_q1[50] < combined_no_q1[100]

    summary = {
        "pred_path": str(PRED_PATH),
        "pred_hash": sha256_file(PRED_PATH),
        "period_u_shape_checks": u_checks,
        "full_test_u_shape_from_prior_run": {
            "topk20": -0.11130933664717045,
            "topk50": -0.14739281261156706,
            "topk100": -0.11893363156968696,
            "u_shape_50_worst": True,
        },
        "excluding_2020Q1_mean_arr_by_topk": {int(k): float(v) for k, v in combined_no_q1.items()},
        "excluding_2020Q1_u_shape_50_worst": u_no_q1,
    }
    (OUT / "task2_u_shape_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df


def task3_excess_segment_full_test(pred: pd.DataFrame, conf: dict) -> pd.DataFrame:
    label = init_instance_by_config(conf["task"]["dataset"])
    label_s = label.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    bench_raw = D.features(
        [BENCHMARK],
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time="2020-01-01",
        end_time="2023-12-31",
    )
    bench_s = bench_raw.droplevel("instrument").iloc[:, 0]
    bench_s.index = pd.to_datetime(bench_s.index)

    merged = pred["score"].to_frame().join(label_s.rename("label"), how="inner").reset_index()
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.merge(
        bench_s.rename("bench").reset_index().rename(columns={"datetime": "datetime"}),
        on="datetime",
        how="inner",
    )
    merged["excess_label"] = merged["label"] - merged["bench"]

    rows = []
    daily_rows = []
    for dt, grp in merged.groupby("datetime"):
        g = grp.sort_values("score", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        drow = {"datetime": dt, "bench": float(g["bench"].iloc[0])}
        for seg, (lo, hi) in SEGMENTS.items():
            sub = g[(g["rank"] >= lo) & (g["rank"] <= hi)]
            ex = float(sub["excess_label"].mean()) if len(sub) else np.nan
            raw = float(sub["label"].mean()) if len(sub) else np.nan
            drow[f"{seg}_excess"] = ex
            drow[f"{seg}_raw"] = raw
        daily_rows.append(drow)

    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(OUT / "task3_daily_segment_raw_vs_excess.csv", index=False)

    for seg in SEGMENTS:
        col_ex = f"{seg}_excess"
        col_raw = f"{seg}_raw"
        ex_s = daily_df[col_ex].dropna()
        raw_s = daily_df[col_raw].dropna()
        rows.append({
            "segment": seg,
            "raw_annualized_return": float(raw_s.mean() * 252),
            "excess_annualized_return": float(ex_s.mean() * 252),
            "excess_daily_std": float(ex_s.std(ddof=1)),
            "n_days": int(len(ex_s)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task3_excess_segment_summary.csv", index=False)
    (OUT / "task3_excess_segment_summary.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conf = init_qlib_from_config()
    pred, label_s = load_pred_and_label(conf)

    print("=== Task 4: manifest verification ===")
    t4 = verify_manifest()
    (OUT / "task4_manifest_verification.json").write_text(json.dumps(t4, indent=2), encoding="utf-8")
    print(json.dumps(t4, indent=2))
    if not t4["all_match"]:
        raise SystemExit("Manifest verification failed")

    print("\n=== Task 1: IC by period ===")
    t1 = task1_ic_by_period(pred, label_s)
    print(t1.to_string(index=False))

    print("\n=== Task 2: TopK by period ===")
    t2 = task2_topk_by_period(pred)
    print(t2.to_string(index=False))

    print("\n=== Task 3: excess segment (full test) ===")
    t3 = task3_excess_segment_full_test(pred, conf)
    print(t3.to_string(index=False))

    meta = {
        "pred_path": str(PRED_PATH),
        "pred_hash": sha256_file(PRED_PATH),
        "pred_md5": hashlib.md5(PRED_PATH.read_bytes()).hexdigest(),
        "label_source": "dataset.prepare(test, label, DK_R) — same as SignalRecord",
        "benchmark": BENCHMARK,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
