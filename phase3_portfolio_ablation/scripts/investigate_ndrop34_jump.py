#!/usr/bin/env python3
"""Investigate ndrop3 vs ndrop4 7.6pp jump: extreme days/stocks or systematic?"""

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
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.utils import fill_placeholder, init_instance_by_config
from qlib.backtest import backtest as normal_backtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "experiment2_ndrop" / "ndrop34_jump_investigation"
PRED_PATH = ROOT / "inputs" / "baseline_pred.pkl"
CFG_BASE = ROOT / "configs" / "baseline.yaml"
ALL_NDROPS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
NDROP_CONFIG = {
    n: ROOT / "configs" / ("baseline.yaml" if n == 2 else f"ndrop{n}.yaml") for n in ALL_NDROPS
}
EXECUTOR = {
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


def run_backtest(ndrop: int, pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg_path = NDROP_CONFIG[ndrop]
    with cfg_path.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    port_cfg = fill_placeholder(copy.deepcopy(conf["port_analysis_config"]), {"<PRED>": pred})
    pmd, _ = normal_backtest(
        executor=copy.deepcopy(EXECUTOR),
        strategy=port_cfg["strategy"],
        **port_cfg["backtest"],
    )
    report, positions = pmd["1day"]
    excess = report["return"] - report["bench"] - report["cost"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(report.index),
        "portfolio_return": report["return"].values,
        "bench": report["bench"].values,
        "cost": report["cost"].values,
        "excess_return": excess.values,
        "turnover": report["turnover"].values,
    })
    return df, positions


def holdings_snapshot(positions: dict, dt: pd.Timestamp) -> set[str]:
    pos = positions.get(dt)
    if pos is None:
        return set()
    return set(pos.get_stock_list())


def load_label_maps(conf: dict) -> tuple[pd.Series, pd.Series]:
    dataset = init_instance_by_config(conf["task"]["dataset"])
    label_s = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    bench = D.features(
        [BENCHMARK],
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time="2020-01-01",
        end_time="2023-12-31",
    ).droplevel(0).iloc[:, 0]
    bench.index = pd.to_datetime(bench.index)
    return label_s, bench


def stock_return_lookup(label_s: pd.Series, bench_s: pd.Series) -> pd.DataFrame:
    df = label_s.rename("label").reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    bench_df = bench_s.rename("bench").reset_index()
    bench_df["datetime"] = pd.to_datetime(bench_df["datetime"])
    df = df.merge(bench_df, on="datetime", how="left")
    df["excess_label"] = df["label"] - df["bench"]
    return df.set_index(["datetime", "instrument"])


def ann_excess(series: pd.Series) -> float:
    ra = risk_analysis(series.dropna(), freq="day")
    return float(ra.loc["annualized_return", "risk"])


def direction_seq(arr_by_ndrop: dict[int, float]) -> str:
    ndrops = sorted(arr_by_ndrop)
    dirs = []
    for i in range(1, len(ndrops)):
        dirs.append("↑" if arr_by_ndrop[ndrops[i]] > arr_by_ndrop[ndrops[i - 1]] else "↓")
    return "".join(dirs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha256_file(PRED_PATH) == "27f0658bc15ed87392ba87b963b2f639c1c1688d3dcb840f72373bd39b78941d"

    with CFG_BASE.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(provider_uri=conf["qlib_init"]["provider_uri"], region=REG_US, kernels=1)
    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)

    label_s, bench_s = load_label_maps(conf)
    ret_lut = stock_return_lookup(label_s, bench_s)

    print("Running ndrop=3 and ndrop=4 backtests...")
    r3, pos3 = run_backtest(3, pred)
    r4, pos4 = run_backtest(4, pred)

    merged = r3.merge(r4, on="datetime", suffixes=("_nd3", "_nd4"))
    merged["excess_diff_nd4_minus_nd3"] = merged["excess_return_nd4"] - merged["excess_return_nd3"]
    merged["abs_diff"] = merged["excess_diff_nd4_minus_nd3"].abs()
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged.to_csv(OUT / "daily_excess_diff_nd4_minus_nd3.csv", index=False)

    total_mean_diff = merged["excess_diff_nd4_minus_nd3"].mean()
    total_ann_contribution_pp = total_mean_diff * 252 * 100
    print(f"Mean daily excess diff (nd4-nd3): {total_mean_diff:.6f}")
    print(f"Annualized contribution: {total_ann_contribution_pp:.2f} pp")

    # --- Task 1: top days ---
    top_n = 20
    top_days = merged.nlargest(top_n, "abs_diff").copy()
    top_days["cum_contribution_pct_of_total"] = (
        top_days["excess_diff_nd4_minus_nd3"].cumsum() / merged["excess_diff_nd4_minus_nd3"].sum() * 100
    )
    top_days.to_csv(OUT / "task1_top_diff_days.csv", index=False)

    # cumulative contribution analysis
    sorted_by_abs = merged.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    sorted_by_abs["cum_diff"] = sorted_by_abs["excess_diff_nd4_minus_nd3"].cumsum()
    n_for_50 = int((sorted_by_abs["excess_diff_nd4_minus_nd3"].abs().cumsum() /
                    sorted_by_abs["excess_diff_nd4_minus_nd3"].abs().sum() >= 0.5).idxmax()) + 1
    n_for_80 = int((sorted_by_abs["excess_diff_nd4_minus_nd3"].abs().cumsum() /
                    sorted_by_abs["excess_diff_nd4_minus_nd3"].abs().sum() >= 0.8).idxmax()) + 1

    # --- Task 2 & 3: holdings diff and stock returns on top days ---
    day_details = []
    stock_rows = []
    for _, row in top_days.head(15).iterrows():
        dt = row["datetime"]
        h3 = holdings_snapshot(pos3, dt)
        h4 = holdings_snapshot(pos4, dt)
        only3 = sorted(h3 - h4)
        only4 = sorted(h4 - h3)
        overlap = len(h3 & h4)
        day_details.append({
            "datetime": dt,
            "excess_nd3": row["excess_return_nd3"],
            "excess_nd4": row["excess_return_nd4"],
            "diff_nd4_minus_nd3": row["excess_diff_nd4_minus_nd3"],
            "n_held_nd3": len(h3),
            "n_held_nd4": len(h4),
            "overlap": overlap,
            "only_nd3": only3,
            "only_nd4": only4,
            "n_only_nd3": len(only3),
            "n_only_nd4": len(only4),
        })
        for side, stocks in [("only_nd3", only3), ("only_nd4", only4)]:
            for s in stocks:
                key = (dt, s)
                label_v = excess_v = np.nan
                if key in ret_lut.index:
                    label_v = float(ret_lut.loc[key, "label"])
                    excess_v = float(ret_lut.loc[key, "excess_label"])
                # also check t+1, t-1 for context
                fwd_excess = np.nan
                dt_next = merged.loc[merged.datetime > dt, "datetime"].min() if (merged.datetime > dt).any() else None
                if dt_next is not None:
                    kn = (dt_next, s)
                    if kn in ret_lut.index:
                        fwd_excess = float(ret_lut.loc[kn, "excess_label"])
                stock_rows.append({
                    "datetime": dt,
                    "side": side,
                    "instrument": s,
                    "label_return": label_v,
                    "excess_label": excess_v,
                    "next_day_excess_label": fwd_excess,
                    "abs_excess_label": abs(excess_v) if pd.notna(excess_v) else np.nan,
                })

    pd.DataFrame(day_details).to_csv(OUT / "task2_holdings_diff_top_days.csv", index=False)
    stock_df = pd.DataFrame(stock_rows)
    stock_df.to_csv(OUT / "task3_key_diff_stocks_returns.csv", index=False)

    # flag extreme stocks (|excess_label| > 10% single day)
    extreme_threshold = 0.10
    extreme_stocks = stock_df[stock_df["abs_excess_label"] > extreme_threshold].copy()
    extreme_stocks.to_csv(OUT / "task3_extreme_diff_stocks.csv", index=False)

    # which top days are dominated by extreme diff stocks
    day_stock_contrib = []
    for _, drow in pd.DataFrame(day_details).iterrows():
        dt = drow["datetime"]
        sub = stock_df[stock_df["datetime"] == dt]
        if len(sub):
            day_stock_contrib.append({
                "datetime": dt,
                "day_diff": drow["diff_nd4_minus_nd3"],
                "max_abs_stock_excess": float(sub["abs_excess_label"].max()),
                "mean_abs_stock_excess_only_diff": float(sub["abs_excess_label"].mean()),
                "n_extreme_stocks_gt10pct": int((sub["abs_excess_label"] > extreme_threshold).sum()),
                "only_nd3": drow["only_nd3"],
                "only_nd4": drow["only_nd4"],
            })
    pd.DataFrame(day_stock_contrib).to_csv(OUT / "task3_day_extreme_flags.csv", index=False)

    # --- Task 4: trim extreme days and recompute 11-point ARR ---
    # Define exclusion sets
    exclude_sets = {
        "none": set(),
        "top5_abs_diff_days": set(sorted_by_abs.head(5)["datetime"]),
        "top10_abs_diff_days": set(sorted_by_abs.head(10)["datetime"]),
        "top20_abs_diff_days": set(sorted_by_abs.head(20)["datetime"]),
        "days_with_extreme_diff_stock_gt10pct": set(
            pd.DataFrame(day_stock_contrib).loc[
                pd.DataFrame(day_stock_contrib)["n_extreme_stocks_gt10pct"] > 0, "datetime"
            ] if day_stock_contrib else []
        ),
    }

    print("Running all 11 ndrop backtests for trim analysis...")
    all_reports: dict[int, pd.DataFrame] = {3: r3, 4: r4}
    for n in ALL_NDROPS:
        if n in all_reports:
            continue
        print(f"  ndrop={n}")
        all_reports[n], _ = run_backtest(n, pred)

    trim_results = {}
    for ex_name, exclude_dates in exclude_sets.items():
        arr_by_ndrop = {}
        n_days_by_ndrop = {}
        for n, rep in all_reports.items():
            sub = rep[~rep["datetime"].isin(exclude_dates)]
            arr_by_ndrop[n] = ann_excess(sub["excess_return"]) * 100
            n_days_by_ndrop[n] = len(sub)
        trim_results[ex_name] = {
            "n_excluded_days": len(exclude_dates),
            "excluded_dates": [str(d.date()) for d in sorted(exclude_dates)],
            "arr_pct_by_ndrop": {int(k): round(v, 4) for k, v in arr_by_ndrop.items()},
            "direction_sequence": direction_seq(arr_by_ndrop),
            "n_direction_flips": sum(
                1 for i in range(1, len(ALL_NDROPS) - 1)
                if (arr_by_ndrop[ALL_NDROPS[i]] - arr_by_ndrop[ALL_NDROPS[i - 1]]) *
                (arr_by_ndrop[ALL_NDROPS[i + 1]] - arr_by_ndrop[ALL_NDROPS[i]]) < 0
            ),
            "nd3_arr": arr_by_ndrop[3],
            "nd4_arr": arr_by_ndrop[4],
            "nd4_minus_nd3_pp": arr_by_ndrop[4] - arr_by_ndrop[3],
        }

    # compare flip counts
    full_arr = {n: ann_excess(all_reports[n]["excess_return"]) * 100 for n in ALL_NDROPS}
    trim_results["baseline_full_sample"] = {
        "arr_pct_by_ndrop": {int(k): round(v, 4) for k, v in full_arr.items()},
        "direction_sequence": direction_seq(full_arr),
        "nd4_minus_nd3_pp": full_arr[4] - full_arr[3],
    }

    (OUT / "task4_trimmed_arr_comparison.json").write_text(
        json.dumps(trim_results, indent=2, default=str), encoding="utf-8"
    )

    # plot before/after for top10 trim
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [
        (axes[0], "baseline_full_sample", "Full sample (no trim)"),
        (axes[1], "top10_abs_diff_days", "Exclude top 10 |diff| days"),
    ]:
        data = trim_results[key]["arr_pct_by_ndrop"]
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, "o-", linewidth=1.5)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xlabel("n_drop")
        ax.set_ylabel("ARR excess w/ cost (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key == "baseline_full_sample":
            ax.annotate(f"nd4-nd3={trim_results[key]['nd4_minus_nd3_pp']:.2f}pp", xy=(0.05, 0.95),
                        xycoords="axes fraction", va="top", fontsize=9)
        else:
            ax.annotate(f"nd4-nd3={trim_results[key]['nd4_minus_nd3_pp']:.2f}pp", xy=(0.05, 0.95),
                        xycoords="axes fraction", va="top", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "task4_arr_curve_full_vs_trimmed.png", dpi=150)
    plt.close()

    # summary
    summary = {
        "pred_sha256": sha256_file(PRED_PATH),
        "full_sample": {
            "nd3_arr_pct": round(full_arr[3], 4),
            "nd4_arr_pct": round(full_arr[4], 4),
            "nd4_minus_nd3_pp": round(full_arr[4] - full_arr[3], 4),
        },
        "task1_top5_diff_days": top_days.head(5)[
            ["datetime", "excess_return_nd3", "excess_return_nd4", "excess_diff_nd4_minus_nd3"]
        ].astype({"datetime": str}).to_dict(orient="records"),
        "task1_concentration": {
            "n_days_for_50pct_abs_diff_mass": n_for_50,
            "n_days_for_80pct_abs_diff_mass": n_for_80,
            "total_backtest_days": len(merged),
        },
        "task3_extreme_stock_count": len(extreme_stocks),
        "task4_trim_comparison": {
            k: {
                "nd4_minus_nd3_pp": v.get("nd4_minus_nd3_pp"),
                "direction_sequence": v.get("direction_sequence"),
                "n_excluded_days": v.get("n_excluded_days", 0),
            }
            for k, v in trim_results.items()
        },
        "verdict": _verdict(trim_results, n_for_50, n_for_80, len(extreme_stocks), len(merged)),
    }
    (OUT / "investigation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== Top 10 diff days ===")
    print(top_days.head(10)[["datetime", "excess_return_nd3", "excess_return_nd4", "excess_diff_nd4_minus_nd3"]].to_string(index=False))
    print(f"\n50% abs diff mass from top {n_for_50} days; 80% from top {n_for_80} days")
    print(f"Extreme diff stocks (|excess|>10%): {len(extreme_stocks)}")
    print("\n=== Trim comparison (nd4-nd3 pp) ===")
    for k, v in trim_results.items():
        if "nd4_minus_nd3_pp" in v:
            print(f"  {k}: {v['nd4_minus_nd3_pp']:.2f} pp, seq={v.get('direction_sequence','')}")
    print(f"\nVerdict: {summary['verdict']}")
    print(f"Outputs: {OUT}")


def _verdict(trim_results, n50, n80, n_extreme, n_days) -> str:
    full_gap = trim_results["baseline_full_sample"]["nd4_minus_nd3_pp"]
    top10_gap = trim_results.get("top10_abs_diff_days", {}).get("nd4_minus_nd3_pp", full_gap)
    top20_gap = trim_results.get("top20_abs_diff_days", {}).get("nd4_minus_nd3_pp", full_gap)
    full_seq = trim_results["baseline_full_sample"]["direction_sequence"]
    top10_seq = trim_results.get("top10_abs_diff_days", {}).get("direction_sequence", full_seq)

    concentrated = n50 <= 10 or n80 <= 30
    gap_shrinks_a_lot = abs(top10_gap) < abs(full_gap) * 0.5
    sawtooth_weakens = top10_seq.count("↑") + top10_seq.count("↓")  # same length; check nd3-4 segment
    nd34_full = "↑" if full_gap > 0 else "↓"
    nd34_top10 = "↑" if top10_gap > 0 else "↓"

    if concentrated and gap_shrinks_a_lot:
        return (
            f"LIKELY_DOMINATED_BY_FEW_DAYS: top10 trim shrinks nd4-nd3 gap from {full_gap:.2f}pp to {top10_gap:.2f}pp; "
            f"50% abs-diff mass from {n50}/{n_days} days; extreme diff stocks={n_extreme}. "
            f"Sawtooth may be partially artifact."
        )
    if abs(top10_gap) > abs(full_gap) * 0.7:
        return (
            f"LIKELY_SYSTEMATIC: nd4-nd3 gap remains {top10_gap:.2f}pp after excluding top10 diff days (was {full_gap:.2f}pp); "
            f"not dominated by a handful of days alone."
        )
    return (
        f"MIXED: top10 trim gap={top10_gap:.2f}pp vs full={full_gap:.2f}pp; top20={top20_gap:.2f}pp; "
        f"concentration {n50}/{n_days} days for 50% mass."
    )


if __name__ == "__main__":
    main()
