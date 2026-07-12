#!/usr/bin/env python3
"""Experiment 3: 11-point hold_thresh ablation (full sample + 3 periods + path-dep check)."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.utils import fill_placeholder
from qlib.backtest import backtest as normal_backtest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "experiment3_holdthresh"
PRED_PATH = ROOT / "inputs" / "baseline_pred.pkl"
HASH_FILE = ROOT / "inputs" / "input_hashes.txt"
CFG_BASE = ROOT / "configs" / "baseline.yaml"
PRED_SHA256 = "27f0658bc15ed87392ba87b963b2f639c1c1688d3dcb840f72373bd39b78941d"

ALL_HOLD_THRESH = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
HT_CONFIG = {
    ht: ROOT / "configs" / ("baseline.yaml" if ht == 1 else f"holdthresh{ht}.yaml")
    for ht in ALL_HOLD_THRESH
}
PERIODS = {
    "full_sample_2020_2023": ("2020-01-01", "2023-12-31", 1006),
    "2020Q1_COVID": ("2020-01-01", "2020-03-31", 62),
    "2020Q2_2021_recovery": ("2020-04-01", "2021-12-31", 443),
    "2022_2023_post_covid": ("2022-01-01", "2023-12-31", 501),
}
EXECUTOR = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}
PATH_DEP_THRESHOLD_PP = 3.0


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


def verify_all_hashes() -> dict:
    manifest = load_manifest()
    rels = ["inputs/baseline_pred.pkl"] + [
        f"configs/{'baseline' if ht == 1 else f'holdthresh{ht}'}.yaml" for ht in ALL_HOLD_THRESH
    ]
    checks = []
    for rel in rels:
        path = ROOT / rel
        actual = sha256_file(path)
        expected = manifest.get(rel)
        checks.append({
            "relative": rel,
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        })
    result = {
        "all_match": all(c["match"] for c in checks),
        "pred_sha256": sha256_file(PRED_PATH),
        "n_configs": len(rels),
        "checks": checks,
    }
    (OUT / "manifest_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_match"]:
        bad = [c for c in checks if not c["match"]]
        raise ValueError(f"Manifest verification failed: {bad}")
    if result["pred_sha256"] != PRED_SHA256:
        raise ValueError(f"pred SHA256 mismatch: {result['pred_sha256']}")
    return result


def run_backtest(
    hold_thresh: int,
    pred: pd.DataFrame,
    start: str,
    end: str,
    *,
    return_positions: bool = False,
) -> dict | tuple[dict, dict, pd.DataFrame]:
    with HT_CONFIG[hold_thresh].open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    port_cfg = copy.deepcopy(conf["port_analysis_config"])
    port_cfg = fill_placeholder(port_cfg, {"<PRED>": pred})
    port_cfg["backtest"]["start_time"] = start
    port_cfg["backtest"]["end_time"] = end
    pmd, _ = normal_backtest(
        executor=copy.deepcopy(EXECUTOR),
        strategy=port_cfg["strategy"],
        **port_cfg["backtest"],
    )
    report, positions = pmd["1day"]
    ex_w = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
    ex_wo = risk_analysis(report["return"] - report["bench"], freq="day")
    counts = [len(p.get_stock_list()) for p in positions.values() if hasattr(p, "get_stock_list")]
    metrics = {
        "hold_thresh": hold_thresh,
        "annualized_return_with_cost": float(ex_w.loc["annualized_return", "risk"]),
        "annualized_return_without_cost": float(ex_wo.loc["annualized_return", "risk"]),
        "annualized_cost_drag": float(ex_wo.loc["annualized_return", "risk"] - ex_w.loc["annualized_return", "risk"]),
        "information_ratio_with_cost": float(ex_w.loc["information_ratio", "risk"]),
        "maximum_drawdown_with_cost": float(ex_w.loc["max_drawdown", "risk"]),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "average_number_of_positions": float(sum(counts) / len(counts)) if counts else None,
        "n_backtest_days": int(len(report)),
    }
    if return_positions:
        excess = report["return"] - report["bench"] - report["cost"]
        daily = pd.DataFrame({
            "datetime": pd.to_datetime(report.index),
            "excess_return": excess.values,
            "turnover": report["turnover"].values,
        })
        return metrics, positions, daily
    return metrics


def direction_seq(arr: dict[int, float]) -> str:
    keys = sorted(arr.keys())
    return "".join("↑" if arr[keys[i]] > arr[keys[i - 1]] else "↓" for i in range(1, len(keys)))


def count_flips(seq: str) -> int:
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def analyze_curve(df: pd.DataFrame, period: str) -> dict:
    arr = {int(r.hold_thresh): float(r.annualized_return_with_cost) * 100 for r in df.itertuples()}
    best = df.loc[df["annualized_return_with_cost"].idxmax()]
    worst = df.loc[df["annualized_return_with_cost"].idxmin()]
    seq = direction_seq(arr)
    return {
        "period": period,
        "n_days": int(df["n_backtest_days"].iloc[0]),
        "arr_pct_by_hold_thresh": {int(k): round(v, 4) for k, v in arr.items()},
        "direction_sequence": seq,
        "n_direction_flips": count_flips(seq),
        "best_hold_thresh": int(best["hold_thresh"]),
        "best_arr_pct": round(float(best["annualized_return_with_cost"]) * 100, 4),
        "worst_hold_thresh": int(worst["hold_thresh"]),
        "worst_arr_pct": round(float(worst["annualized_return_with_cost"]) * 100, 4),
        "mean_turnover_by_ht": {
            int(r.hold_thresh): round(float(r.mean_daily_turnover) * 100, 4) for r in df.itertuples()
        },
        "cost_drag_by_ht": {
            int(r.hold_thresh): round(float(r.annualized_cost_drag) * 100, 4) for r in df.itertuples()
        },
    }


def adjacent_gaps(df: pd.DataFrame) -> list[dict]:
    pdf = df.sort_values("hold_thresh")
    gaps = []
    for i in range(1, len(pdf)):
        lo = pdf.iloc[i - 1]
        hi = pdf.iloc[i]
        gap_pp = (float(hi["annualized_return_with_cost"]) - float(lo["annualized_return_with_cost"])) * 100
        gaps.append({
            "ht_lo": int(lo["hold_thresh"]),
            "ht_hi": int(hi["hold_thresh"]),
            "arr_lo_pct": round(float(lo["annualized_return_with_cost"]) * 100, 4),
            "arr_hi_pct": round(float(hi["annualized_return_with_cost"]) * 100, 4),
            "gap_pp": round(gap_pp, 4),
            "abs_gap_pp": round(abs(gap_pp), 4),
            "exceeds_3pp": abs(gap_pp) > PATH_DEP_THRESHOLD_PP,
        })
    return gaps


def holdings_snapshot(positions: dict, dt: pd.Timestamp) -> set[str]:
    pos = positions.get(dt)
    if pos is None:
        return set()
    return set(pos.get_stock_list())


def investigate_path_dependency(
    ht_lo: int,
    ht_hi: int,
    pred: pd.DataFrame,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, pos_lo, daily_lo = run_backtest(ht_lo, pred, "2020-01-01", "2023-12-31", return_positions=True)
    _, pos_hi, daily_hi = run_backtest(ht_hi, pred, "2020-01-01", "2023-12-31", return_positions=True)

    merged = daily_lo.merge(daily_hi, on="datetime", suffixes=("_lo", "_hi"))
    merged["excess_diff_hi_minus_lo"] = merged["excess_return_hi"] - merged["excess_return_lo"]
    merged["abs_diff"] = merged["excess_diff_hi_minus_lo"].abs()
    merged.to_csv(out_dir / f"daily_excess_diff_ht{ht_hi}_minus_ht{ht_lo}.csv", index=False)

    top_days = merged.nlargest(20, "abs_diff").copy()
    top_days["date_str"] = top_days["datetime"].dt.strftime("%Y-%m-%d")
    holdings_rows = []
    overlap_vals = []
    for _, row in top_days.iterrows():
        dt = row["datetime"]
        h_lo = holdings_snapshot(pos_lo, dt)
        h_hi = holdings_snapshot(pos_hi, dt)
        overlap = len(h_lo & h_hi)
        n_lo, n_hi = len(h_lo), len(h_hi)
        overlap_vals.append(overlap)
        holdings_rows.append({
            "datetime": row["date_str"],
            "abs_diff_pp": round(row["abs_diff"] * 100, 4),
            "excess_diff_hi_minus_lo_pp": round(row["excess_diff_hi_minus_lo"] * 100, 4),
            "overlap_count": overlap,
            "n_holdings_lo": n_lo,
            "n_holdings_hi": n_hi,
            "overlap_pct_of_max": round(overlap / max(n_lo, n_hi, 1) * 100, 1),
            "only_in_lo": sorted(h_lo - h_hi)[:5],
            "only_in_hi": sorted(h_hi - h_lo)[:5],
        })
    holdings_df = pd.DataFrame(holdings_rows)
    holdings_df.to_csv(out_dir / "top_diff_days_holdings.csv", index=False)

    arr_lo = float(daily_lo["excess_return"].mean()) * 252 * 100
    arr_hi = float(daily_hi["excess_return"].mean()) * 252 * 100
    summary = {
        "pair": f"ht{ht_lo}_vs_ht{ht_hi}",
        "ht_lo": ht_lo,
        "ht_hi": ht_hi,
        "arr_lo_pct": round(arr_lo, 4),
        "arr_hi_pct": round(arr_hi, 4),
        "gap_pp": round(arr_hi - arr_lo, 4),
        "top20_mean_overlap": round(float(np.mean(overlap_vals)), 2),
        "top20_min_overlap": int(min(overlap_vals)),
        "top20_max_overlap": int(max(overlap_vals)),
        "median_overlap_all_days": int(np.median([
            len(holdings_snapshot(pos_lo, dt) & holdings_snapshot(pos_hi, dt))
            for dt in merged["datetime"]
        ])),
        "path_dependency_likely": float(np.mean(overlap_vals)) < 15,
        "interpretation": (
            "Low EOD overlap on high-diff days suggests path-dependency divergence"
            if float(np.mean(overlap_vals)) < 15
            else "Holdings remain relatively aligned; gap may be turnover/cost not path fork"
        ),
    }
    (out_dir / "investigation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_arr_curve(period_dfs: dict[str, pd.DataFrame], analyses: dict, path: Path) -> None:
    panels = [
        ("full_sample_2020_2023", "Full sample 2020-2023 (1006d)"),
        ("2020Q1_COVID", "2020Q1 COVID (62d)"),
        ("2020Q2_2021_recovery", "2020Q2-2021 Recovery (443d)"),
        ("2022_2023_post_covid", "2022-2023 Post-COVID (501d)"),
    ]
    panels = [(k, t) for k, t in panels if k in period_dfs]
    n = len(panels)
    if n == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
    else:
        fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
        axes = list(axes_grid.flatten())[:n]
    for ax, (key, title) in zip(axes, panels):
        pdf = period_dfs[key].sort_values("hold_thresh")
        x = pdf["hold_thresh"].values
        y = pdf["annualized_return_with_cost"].values * 100
        ax.plot(x, y, "o-", linewidth=1.5, color="#059669", markersize=7)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7)
        a = analyses[key]
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xlabel("hold_thresh")
        ax.set_ylabel("ARR excess w/ cost (%)")
        ax.grid(True, alpha=0.3)
        subtitle = (
            f"seq={a['direction_sequence']}  flips={a['n_direction_flips']}  "
            f"best={a['best_hold_thresh']}({a['best_arr_pct']:.1f}%)"
        )
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")
    plt.suptitle("Experiment 3: ARR vs hold_thresh (11 points, topk=20, n_drop=2)", fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


def resume_from_csv() -> None:
    """Re-run analysis/plots/path-dep from saved task_all_runs.csv."""
    csv_path = OUT / "task_all_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    period_dfs = {k: v for k, v in df.groupby("period")}
    analyses = {k: analyze_curve(v, k) for k, v in period_dfs.items()}
    (OUT / "curve_analysis.json").write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    full_df = period_dfs["full_sample_2020_2023"]
    gaps = adjacent_gaps(full_df)
    (OUT / "adjacent_gaps_full_sample.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    plot_arr_curve(period_dfs, analyses, OUT / "arr_vs_holdthresh_four_panel.png")
    plot_arr_curve(
        {"full_sample_2020_2023": full_df},
        {"full_sample_2020_2023": analyses["full_sample_2020_2023"]},
        OUT / "arr_vs_holdthresh_full_sample.png",
    )
    plot_turnover_cost(full_df, OUT / "turnover_cost_vs_holdthresh.png")
    with CFG_BASE.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(provider_uri=conf["qlib_init"]["provider_uri"], region=REG_US, kernels=1)
    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)
    path_dep_dir = OUT / "path_dependency_checks"
    path_investigations = []
    for g in gaps:
        if g["exceeds_3pp"]:
            print(f"Path-dep: ht{g['ht_lo']} vs ht{g['ht_hi']} ({g['gap_pp']:.1f}pp)")
            inv = investigate_path_dependency(
                g["ht_lo"], g["ht_hi"], pred,
                path_dep_dir / f"ht{g['ht_lo']}_vs_ht{g['ht_hi']}",
            )
            path_investigations.append(inv)
    turnover_mono = monotonic_check(analyses["full_sample_2020_2023"]["mean_turnover_by_ht"], decreasing=True)
    cost_mono = monotonic_check(analyses["full_sample_2020_2023"]["cost_drag_by_ht"], decreasing=True)
    extremes = {
        "hold_thresh_1": {k: analyses[k]["arr_pct_by_hold_thresh"].get(1) for k in analyses},
        "hold_thresh_30": {k: analyses[k]["arr_pct_by_hold_thresh"].get(30) for k in analyses},
    }
    key_findings = {
        "pred_sha256": PRED_SHA256,
        "manifest_verified": True,
        "n_runs": len(df),
        "full_sample_curve": analyses["full_sample_2020_2023"],
        "turnover_monotonic_decreasing": turnover_mono,
        "cost_drag_monotonic_decreasing": cost_mono,
        "adjacent_gaps_full_sample": gaps,
        "path_dependency_investigations": path_investigations,
        "regime_optima": {
            k: {"best_ht": v["best_hold_thresh"], "best_arr_pct": v["best_arr_pct"]}
            for k, v in analyses.items()
        },
        "extremes": extremes,
        "task4_answers": {
            "q1_turnover_cost_monotone": {
                "turnover_decreasing": turnover_mono["is_monotonic_decreasing"],
                "cost_drag_decreasing": cost_mono["is_monotonic_decreasing"],
            },
            "q2_arr_stability": {
                "full_sample_flips": analyses["full_sample_2020_2023"]["n_direction_flips"],
                "still_sawtooth": analyses["full_sample_2020_2023"]["n_direction_flips"] >= 4,
                "regime_dependent": len(set(v["best_hold_thresh"] for v in analyses.values())) > 1,
            },
            "q3_extremes": extremes,
        },
    }
    (OUT / "key_findings.json").write_text(json.dumps(key_findings, indent=2, default=str), encoding="utf-8")
    print(json.dumps(key_findings, indent=2, default=str))


def plot_turnover_cost(full_df: pd.DataFrame, path: Path) -> None:
    pdf = full_df.sort_values("hold_thresh")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = pdf["hold_thresh"].values
    ax1.plot(x, pdf["mean_daily_turnover"].values * 100, "o-", color="#2563eb")
    ax1.set_xlabel("hold_thresh")
    ax1.set_ylabel("Mean daily turnover (%)")
    ax1.set_title("Turnover vs hold_thresh")
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, pdf["annualized_cost_drag"].values * 100, "o-", color="#dc2626")
    ax2.set_xlabel("hold_thresh")
    ax2.set_ylabel("Annualized cost drag (pp)")
    ax2.set_title("Cost drag vs hold_thresh")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


def monotonic_check(series: dict[int, float], decreasing: bool = True) -> dict:
    keys = sorted(series.keys())
    violations = []
    for i in range(1, len(keys)):
        if decreasing and series[keys[i]] > series[keys[i - 1]]:
            violations.append((keys[i - 1], keys[i]))
        elif not decreasing and series[keys[i]] < series[keys[i - 1]]:
            violations.append((keys[i - 1], keys[i]))
    return {
        "is_monotonic_decreasing": len(violations) == 0 if decreasing else None,
        "n_violations": len(violations),
        "violations": violations,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "task_all_runs.csv").exists() and not (OUT / "key_findings.json").exists():
        print("Resuming from saved CSV...")
        resume_from_csv()
        return

    print("=== Batch verify-hash (44 configs + pred) ===")
    manifest_result = verify_all_hashes()
    print(f"All {manifest_result['n_configs']} entries verified OK")

    with CFG_BASE.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(provider_uri=conf["qlib_init"]["provider_uri"], region=REG_US, kernels=1)
    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)

    rows = []
    period_dfs: dict[str, pd.DataFrame] = {}

    for period, (start, end, _) in PERIODS.items():
        print(f"\n=== {period} ({start} ~ {end}) ===")
        period_rows = []
        for ht in ALL_HOLD_THRESH:
            print(f"  hold_thresh={ht}")
            m = run_backtest(ht, pred, start, end)
            row = {
                "period": period,
                "start": start,
                "end": end,
                "pred_hash": PRED_SHA256,
                **m,
            }
            rows.append(row)
            period_rows.append(row)
        period_dfs[period] = pd.DataFrame(period_rows)
        sub = period_dfs[period].sort_values("hold_thresh")
        print(sub[["hold_thresh", "annualized_return_with_cost", "mean_daily_turnover",
                    "annualized_cost_drag"]].to_string(index=False))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task_all_runs.csv", index=False)
    (OUT / "task_all_runs.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    resume_from_csv()
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
