#!/usr/bin/env python3
"""Decompose ~8pp gap between static rank1-20 excess label ARR and portfolio ARR."""

from __future__ import annotations

import copy
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gap_decomposition"
PRED_PATH = ROOT / "inputs" / "baseline_pred.pkl"
CFG_PATH = ROOT / "configs" / "baseline.yaml"
BENCHMARK = "P10104"
EXECUTOR = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}


def ann(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean() * 252) if len(s) else float("nan")


def risk_metrics(excess_daily: pd.Series) -> dict:
    ra = risk_analysis(excess_daily.dropna(), freq="day")
    return {
        "annualized_return": float(ra.loc["annualized_return", "risk"]),
        "information_ratio": float(ra.loc["information_ratio", "risk"]),
        "max_drawdown": float(ra.loc["max_drawdown", "risk"]),
        "n_days": int(excess_daily.dropna().shape[0]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CFG_PATH.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    qlib.init(provider_uri=conf["qlib_init"]["provider_uri"], region=REG_US, kernels=1)

    with PRED_PATH.open("rb") as f:
        pred = pickle.load(f)
    pred_s = pred["score"].copy()

    dataset = init_instance_by_config(conf["task"]["dataset"])
    label_s = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]

    bench_raw = D.features(
        [BENCHMARK],
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time="2020-01-01",
        end_time="2023-12-31",
    )
    bench_label = bench_raw.droplevel("instrument").iloc[:, 0]
    bench_label.index = pd.to_datetime(bench_label.index)

    # close-to-close return for deal_price=close alignment check
    close_raw = D.features(
        D.instruments("sp500"),
        ["$close", "Ref($close, 1)"],
        start_time="2019-12-01",
        end_time="2023-12-31",
    )
    close = close_raw["$close"].unstack("instrument")
    close_prev = close_raw["Ref($close, 1)"].unstack("instrument")
    ret_cc = (close - close_prev) / close_prev  # return from T-1 close to T close

    port_cfg = fill_placeholder(copy.deepcopy(conf["port_analysis_config"]), {"<PRED>": pred})
    pmd, _ = normal_backtest(
        executor=copy.deepcopy(EXECUTOR),
        strategy=port_cfg["strategy"],
        **port_cfg["backtest"],
    )
    report, positions = pmd["1day"]

    # lookup tables
    label_df = label_s.rename("label").reset_index()
    label_map = label_df.set_index(["datetime", "instrument"])["label"]

    bench_map = bench_label.rename("bench")

    pred_df = pred_s.rename("score").reset_index()

    dates = sorted(pd.to_datetime(report.index))
    label_df["datetime"] = pd.to_datetime(label_df["datetime"])
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
    daily_rows = []

    for i, dt in enumerate(dates):
        pos = positions.get(dt)
        if pos is None:
            continue
        held = pos.get_stock_list()
        if not held:
            continue

        # scores
        day_pred = pred_df[pred_df["datetime"] == dt].set_index("instrument")["score"]
        prev_dt = dates[i - 1] if i > 0 else None
        prev_pred = (
            pred_df[pred_df["datetime"] == prev_dt].set_index("instrument")["score"]
            if prev_dt is not None
            else day_pred
        )

        top20_today = day_pred.sort_values(ascending=False).head(20).index.tolist()
        top20_lag = prev_pred.sort_values(ascending=False).head(20).index.tolist()

        overlap_today = len(set(held) & set(top20_today)) / 20.0
        overlap_lag = len(set(held) & set(top20_lag)) / 20.0
        outside_top20_today = [s for s in held if s not in top20_today]
        outside_top20_lag = [s for s in held if s not in top20_lag]

        # ranks of held stocks (by today's score)
        ranks = day_pred.sort_values(ascending=False).rank(ascending=False, method="first")
        held_ranks = ranks.reindex(held).dropna()
        mean_held_rank = float(held_ranks.mean())
        max_held_rank = float(held_ranks.max())

        bench_v = float(bench_map.get(dt, np.nan))

        def segment_stats(stocks: list[str], use_cc: bool = False) -> dict:
            if not stocks:
                return {"raw": np.nan, "excess_label": np.nan, "excess_cc": np.nan}
            raw_vals, ex_label, ex_cc = [], [], []
            for s in stocks:
                key = (dt, s)
                if key in label_map.index:
                    lv = float(label_map.loc[key])
                    raw_vals.append(lv)
                    ex_label.append(lv - bench_v)
                if use_cc and s in ret_cc.columns and dt in ret_cc.index:
                    rv = ret_cc.loc[dt, s]
                    if pd.notna(rv):
                        ex_cc.append(float(rv) - bench_v)
            return {
                "raw": float(np.mean(raw_vals)) if raw_vals else np.nan,
                "excess_label": float(np.mean(ex_label)) if ex_label else np.nan,
                "excess_cc": float(np.mean(ex_cc)) if ex_cc else np.nan,
            }

        st_top20_today = segment_stats(top20_today)
        st_top20_lag = segment_stats(top20_lag)
        st_held_ew = segment_stats(held)
        st_held_ew_cc = segment_stats(held, use_cc=True)
        st_outside_lag = segment_stats(outside_top20_lag)
        st_inside_lag = segment_stats([s for s in held if s in top20_lag])

        # position weights
        wdict = pos.get_stock_weight_dict(only_stock=True)
        stock_w = {k: v for k, v in wdict.items() if k != "cash"}
        cash_w = 1.0 - sum(stock_w.values()) if stock_w else 1.0

        def weighted_excess_label(stocks_weights: dict[str, float]) -> float:
            num, den = 0.0, 0.0
            for s, w in stocks_weights.items():
                key = (dt, s)
                if key in label_map.index and pd.notna(w) and w > 0:
                    num += w * (float(label_map.loc[key]) - bench_v)
                    den += w
            return num / den if den > 0 else np.nan

        wt_held_label = weighted_excess_label(stock_w)

        # weight dispersion vs equal
        if stock_w:
            ew = 1.0 / len(stock_w)
            weight_l1_dev = float(sum(abs(w - ew) for w in stock_w.values()))
            max_weight = float(max(stock_w.values()))
            min_weight = float(min(stock_w.values()))
        else:
            weight_l1_dev = max_weight = min_weight = np.nan

        port_excess = float(report.loc[dt, "return"] - report.loc[dt, "bench"] - report.loc[dt, "cost"])
        port_excess_nocost = float(report.loc[dt, "return"] - report.loc[dt, "bench"])

        daily_rows.append({
            "datetime": dt,
            "n_held": len(held),
            "overlap_with_top20_same_day": overlap_today,
            "overlap_with_top20_lag1_signal": overlap_lag,
            "n_outside_top20_lag1": len(outside_top20_lag),
            "mean_held_rank": mean_held_rank,
            "max_held_rank": max_held_rank,
            "cash_weight": cash_w,
            "weight_l1_dev_from_ew": weight_l1_dev,
            "max_stock_weight": max_weight,
            "min_stock_weight": min_weight,
            "bench": bench_v,
            # static ideals
            "static_top20_today_excess_label": st_top20_today["excess_label"],
            "static_top20_lag1_excess_label": st_top20_lag["excess_label"],
            "static_top20_today_excess_cc": st_top20_today["excess_cc"],
            "static_top20_lag1_excess_cc": st_top20_lag["excess_cc"],
            # actual holdings
            "held_ew_excess_label": st_held_ew["excess_label"],
            "held_ew_excess_cc": st_held_ew_cc["excess_cc"],
            "held_weighted_excess_label": wt_held_label,
            "held_inside_lag_top20_excess_label": st_inside_lag["excess_label"],
            "held_outside_lag_top20_excess_label": st_outside_lag["excess_label"],
            # portfolio
            "portfolio_excess_with_cost": port_excess,
            "portfolio_excess_without_cost": port_excess_nocost,
            "portfolio_cost": float(report.loc[dt, "cost"]),
        })

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / "daily_decomposition.csv", index=False)

    # full rebalance ideal: daily EW top-20 by today's score, close-to-close excess
    rebal_rows = []
    for dt in sorted(pred_df["datetime"].unique()):
        day_pred = pred_df[pred_df["datetime"] == dt].set_index("instrument")["score"]
        top20 = day_pred.sort_values(ascending=False).head(20).index.tolist()
        bench_v = float(bench_map.get(dt, np.nan))
        ex_cc = []
        for s in top20:
            if s in ret_cc.columns and dt in ret_cc.index and pd.notna(ret_cc.loc[dt, s]):
                ex_cc.append(float(ret_cc.loc[dt, s]) - bench_v)
        rebal_rows.append({
            "datetime": dt,
            "ideal_daily_ew_rebal_top20_excess_cc": float(np.mean(ex_cc)) if ex_cc else np.nan,
        })
    rebal = pd.DataFrame(rebal_rows)
    daily = daily.merge(rebal, on="datetime", how="left")

    metrics = {}
    series_map = {
        "L0_static_top20_today_excess_label": daily["static_top20_today_excess_label"],
        "L1_static_top20_lag1_excess_label": daily["static_top20_lag1_excess_label"],
        "L2_held_ew_excess_label": daily["held_ew_excess_label"],
        "L3_held_weighted_excess_label": daily["held_weighted_excess_label"],
        "L4_static_top20_today_excess_cc": daily["static_top20_today_excess_cc"],
        "L5_static_top20_lag1_excess_cc": daily["static_top20_lag1_excess_cc"],
        "L6_held_ew_excess_cc": daily["held_ew_excess_cc"],
        "L7_ideal_daily_ew_rebal_top20_excess_cc": daily["ideal_daily_ew_rebal_top20_excess_cc"],
        "L8_portfolio_excess_without_cost": daily["portfolio_excess_without_cost"],
        "L9_portfolio_excess_with_cost": daily["portfolio_excess_with_cost"],
    }
    for name, s in series_map.items():
        metrics[name] = {"mean_daily": float(s.mean()), **risk_metrics(s)}

    # incremental gaps (annualized)
    def gap(a: str, b: str) -> float:
        return metrics[b]["annualized_return"] - metrics[a]["annualized_return"]

    gaps = {
        "H2_signal_lag_label": {
            "from": "L0_static_top20_today_excess_label",
            "to": "L1_static_top20_lag1_excess_label",
            "annualized_gap_pp": gap("L0_static_top20_today_excess_label", "L1_static_top20_lag1_excess_label"),
            "hypothesis": "signal shift=1 vs same-day score (timing/label alignment partial)",
        },
        "H1_dynamic_holdings_label": {
            "from": "L1_static_top20_lag1_excess_label",
            "to": "L2_held_ew_excess_label",
            "annualized_gap_pp": gap("L1_static_top20_lag1_excess_label", "L2_held_ew_excess_label"),
            "hypothesis": "n_drop dynamic holdings vs static lagged top-20 (label space)",
        },
        "H3_weighting_label": {
            "from": "L2_held_ew_excess_label",
            "to": "L3_held_weighted_excess_label",
            "annualized_gap_pp": gap("L2_held_ew_excess_label", "L3_held_weighted_excess_label"),
            "hypothesis": "equal-weight vs actual position weights (label space)",
        },
        "H2_label_vs_close_to_close_static": {
            "from": "L0_static_top20_today_excess_label",
            "to": "L4_static_top20_today_excess_cc",
            "annualized_gap_pp": gap("L0_static_top20_today_excess_label", "L4_static_top20_today_excess_cc"),
            "hypothesis": "label formula vs close-to-close deal_price return (static top20)",
        },
        "H2_label_vs_close_to_close_held": {
            "from": "L2_held_ew_excess_label",
            "to": "L6_held_ew_excess_cc",
            "annualized_gap_pp": gap("L2_held_ew_excess_label", "L6_held_ew_excess_cc"),
            "hypothesis": "label formula vs close-to-close on actual holdings",
        },
        "H1+H3_ideal_rebal_vs_portfolio": {
            "from": "L7_ideal_daily_ew_rebal_top20_excess_cc",
            "to": "L8_portfolio_excess_without_cost",
            "annualized_gap_pp": gap("L7_ideal_daily_ew_rebal_top20_excess_cc", "L8_portfolio_excess_without_cost"),
            "hypothesis": "daily EW rebal top20 (cc) vs actual strategy w/o cost",
        },
        "cost_only": {
            "from": "L8_portfolio_excess_without_cost",
            "to": "L9_portfolio_excess_with_cost",
            "annualized_gap_pp": gap("L8_portfolio_excess_without_cost", "L9_portfolio_excess_with_cost"),
            "hypothesis": "transaction cost",
        },
        "TOTAL_label_ideal_to_portfolio": {
            "from": "L0_static_top20_today_excess_label",
            "to": "L9_portfolio_excess_with_cost",
            "annualized_gap_pp": gap("L0_static_top20_today_excess_label", "L9_portfolio_excess_with_cost"),
            "hypothesis": "full gap to explain",
        },
    }

    # overlap / holdings diagnostics
    overlap_stats = {
        "mean_overlap_same_day_top20": float(daily["overlap_with_top20_same_day"].mean()),
        "mean_overlap_lag1_top20": float(daily["overlap_with_top20_lag1_signal"].mean()),
        "mean_n_outside_lag1_top20": float(daily["n_outside_top20_lag1"].mean()),
        "mean_held_rank": float(daily["mean_held_rank"].mean()),
        "median_max_held_rank": float(daily["max_held_rank"].median()),
        "pct_days_max_held_rank_gt_20": float((daily["max_held_rank"] > 20).mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "median_cash_weight": float(daily["cash_weight"].median()),
        "mean_weight_l1_dev_from_ew": float(daily["weight_l1_dev_from_ew"].mean()),
    }

    # inside vs outside lag-top20 contribution
    inside_ann = ann(daily["held_inside_lag_top20_excess_label"])
    outside_ann = ann(daily["held_outside_lag_top20_excess_label"])
    outside_share = float(daily["n_outside_top20_lag1"].mean() / 20.0)

    # verify label definition empirically: corr(label, close-to-close)
    sample_keys = label_df.set_index(["datetime", "instrument"])
    cc_vals, lb_vals = [], []
    for dt in dates[::5]:
        for s in sample_keys.loc[dt].index[:30] if dt in sample_keys.index.get_level_values(0) else []:
            try:
                lv = float(label_map.loc[(dt, s)])
                if s in ret_cc.columns and dt in ret_cc.index:
                    rv = float(ret_cc.loc[dt, s])
                    if pd.notna(rv) and pd.notna(lv):
                        lb_vals.append(lv)
                        cc_vals.append(rv)
            except KeyError:
                pass
    label_cc_corr = float(np.corrcoef(lb_vals, cc_vals)[0, 1]) if len(cc_vals) > 10 else float("nan")

    # label vs forward return: Ref($close,-1)/$close - 1
    sample_insts = list(pred_df["instrument"].unique()[:50])
    fwd_raw = D.features(
        sample_insts,
        ["Ref($close, -1)/$close - 1"],
        start_time="2020-01-01",
        end_time="2023-12-31",
    )
    fwd = fwd_raw.iloc[:, 0].unstack("instrument")
    fwd_vals, lb2_vals = [], []
    for dt in dates[::10]:
        for s in fwd.columns[:20]:
            key = (dt, s)
            if key in label_map.index and dt in fwd.index and pd.notna(fwd.loc[dt, s]):
                lb2_vals.append(float(label_map.loc[key]))
                fwd_vals.append(float(fwd.loc[dt, s]))
    label_fwd_corr = float(np.corrcoef(lb2_vals, fwd_vals)[0, 1]) if len(fwd_vals) > 10 else float("nan")

    summary = {
        "reference": {
            "task3_static_rank1_20_excess_label_arr": -0.030492896645752388,
            "portfolio_topk20_excess_arr_with_cost": -0.1113093366471705,
            "total_gap_pp": -0.08081643999941812,
            "portfolio_cost_drag_arr": float(
                metrics["L8_portfolio_excess_without_cost"]["annualized_return"]
                - metrics["L9_portfolio_excess_with_cost"]["annualized_return"]
            ),
        },
        "level_metrics": metrics,
        "incremental_gaps_pp": gaps,
        "overlap_stats": overlap_stats,
        "inside_vs_outside_lag_top20": {
            "inside_lag_top20_ew_excess_label_ann": inside_ann,
            "outside_lag_top20_ew_excess_label_ann": outside_ann,
            "mean_outside_fraction_of_portfolio": outside_share,
        },
        "label_timing_diagnostics": {
            "corr_label_with_close_to_close_same_day": label_cc_corr,
            "corr_label_with_forward_return_ref_close_m1_over_close": label_fwd_corr,
            "note": "label=Ref($close,-2)/Ref($close,-1)-1; deal_price=close uses close-to-close",
        },
    }
    (OUT / "gap_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # compact table
    rows = []
    for k, v in metrics.items():
        rows.append({"level": k, **v})
    pd.DataFrame(rows).to_csv(OUT / "level_metrics.csv", index=False)

    print("=== Level metrics (annualized excess return) ===")
    for k, v in metrics.items():
        print(f"{k:45s} {v['annualized_return']:+.4f} ({v['n_days']} days)")
    print("\n=== Incremental gaps (pp) ===")
    for k, v in gaps.items():
        print(f"{k:40s} {v['annualized_gap_pp']:+.4f}  # {v['hypothesis']}")
    print("\n=== Overlap stats ===")
    print(json.dumps(overlap_stats, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
