#!/usr/bin/env python3
"""Phase 3 portfolio-only ablation runner (fixed pred.pkl, no model retrain)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# TopkDropoutStrategy defaults not always set in YAML (must be recorded explicitly).
STRATEGY_DEFAULTS: dict[str, Any] = {
    "risk_degree": 0.95,
    "hold_thresh": 1,
    "method_sell": "bottom",
    "method_buy": "top",
    "only_tradable": False,
    "forbid_all_trade_at_limit": True,
}

EXECUTOR_CONFIG: dict[str, Any] = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    },
}

BASELINE_EXPECTED = {
    "annualized_return": -0.11130933664717044,
    "max_drawdown": -0.809844659491796,
}
BASELINE_TOLERANCE = 1e-4

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_ROOT = SCRIPT_DIR.parent
DEFAULT_PRED = PHASE3_ROOT / "inputs" / "baseline_pred.pkl"
DEFAULT_HASH_FILE = PHASE3_ROOT / "inputs" / "input_hashes.txt"
DEFAULT_CONFIG = PHASE3_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT = PHASE3_ROOT / "results" / "baseline"

FORBIDDEN_TOKENS = (
    "LGBModel",
    "model.fit",
    "model.predict",
    "SignalRecord",
    "SigAnaRecord",
    "task_train",
    "qrun",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_manifest(path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    if not path.exists():
        return manifest
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            manifest[parts[1]] = parts[0]
    return manifest


def verify_hash(path: Path, manifest: dict[str, str], label: str) -> None:
    actual = sha256_file(path)
    rel = str(path.relative_to(PHASE3_ROOT)) if path.is_relative_to(PHASE3_ROOT) else path.name
    candidates = [rel, path.name, str(path)]
    expected = None
    for key in candidates:
        if key in manifest:
            expected = manifest[key]
            break
    if expected is None:
        raise ValueError(f"No SHA256 entry found in manifest for {label}: {path}")
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {label} ({path})\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )


def load_yaml_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with config_path.open(encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    if "qlib_init" not in conf:
        raise KeyError(f"Missing qlib_init in {config_path}")
    if "port_analysis_config" not in conf:
        raise KeyError(f"Missing port_analysis_config in {config_path}")
    return conf["qlib_init"], copy.deepcopy(conf["port_analysis_config"])


def init_qlib(qlib_init: dict[str, Any]) -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US

    region = qlib_init.get("region", "cn")
    region_const = REG_US if str(region).lower() == "us" else REG_CN
    kwargs = {
        "provider_uri": qlib_init["provider_uri"],
        "region": region_const,
        "kernels": 1,
    }
    qlib.init(**kwargs)


def load_and_validate_pred(pred_path: Path, backtest_start: str, backtest_end: str) -> pd.DataFrame:
    with pred_path.open("rb") as handle:
        pred = pickle.load(handle)

    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"pred.pkl must be a pandas DataFrame, got {type(pred)}")

    if list(pred.index.names) != ["datetime", "instrument"]:
        raise ValueError(
            f"pred.pkl index names must be ['datetime', 'instrument'], got {pred.index.names}"
        )

    if "score" not in pred.columns:
        raise ValueError(f"pred.pkl must contain a 'score' column, got {list(pred.columns)}")

    pred_dates = pred.index.get_level_values("datetime")
    pred_min = pd.Timestamp(pred_dates.min())
    pred_max = pd.Timestamp(pred_dates.max())
    bt_start = pd.Timestamp(backtest_start)
    bt_end = pd.Timestamp(backtest_end)

    mask = (pred_dates >= bt_start) & (pred_dates <= bt_end)
    if not mask.any():
        raise ValueError(
            "pred.pkl has no scores overlapping the backtest window: "
            f"pred=[{pred_min.date()}, {pred_max.date()}], "
            f"backtest=[{bt_start.date()}, {bt_end.date()}]"
        )

    return pred


def effective_strategy_params(strategy_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = copy.deepcopy(strategy_cfg.get("kwargs", {}))
    signal = kwargs.get("signal")
    if isinstance(signal, pd.DataFrame):
        kwargs["signal"] = "<PRED>"
    elif isinstance(signal, pd.Series):
        kwargs["signal"] = "<PRED>"
    merged = dict(STRATEGY_DEFAULTS)
    merged.update(kwargs)
    return {
        "class": strategy_cfg.get("class"),
        "module_path": strategy_cfg.get("module_path"),
        "kwargs": merged,
    }


def average_position_count(positions: dict[Any, Any]) -> float | None:
    if not positions:
        return None
    counts: list[int] = []
    for pos in positions.values():
        if hasattr(pos, "get_stock_list"):
            counts.append(len(pos.get_stock_list()))
    if not counts:
        return None
    return float(sum(counts) / len(counts))


def run_backtest(port_cfg: dict[str, Any], pred: pd.DataFrame) -> dict[str, Any]:
    from qlib.utils import fill_placeholder
    from qlib.backtest import backtest as normal_backtest
    from qlib.contrib.evaluate import indicator_analysis, risk_analysis

    filled_cfg = fill_placeholder(copy.deepcopy(port_cfg), {"<PRED>": pred})
    strategy_cfg = filled_cfg["strategy"]
    backtest_cfg = filled_cfg["backtest"]

    portfolio_metric_dict, indicator_dict = normal_backtest(
        executor=copy.deepcopy(EXECUTOR_CONFIG),
        strategy=strategy_cfg,
        **backtest_cfg,
    )

    report, positions = portfolio_metric_dict["1day"]
    indicators_df, indicators_obj = indicator_dict["1day"]

    excess_without_cost = risk_analysis(report["return"] - report["bench"], freq="day")
    excess_with_cost = risk_analysis(
        report["return"] - report["bench"] - report["cost"],
        freq="day",
    )
    port_analysis = pd.concat(
        {
            "excess_return_without_cost": excess_without_cost,
            "excess_return_with_cost": excess_with_cost,
        }
    )
    indicator_analysis_df = indicator_analysis(indicators_df)

    mean_daily_cost = float(report["cost"].mean())
    arr_without = float(excess_without_cost.loc["annualized_return", "risk"])
    arr_with = float(excess_with_cost.loc["annualized_return", "risk"])
    annualized_cost_drag = arr_without - arr_with

    return {
        "filled_port_cfg": filled_cfg,
        "report": report,
        "positions": positions,
        "indicators_df": indicators_df,
        "indicators_obj": indicators_obj,
        "port_analysis": port_analysis,
        "indicator_analysis_df": indicator_analysis_df,
        "excess_without_cost": excess_without_cost,
        "excess_with_cost": excess_with_cost,
        "metrics": {
            "annualized_return_without_cost": arr_without,
            "annualized_return_with_cost": arr_with,
            "information_ratio_with_cost": float(
                excess_with_cost.loc["information_ratio", "risk"]
            ),
            "maximum_drawdown_with_cost": float(
                excess_with_cost.loc["max_drawdown", "risk"]
            ),
            "mean_daily_cost": mean_daily_cost,
            "annualized_cost_drag": annualized_cost_drag,
            "average_number_of_positions": average_position_count(positions),
            "mean_daily_turnover": float(report["turnover"].mean()),
        },
    }


def save_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    effective_config: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")

    with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False)

    with (output_dir / "report_normal_1day.pkl").open("wb") as handle:
        pickle.dump(artifacts["report"], handle)

    with (output_dir / "positions_normal_1day.pkl").open("wb") as handle:
        pickle.dump(artifacts["positions"], handle)

    with (output_dir / "indicators_normal_1day.pkl").open("wb") as handle:
        pickle.dump(artifacts["indicators_df"], handle)

    with (output_dir / "indicators_normal_1day_obj.pkl").open("wb") as handle:
        pickle.dump(artifacts["indicators_obj"], handle)

    with (output_dir / "port_analysis_1day.pkl").open("wb") as handle:
        pickle.dump(artifacts["port_analysis"], handle)

    with (output_dir / "indicator_analysis_1day.pkl").open("wb") as handle:
        pickle.dump(artifacts["indicator_analysis_df"], handle)


def validate_baseline(config_path: Path, metrics: dict[str, Any]) -> None:
    if config_path.name != "baseline.yaml":
        return
    arr = metrics["annualized_return_with_cost"]
    mdd = metrics["maximum_drawdown_with_cost"]
    if abs(arr - BASELINE_EXPECTED["annualized_return"]) > BASELINE_TOLERANCE:
        raise AssertionError(
            "baseline.yaml annualized_return_with_cost out of tolerance: "
            f"got {arr}, expected {BASELINE_EXPECTED['annualized_return']} ± {BASELINE_TOLERANCE}"
        )
    if abs(mdd - BASELINE_EXPECTED["max_drawdown"]) > BASELINE_TOLERANCE:
        raise AssertionError(
            "baseline.yaml maximum_drawdown_with_cost out of tolerance: "
            f"got {mdd}, expected {BASELINE_EXPECTED['max_drawdown']} ± {BASELINE_TOLERANCE}"
        )


def static_inspection(script_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8")
    violations: list[str] = []
    in_forbidden_block = False
    for line_no, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("FORBIDDEN_TOKENS"):
            in_forbidden_block = True
            continue
        if in_forbidden_block:
            if stripped == ")":
                in_forbidden_block = False
            continue
        for token in FORBIDDEN_TOKENS:
            if token in line:
                violations.append(f"{token} (line {line_no})")
    if violations:
        raise RuntimeError(
            "Static inspection failed: forbidden tokens present in script: "
            + ", ".join(violations)
        )


def print_compact_summary(summary: dict[str, Any], output_dir: Path) -> None:
    m = summary["metrics"]
    print("Phase 3 portfolio ablation complete")
    print(f"  output:      {output_dir}")
    print(f"  pred hash:   {summary['pred_hash'][:12]}...")
    print(f"  config hash: {summary['config_hash'][:12]}...")
    print(f"  backtest:    {summary['backtest_window']['start']} .. {summary['backtest_window']['end']}")
    print(f"  ARR (w/o):   {m['annualized_return_without_cost']:.6f}")
    print(f"  ARR (with):  {m['annualized_return_with_cost']:.6f}")
    print(f"  IR (with):   {m['information_ratio_with_cost']:.6f}")
    print(f"  MDD (with):  {m['maximum_drawdown_with_cost']:.6f}")
    print(f"  cost drag:   {m['annualized_cost_drag']:.6f} (annualized)")
    if m.get("average_number_of_positions") is not None:
        print(f"  avg pos:     {m['average_number_of_positions']:.2f}")
    if m.get("mean_daily_turnover") is not None:
        print(f"  turnover:    {m['mean_daily_turnover']:.6f} (mean daily)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portfolio-only ablation with a fixed pred.pkl (no model retrain)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config containing qlib_init and port_analysis_config",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PRED,
        help="Fixed prediction pickle (DataFrame with score column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for summary.json and artifacts",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Verify pred/config SHA256 against inputs/input_hashes.txt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = args.config.resolve()
    pred_path = args.pred.resolve()
    output_dir = args.output.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    manifest = load_hash_manifest(DEFAULT_HASH_FILE)
    if args.verify_hash:
        verify_hash(pred_path, manifest, "pred.pkl")
        verify_hash(config_path, manifest, "config")

    pred_hash = sha256_file(pred_path)
    config_hash = sha256_file(config_path)

    qlib_init, port_cfg = load_yaml_config(config_path)
    backtest_start = port_cfg["backtest"]["start_time"]
    backtest_end = port_cfg["backtest"]["end_time"]

    init_qlib(qlib_init)
    pred = load_and_validate_pred(pred_path, backtest_start, backtest_end)

    result = run_backtest(port_cfg, pred)
    metrics = result["metrics"]

    pred_dates = pred.index.get_level_values("datetime")
    strategy_effective = effective_strategy_params(result["filled_port_cfg"]["strategy"])

    effective_config = {
        "qlib_init": qlib_init,
        "executor": EXECUTOR_CONFIG,
        "port_analysis_config": {
            "strategy": strategy_effective,
            "backtest": result["filled_port_cfg"]["backtest"],
        },
    }

    summary = {
        "pred_hash": pred_hash,
        "config_hash": config_hash,
        "pred_path": str(pred_path),
        "config_path": str(config_path),
        "pred_date_range": {
            "start": str(pd.Timestamp(pred_dates.min()).date()),
            "end": str(pd.Timestamp(pred_dates.max()).date()),
        },
        "backtest_window": {
            "start": str(backtest_start),
            "end": str(backtest_end),
        },
        "executor": EXECUTOR_CONFIG,
        "strategy": strategy_effective,
        "metrics": metrics,
    }

    validate_baseline(config_path, metrics)

    save_outputs(
        output_dir,
        summary=summary,
        effective_config=effective_config,
        artifacts={
            "report": result["report"],
            "positions": result["positions"],
            "indicators_df": result["indicators_df"],
            "indicators_obj": result["indicators_obj"],
            "port_analysis": result["port_analysis"],
            "indicator_analysis_df": result["indicator_analysis_df"],
        },
    )

    static_inspection(Path(__file__).resolve())
    print_compact_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
