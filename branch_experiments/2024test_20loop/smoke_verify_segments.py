#!/usr/bin/env python3
"""Minimal smoke test: verify branch time segments reach qrun / pred.pkl."""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

from jinja2 import Template, meta
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent
SMOKE_DIR = ROOT / "smoke_qrun"
CONF = SMOKE_DIR / "conf_baseline.yaml"

BRANCH_ENV = {
    "PYTHONPATH": "./",
    "train_start": "2008-01-01",
    "train_end": "2017-12-31",
    "valid_start": "2022-01-01",
    "valid_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2025-12-31",
}


def render_conf(config_path: Path, env: dict[str, str]) -> dict:
    text = config_path.read_text()
    template = Template(text)
    parsed = template.environment.parse(text)
    variables = meta.find_undeclared_variables(parsed)
    context = {var: env.get(var, os.getenv(var, "")) for var in variables}
    rendered = template.render(context)
    yaml = YAML(typ="safe", pure=True)
    return yaml.load(rendered), context


def main() -> int:
    sys.path.insert(0, str(Path("/Users/yufei/RD-Agent")))
    from rdagent.utils.qlib import ALPHA20

    env = BRANCH_ENV.copy()
    env["feature_names"] = str(list(ALPHA20.keys()))
    env["feature_expressions"] = str(list(ALPHA20.values()))

    config, context = render_conf(CONF, env)
    segments = config["task"]["dataset"]["kwargs"]["segments"]
    backtest = config["port_analysis_config"]["backtest"]
    print("=== Rendered segments (pre-qrun) ===")
    print("train:", segments["train"])
    print("valid:", segments["valid"])
    print("test:", segments["test"])
    print("backtest:", backtest["start_time"], backtest["end_time"])
    print("jinja context keys:", sorted(context.keys()))

    def norm(seg):
        return [str(x)[:10] for x in seg]

    expected = {
        "train": ["2008-01-01", "2017-12-31"],
        "valid": ["2022-01-01", "2023-12-31"],
        "test": ["2024-01-01", "2025-12-31"],
    }
    for k, v in expected.items():
        if norm(segments[k]) != v:
            print(f"FAIL segment {k}: got {norm(segments[k])}, want {v}")
            return 1

    print("\n=== Running qrun conf_baseline.yaml (smoke) ===")
    proc = subprocess.run(
        ["conda", "run", "-n", "rdagent4qlib", "--no-capture-output", "qrun", "conf_baseline.yaml"],
        cwd=SMOKE_DIR,
        env={**os.environ, **{k: str(v) for k, v in env.items()}},
        capture_output=True,
        text=True,
    )
    print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-3000:] if len(proc.stderr) > 3000 else proc.stderr)
        print("qrun failed, exit", proc.returncode)
        return proc.returncode

    mlruns = SMOKE_DIR / "mlruns"
    pred_paths = list(mlruns.rglob("pred.pkl"))
    if not pred_paths:
        print("FAIL: pred.pkl not found under", mlruns)
        return 1

    pred_path = max(pred_paths, key=lambda p: p.stat().st_mtime)
    print("\n=== pred.pkl check ===")
    print("path:", pred_path)
    with pred_path.open("rb") as f:
        pred = pickle.load(f)
    idx = pred.index
    dt_min = idx.get_level_values("datetime").min()
    dt_max = idx.get_level_values("datetime").max()
    print("pred datetime range:", dt_min, "->", dt_max)

    if str(dt_min)[:10] < "2024-01-01" or str(dt_max)[:10] > "2025-12-31":
        print("FAIL: pred dates not within test window 2024-2025")
        return 1
    if str(dt_min)[:10] >= "2020-01-01" and str(dt_max)[:10] <= "2023-12-31":
        print("FAIL: pred looks like main-experiment test window 2020-2023")
        return 1

    print("PASS: branch time segments verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
