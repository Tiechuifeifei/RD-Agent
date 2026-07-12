#!/usr/bin/env python3
"""Experiment 2 densification: n_drop = 3,4,6,7,8,15 + merged 11-point curve."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "experiment2_ndrop"
PRED = ROOT / "inputs" / "baseline_pred.pkl"
HASH_FILE = ROOT / "inputs" / "input_hashes.txt"
RUNNER = ROOT / "scripts" / "run_portfolio_ablation.py"
PYTHON = "/opt/anaconda3/envs/rdagent4qlib/bin/python"

NEW_NDROPS = [3, 4, 6, 7, 8, 15]
ORIGINAL_NDROPS = [1, 2, 5, 10, 20]
ALL_NDROPS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]

NDROP_CONFIG = {
    1: ROOT / "configs" / "ndrop1.yaml",
    2: ROOT / "configs" / "baseline.yaml",
    3: ROOT / "configs" / "ndrop3.yaml",
    4: ROOT / "configs" / "ndrop4.yaml",
    5: ROOT / "configs" / "ndrop5.yaml",
    6: ROOT / "configs" / "ndrop6.yaml",
    7: ROOT / "configs" / "ndrop7.yaml",
    8: ROOT / "configs" / "ndrop8.yaml",
    10: ROOT / "configs" / "ndrop10.yaml",
    15: ROOT / "configs" / "ndrop15.yaml",
    20: ROOT / "configs" / "ndrop20.yaml",
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
    rels = ["inputs/baseline_pred.pkl"] + [f"configs/{'baseline' if n == 2 else f'ndrop{n}'}.yaml" for n in ALL_NDROPS]
    checks = []
    for rel in rels:
        path = ROOT / rel
        actual = sha256_file(path)
        expected = manifest.get(rel)
        checks.append({"relative": rel, "match": actual == expected, "sha256": actual})
    return {"all_match": all(c["match"] for c in checks), "pred_sha256": sha256_file(PRED), "checks": checks}


def run_ndrop(ndrop: int) -> None:
    cfg = NDROP_CONFIG[ndrop]
    out_dir = OUT / f"ndrop{ndrop}"
    if (out_dir / "summary.json").exists() and ndrop in ORIGINAL_NDROPS:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(RUNNER), "--config", str(cfg), "--pred", str(PRED), "--output", str(out_dir), "--verify-hash"]
    print(f"Running n_drop={ndrop}...")
    subprocess.run(cmd, check=True)


def load_summary_row(ndrop: int) -> dict:
    summary = json.loads((OUT / f"ndrop{ndrop}" / "summary.json").read_text(encoding="utf-8"))
    m = summary["metrics"]
    return {
        "n_drop": ndrop,
        "topk": 20,
        "pred_hash": summary["pred_hash"],
        "config_hash": summary["config_hash"],
        "annualized_return_with_cost": m["annualized_return_with_cost"],
        "annualized_return_without_cost": m["annualized_return_without_cost"],
        "annualized_cost_drag": m["annualized_cost_drag"],
        "information_ratio_with_cost": m["information_ratio_with_cost"],
        "maximum_drawdown_with_cost": m["maximum_drawdown_with_cost"],
        "mean_daily_turnover": m["mean_daily_turnover"],
        "average_number_of_positions": m.get("average_number_of_positions"),
        "densification_new_point": ndrop in NEW_NDROPS,
    }


def analyze_shape(df: pd.DataFrame) -> dict:
    df = df.sort_values("n_drop")
    pts = list(zip(df["n_drop"].astype(int), (df["annualized_return_with_cost"] * 100).round(4)))
    dirs = []
    for i in range(1, len(pts)):
        dirs.append("up" if pts[i][1] > pts[i - 1][1] else "down")

    # local peaks and valleys (interior points only)
    arr = df["annualized_return_with_cost"].values
    ndrops = df["n_drop"].astype(int).values
    peaks, valleys = [], []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append({"n_drop": int(ndrops[i]), "arr_pct": round(arr[i] * 100, 4)})
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            valleys.append({"n_drop": int(ndrops[i]), "arr_pct": round(arr[i] * 100, 4)})

    best = df.loc[df["annualized_return_with_cost"].idxmax()]
    worst = df.loc[df["annualized_return_with_cost"].idxmin()]

    orig = df[df["n_drop"].isin(ORIGINAL_NDROPS)].sort_values("n_drop")
    orig_dirs = []
    op = list(zip(orig["n_drop"].astype(int), orig["annualized_return_with_cost"] * 100))
    for i in range(1, len(op)):
        orig_dirs.append("up" if op[i][1] > op[i - 1][1] else "down")

    return {
        "n_points": len(df),
        "ndrop_values": [int(x) for x in df["n_drop"]],
        "arr_pct_by_ndrop": {int(r.n_drop): float(r.annualized_return_with_cost * 100) for r in df.itertuples()},
        "segment_directions": {f"{pts[i][0]}->{pts[i+1][0]}": dirs[i] for i in range(len(dirs))},
        "direction_sequence": "".join("↑" if d == "up" else "↓" for d in dirs),
        "original_5pt_direction_sequence": "".join("↑" if d == "up" else "↓" for d in orig_dirs),
        "global_peak": {"n_drop": int(best["n_drop"]), "arr_pct": round(best["annualized_return_with_cost"] * 100, 4)},
        "global_valley": {"n_drop": int(worst["n_drop"]), "arr_pct": round(worst["annualized_return_with_cost"] * 100, 4)},
        "local_peaks": peaks,
        "local_valleys": valleys,
        "original_peak_ndrop10_still_global_max": int(best["n_drop"]) == 10,
        "original_valley_ndrop5_still_global_min": int(worst["n_drop"]) == 5,
        "verdict": _shape_verdict(dirs, peaks, valleys, int(best["n_drop"]), int(worst["n_drop"])),
    }


def _shape_verdict(dirs, peaks, valleys, best_ndrop, worst_ndrop) -> str:
    n_flip = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
    if n_flip >= 4:
        shape = "still_irregular_sawtooth"
    elif len(peaks) <= 1 and len(valleys) <= 1:
        shape = "smoother_single_peak_pattern"
    else:
        shape = "multi_peak_irregular"
    return (
        f"{shape}; global peak n_drop={best_ndrop}, global valley n_drop={worst_ndrop}; "
        f"densification did not produce a smooth inverted-U"
        if n_flip >= 3 else
        f"{shape}; global peak n_drop={best_ndrop}, global valley n_drop={worst_ndrop}"
    )


def plot_curve(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values("n_drop")
    x = df["n_drop"].values
    y = df["annualized_return_with_cost"].values * 100
    new_mask = df["densification_new_point"].values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, linewidth=1.5, color="#94a3b8", zorder=1)
    ax.scatter(x[~new_mask], y[~new_mask], s=80, color="#2563eb", zorder=3, label="original 5 points")
    ax.scatter(x[new_mask], y[new_mask], s=80, color="#f59e0b", marker="s", zorder=3, label="densification 6 points")
    for xi, yi, is_new in zip(x, y, new_mask):
        color = "#b45309" if is_new else "#1d4ed8"
        ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color=color)
    if 2 in x:
        base = df.loc[df.n_drop == 2, "annualized_return_with_cost"].iloc[0] * 100
        ax.axhline(base, color="gray", linestyle="--", alpha=0.5, label=f"baseline n_drop=2 ({base:.2f}%)")
    ax.set_xlabel("n_drop", fontsize=12)
    ax.set_ylabel("ARR excess (with cost, %)", fontsize=12)
    ax.set_title("Experiment 2 Densified: Full-sample ARR vs n_drop (11 points, topk=20)", fontsize=13)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = verify_manifest()
    (OUT / "densification_manifest_verification.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["all_match"]:
        raise SystemExit("Manifest verification failed")
    print("Manifest OK, pred SHA256:", manifest["pred_sha256"])

    for ndrop in NEW_NDROPS:
        run_ndrop(ndrop)

    rows = [load_summary_row(n) for n in ALL_NDROPS]
    dense_df = pd.DataFrame(rows).sort_values("n_drop")
    dense_df.to_csv(OUT / "task1_full_sample_densified.csv", index=False)
    (OUT / "task1_full_sample_densified.json").write_text(dense_df.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n=== Densified full sample (11 points) ===")
    print(dense_df[["n_drop", "annualized_return_with_cost", "annualized_cost_drag", "mean_daily_turnover"]].to_string(index=False))

    shape = analyze_shape(dense_df)
    shape["original_5pt"] = {
        "peak_ndrop": 10,
        "valley_ndrop": 5,
        "peak_arr_pct": -10.9235,
        "valley_arr_pct": -12.5607,
    }
    (OUT / "task3_shape_verification.json").write_text(json.dumps(shape, indent=2, default=str), encoding="utf-8")

    # update combined relationship json
    rel_path = OUT / "task3_relationship_analysis.json"
    rel = json.loads(rel_path.read_text(encoding="utf-8")) if rel_path.exists() else {}
    rel["densification"] = shape
    rel["full_sample_arr_by_ndrop_pct_densified"] = shape["arr_pct_by_ndrop"]
    rel["shape_verdict"] = shape["verdict"]
    rel_path.write_text(json.dumps(rel, indent=2, default=str), encoding="utf-8")

    plot_curve(dense_df, OUT / "task3_arr_vs_ndrop_curve.png")
    print("\nShape analysis:")
    print(json.dumps(shape, indent=2, default=str))
    print(f"\nSaved curve: {OUT / 'task3_arr_vs_ndrop_curve.png'}")


if __name__ == "__main__":
    main()
