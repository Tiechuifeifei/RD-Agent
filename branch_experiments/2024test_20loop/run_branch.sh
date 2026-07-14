#!/usr/bin/env bash
# Branch experiment launcher — isolated from main RD-Agent log / .env overrides.
set -euo pipefail

BRANCH_ROOT="/Users/yufei/RD-Agent/branch_experiments/2024test_20loop"
RDAGENT_ROOT="/Users/yufei/RD-Agent"
RDAGENT_PYTHON="/opt/anaconda3/envs/qlib/bin/python"

export LOG_TRACE_PATH="${LOG_TRACE_PATH:-${BRANCH_ROOT}/log}"
export QLIB_FACTOR_TRAIN_START=2008-01-01
export QLIB_FACTOR_TRAIN_END=2017-12-31
export QLIB_FACTOR_VALID_START=2022-01-01
export QLIB_FACTOR_VALID_END=2023-12-31
export QLIB_FACTOR_TEST_START=2024-01-01
export QLIB_FACTOR_TEST_END=2025-12-30

# Isolate factor_runner pickle cache from main experiments.
# Baseline runs use md5("") as cache key; sharing causes stale IC (e.g. 0.031) to leak in.
export PICKLE_CACHE_FOLDER_PATH_STR="${BRANCH_ROOT}/pickle_cache"

if [[ ! -x "$RDAGENT_PYTHON" ]]; then
  echo "ERROR: qlib env python not found: $RDAGENT_PYTHON" >&2
  exit 1
fi

if [[ ! -f "$RDAGENT_ROOT/rdagent/app/qlib_rd_loop/factor.py" ]]; then
  echo "ERROR: factor.py not found under $RDAGENT_ROOT" >&2
  exit 1
fi

mkdir -p "$PICKLE_CACHE_FOLDER_PATH_STR"

"$RDAGENT_PYTHON" - <<'PY'
import os
import rdagent.components.workflow.rd_loop as m
from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS

if not hasattr(RDLoop, "_init_base_features"):
    raise SystemExit(
        "RDLoop missing _init_base_features — likely site-packages rdagent 0.8.0; "
        "use conda env qlib with editable /Users/yufei/RD-Agent"
    )

rd_loop_path = m.__file__.replace("\\", "/")
if "/RD-Agent/rdagent" not in rd_loop_path:
    raise SystemExit(f"rd_loop not loaded from editable repo: {rd_loop_path}")

fbps = FactorBasePropSetting()
expected = {
    "train_end": "2017-12-31",
    "valid_start": "2022-01-01",
    "valid_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2025-12-30",
}
for key, val in expected.items():
    got = getattr(fbps, key)
    if got != val:
        raise SystemExit(
            f"Branch time split mismatch: {key}={got!r}, expected {val!r}. "
            "Use run_branch.sh (dotenv --no-override) so QLIB_FACTOR_* exports win over .env."
        )

cache_path = RD_AGENT_SETTINGS.pickle_cache_folder_path_str
if "branch_experiments/2024test_20loop/pickle_cache" not in cache_path.replace("\\", "/"):
    raise SystemExit(f"Pickle cache not isolated to branch dir: {cache_path}")

print(f"preflight OK: python={__import__('sys').executable}")
print(f"preflight OK: rd_loop={rd_loop_path}")
print(f"preflight OK: test={fbps.test_start}..{fbps.test_end}")
print(f"preflight OK: pickle_cache={cache_path}")
PY

cd "$RDAGENT_ROOT"
# --no-override: keep shell QLIB_FACTOR_* above .env main-experiment dates.
exec dotenv run --no-override -- "$RDAGENT_PYTHON" rdagent/app/qlib_rd_loop/factor.py "$@"
