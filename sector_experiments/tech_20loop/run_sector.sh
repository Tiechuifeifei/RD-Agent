#!/usr/bin/env bash
# Sector 20-loop launcher (tech or bank). Independent from main + branch experiments.
# Uses MAIN experiment time split and TopK=20/n_drop=2; only instruments differ.
#
# Usage:
#   ./run_sector.sh tech --loop_n 20
#   ./run_sector.sh bank --loop_n 20
#
set -euo pipefail

SECTOR="${1:?usage: run_sector.sh tech|bank [factor.py args...]}"
shift

case "$SECTOR" in
  tech)
    MARKET="sp500_tech"
    SECTOR_ROOT="/Users/yufei/RD-Agent/sector_experiments/tech_20loop"
    ;;
  bank)
    MARKET="sp500_bank"
    SECTOR_ROOT="/Users/yufei/RD-Agent/sector_experiments/bank_20loop"
    ;;
  *)
    echo "ERROR: sector must be tech or bank, got: $SECTOR" >&2
    exit 1
    ;;
esac

RDAGENT_ROOT="/Users/yufei/RD-Agent"
RDAGENT_PYTHON="/opt/anaconda3/envs/qlib/bin/python"
WRDS_ROOT="/Users/yufei/wrds_project"

export LOG_TRACE_PATH="${LOG_TRACE_PATH:-${SECTOR_ROOT}/log}"
# Main experiment windows (conf_alpha20_sp500_transfer.yaml)
export QLIB_FACTOR_TRAIN_START=2008-01-01
export QLIB_FACTOR_TRAIN_END=2017-12-31
export QLIB_FACTOR_VALID_START=2018-01-01
export QLIB_FACTOR_VALID_END=2019-12-31
export QLIB_FACTOR_TEST_START=2020-01-01
export QLIB_FACTOR_TEST_END=2023-12-31
export QLIB_FACTOR_MARKET="${MARKET}"
export PICKLE_CACHE_FOLDER_PATH_STR="${SECTOR_ROOT}/pickle_cache"

if [[ ! -x "$RDAGENT_PYTHON" ]]; then
  echo "ERROR: qlib env python not found: $RDAGENT_PYTHON" >&2
  exit 1
fi

INST_FILE="${WRDS_ROOT}/staging/qlib_data/instruments/${MARKET}.txt"
if [[ ! -f "$INST_FILE" ]]; then
  echo "ERROR: missing instruments file: $INST_FILE" >&2
  echo "Run: python ${WRDS_ROOT}/scripts/build_permno_sector_map.py" >&2
  exit 1
fi

mkdir -p "$PICKLE_CACHE_FOLDER_PATH_STR" "$SECTOR_ROOT"

"$RDAGENT_PYTHON" - <<PY
import rdagent.components.workflow.rd_loop as m
from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS

if not hasattr(RDLoop, "_init_base_features"):
    raise SystemExit(
        "RDLoop missing _init_base_features — use conda env qlib with editable RD-Agent"
    )

fbps = FactorBasePropSetting()
expected = {
    "train_end": "2017-12-31",
    "valid_start": "2018-01-01",
    "valid_end": "2019-12-31",
    "test_start": "2020-01-01",
    "test_end": "2023-12-31",
    "market": "${MARKET}",
}
for key, val in expected.items():
    got = getattr(fbps, key)
    if got != val:
        raise SystemExit(f"Preflight mismatch: {key}={got!r}, expected {val!r}")

cache_path = RD_AGENT_SETTINGS.pickle_cache_folder_path_str
if "${SECTOR_ROOT}/pickle_cache" not in cache_path.replace("\\\\", "/"):
    raise SystemExit(f"Pickle cache not isolated: {cache_path}")

print(f"preflight OK: sector=${SECTOR} market={fbps.market}")
print(f"preflight OK: test={fbps.test_start}..{fbps.test_end} (main experiment window)")
print(f"preflight OK: pickle_cache={cache_path}")
PY

cd "$RDAGENT_ROOT"
exec dotenv run --no-override -- "$RDAGENT_PYTHON" rdagent/app/qlib_rd_loop/factor.py "$@"
