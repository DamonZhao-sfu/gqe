#!/usr/bin/env bash
# Run one (or more, or all) TPC-H queries on GQE end-to-end:
#   start node_manager -> load Parquet data -> run query via gqe-cli -> stop.
#
# Prereqs:
#   conda activate gqe
#   ./build_gqe.sh                          # engine + gqe-cli built
#   python ../common/gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1 --queries
#
# Usage:
#   ./run_gqe_tpch.sh /data/tpch_sf1 1            # query 1
#   ./run_gqe_tpch.sh /data/tpch_sf1 1 6 14       # several
#   ./run_gqe_tpch.sh /data/tpch_sf1 all          # all 22
#
# Env knobs:
#   GQE_SRC   path to gqe repo   (default: repo containing this script)
#   PORT      server port        (default: 50051)
#   NUM_GPUS  GPUs to use        (default: 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
PORT="${PORT:-50051}"
NUM_GPUS="${NUM_GPUS:-1}"

DATA_DIR="${1:?usage: run_gqe_tpch.sh <data_dir> <query|all> [query ...]}"
shift
QUERIES=("$@")
[[ ${#QUERIES[@]} -gt 0 ]] || { echo "ERROR: specify query number(s) or 'all'" >&2; exit 1; }

BUILD_DIR="$GQE_SRC/build"
NODE_MGR="$BUILD_DIR/src/node_manager/gqe_node_manager"
TASK_MGR="$BUILD_DIR/src/task_manager/gqe_task_manager"
export GQE_CLI="${GQE_CLI:-$GQE_SRC/rust/target/release/gqe-cli}"

for b in "$NODE_MGR" "$TASK_MGR" "$GQE_CLI"; do
  [[ -x "$b" ]] || { echo "ERROR: missing binary $b -- run build_gqe.sh first" >&2; exit 1; }
done

LOG=/tmp/gqe_node_manager.log
echo "==> starting node manager on 127.0.0.1:$PORT (log: $LOG)"
"$NODE_MGR" \
  --address 127.0.0.1 --port "$PORT" --num-gpus "$NUM_GPUS" \
  --task-manager-binary "$TASK_MGR" > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { echo "==> stopping node manager ($SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> waiting for server ..."
for _ in $(seq 1 120); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&- 3<&-; break; fi
  sleep 0.5
done

echo "==> loading TPC-H data from $DATA_DIR"
SCHEMA="schema.sql"; [[ -f "$DATA_DIR/$SCHEMA" ]] || SCHEMA="ci_schema.sql"
"$GQE_SRC/scripts/load_tpch.py" \
  --server-url "http://127.0.0.1:$PORT" --schema "$SCHEMA" "$DATA_DIR"

echo "==> running queries: ${QUERIES[*]}"
"$GQE_SRC/scripts/run_tpch.py" \
  --mode sql --server-url "http://127.0.0.1:$PORT" \
  "$DATA_DIR/queries" "${QUERIES[@]}"

echo "==> done."
