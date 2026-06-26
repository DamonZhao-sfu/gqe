#!/usr/bin/env bash
# Run cudf's standalone TPC-H program(s) for one or more queries.
#
# Works with the cpp/examples/tpch binaries (tpch_qN), which take a dataset
# directory of Parquet files. Generate that data with:
#   python ../common/gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1
#
# The examples expect per-table Parquet; our generator already lays them out as
# <data>/<table>/<table>.parquet. Some example versions expect flat files named
# <table>.parquet in one dir -- this script auto-flattens into a temp dir if so.
#
# Usage:
#   ./run_cudf_ndsh.sh /data/tpch_sf1 1
#   ./run_cudf_ndsh.sh /data/tpch_sf1 1 6 9
#   ./run_cudf_ndsh.sh /data/tpch_sf1 all     # all implemented (1 5 6 9 10)
#
# Env knobs:
#   CUDF_SRC   cudf checkout root  (default: $HOME/cudf-bench)
#   BIN_DIR    dir with tpch_qN    (default: $CUDF_SRC/cpp/examples/tpch/build)
set -euo pipefail

CUDF_SRC="${CUDF_SRC:-$HOME/cudf-bench}"
BIN_DIR="${BIN_DIR:-$CUDF_SRC/cpp/examples/tpch/build}"
IMPLEMENTED=(1 5 6 9 10)   # queries the cudf examples currently ship

DATA_DIR="${1:?usage: run_cudf_ndsh.sh <data_dir> <query|all> [query ...]}"
shift
REQ=("$@")
[[ ${#REQ[@]} -gt 0 ]] || { echo "ERROR: specify query number(s) or 'all'" >&2; exit 1; }
if [[ "${REQ[0]}" == "all" ]]; then REQ=("${IMPLEMENTED[@]}"); fi

# Build a flat dataset dir (<table>.parquet) in case the binary wants that form.
FLAT_DIR="$(mktemp -d)"
trap 'rm -rf "$FLAT_DIR"' EXIT
for t in region nation supplier customer part partsupp orders lineitem; do
  if [[ -f "$DATA_DIR/$t/$t.parquet" ]]; then
    ln -sf "$DATA_DIR/$t/$t.parquet" "$FLAT_DIR/$t.parquet"
  elif [[ -f "$DATA_DIR/$t.parquet" ]]; then
    ln -sf "$DATA_DIR/$t.parquet" "$FLAT_DIR/$t.parquet"
  fi
done

for q in "${REQ[@]}"; do
  bin="$BIN_DIR/tpch_q$q"
  if [[ ! -x "$bin" ]]; then
    echo "WARN: $bin not found (q$q not implemented or different path). " \
         "Check '$BIN_DIR' and the binary's --help." >&2
    continue
  fi
  echo "==> cudf TPC-H q$q"
  # Most example versions accept the dataset dir as the first arg. If yours
  # differs, run \"$bin --help\" and adjust here.
  "$bin" "$FLAT_DIR" || "$bin" -d "$FLAT_DIR" || {
    echo "ERROR: q$q failed; inspect '$bin --help' for its argument form." >&2
  }
done

echo "==> done."
