#!/usr/bin/env bash
# Run GQE's standalone hardcoded benchmark programs (benchmark/hardcoded/*).
#
# IMPORTANT: these are all TPC-DS queries (q3, q6, q7, q22, q38, q43, q48, and
# q3_udr -- the custom-kernel variant of q3). They link the gqe library directly
# and run IN-PROCESS: no node_manager / Flight SQL server is involved.
#
# Each program takes a single arg: a TPC-DS Parquet dataset directory laid out
# as <dataset>/<table>/*.parquet. Generate it with:
#   python ../common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
#
# Usage:
#   ./run_gqe_benchmark.sh /data/tpcds_sf1 3
#   ./run_gqe_benchmark.sh /data/tpcds_sf1 3 6 22
#   ./run_gqe_benchmark.sh /data/tpcds_sf1 udr        # q3_udr (custom kernel)
#   ./run_gqe_benchmark.sh /data/tpcds_sf1 all        # all hardcoded queries
#
# Env knobs:
#   GQE_SRC    path to gqe repo  (default: repo containing this script)
#   BIN_DIR    benchmark bin dir (default: $GQE_SRC/build/benchmark)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
BIN_DIR="${BIN_DIR:-$GQE_SRC/build/benchmark}"

# query-number -> binary name. The fused-kernel variants q3_udr / q7_udr can be
# selected by their full name, e.g. `run_gqe_benchmark.sh <data> q3_udr q7_udr`.
ALL=(q3 q6 q7 q22 q38 q43 q48 q3_udr q7_udr)

DATA_DIR="${1:?usage: run_gqe_benchmark.sh <tpcds_dataset_dir> <query|udr|all> [more...]}"
shift
REQ=("$@")
[[ ${#REQ[@]} -gt 0 ]] || { echo "ERROR: specify query number(s), 'udr', or 'all'" >&2; exit 1; }

# Expand selectors into a list of binary names.
BINS=()
for q in "${REQ[@]}"; do
  case "$q" in
    all) BINS=("${ALL[@]}"); break ;;
    udr|q3_udr) BINS+=("q3_udr") ;;
    q*) BINS+=("$q") ;;
    *) BINS+=("q$q") ;;
  esac
done

[[ -d "$DATA_DIR" ]] || { echo "ERROR: dataset dir not found: $DATA_DIR" >&2; exit 1; }

rc=0
for b in "${BINS[@]}"; do
  bin="$BIN_DIR/$b"
  if [[ ! -x "$bin" ]]; then
    echo "WARN: $bin not found -- build GQE first (build_gqe.sh) or check name." >&2
    rc=1
    continue
  fi
  echo "==> $b  (dataset: $DATA_DIR)"
  "$bin" "$DATA_DIR" || { echo "ERROR: $b failed" >&2; rc=1; }
  echo
done
exit "$rc"
