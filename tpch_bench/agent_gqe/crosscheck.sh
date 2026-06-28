#!/usr/bin/env bash
# Phase 0: validate the reference pipeline on KNOWN-GOOD GPU code. Runs the existing hardcoded
# TPC-DS benchmark binaries (q3, q6, q7, q22, q38, q43, q48), and compares each one's output to the
# DuckDB CPU reference. If these match, the DuckDB reference + comparator are trustworthy for
# checking agent-generated plans.
#
# Prereqs: conda activate gqe ; gqe built (build/benchmark/qN exist) ; a TPC-DS parquet dataset.
#
# Usage:
#   ./crosscheck.sh /data/tpcds_sf1                 # default set: 3 6 7 22 38 43 48
#   ./crosscheck.sh /data/tpcds_sf1 3 7 43
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
BIN_DIR="${BIN_DIR:-$GQE_SRC/build/benchmark}"
REF_DIR="${REF_DIR:-$PWD/tpcds_ref}"
CMP_PY="$GQE_SRC/tpch_bench/common/compare_parquet.py"
GEN_REF="$HERE/gen_reference.py"
export GQE_LOG_LEVEL="${GQE_LOG_LEVEL:-warn}"

# gqe's vendored nvcomp on the loader path.
while IFS= read -r _so; do
  _d="$(dirname "$_so")"
  case ":${LD_LIBRARY_PATH:-}:" in *":$_d:"*) ;; *) export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
done < <(find "$GQE_SRC/build" -name 'libnvcomp*.so*' 2>/dev/null)

DATA="${1:?usage: crosscheck.sh <tpcds_dataset_dir> [query ...]}"; shift || true
QUERIES=("$@"); [[ ${#QUERIES[@]} -gt 0 ]] || QUERIES=(3 6 7 22 38 43 48)

mkdir -p "$REF_DIR"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

rc=0
for n in "${QUERIES[@]}"; do
  bin="$BIN_DIR/q$n"
  echo "== q$n =="
  [[ -x "$bin" ]] || { echo "  skip: $bin not built"; rc=1; continue; }
  ref="$REF_DIR/q$n.parquet"
  [[ -f "$ref" ]] || python "$GEN_REF" --data "$DATA" --query "$n" --outdir "$REF_DIR" >/dev/null
  if ! ( cd "$WORK" && rm -f output.parquet && "$bin" "$DATA" ) >/dev/null 2>"$WORK/err"; then
    echo "  RUN FAILED:"; tail -3 "$WORK/err" | sed 's/^/    /'; rc=1; continue
  fi
  if python "$CMP_PY" "$WORK/output.parquet" "$ref" | sed 's/^/  /'; then :; else rc=1; fi
done

echo
echo "References saved in $REF_DIR/. MATCH on all => reference pipeline is trustworthy."
exit "$rc"
