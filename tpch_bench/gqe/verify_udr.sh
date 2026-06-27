#!/usr/bin/env bash
# Verify a fused-kernel (UDR) query produces the SAME result as the original, and SAVE both
# results so you can inspect them. For each base query that has a <base>_udr counterpart, runs both
# binaries, copies each one's output.parquet into a results dir, and compares them.
#
# Prereqs:
#   conda activate gqe
#   ./build_query.sh q3 q3_udr q7 q7_udr q43 q43_udr      # (or build_gqe[_v2].sh)
#   python ../common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
#
# Usage:
#   ./verify_udr.sh /data/tpcds_sf1                 # auto-detect all built *_udr pairs
#   ./verify_udr.sh /data/tpcds_sf1 q3 q7 q43       # only these bases
#
# Env knobs:
#   GQE_SRC      gqe repo root      (default: repo containing this script)
#   BIN_DIR      benchmark bin dir  (default: $GQE_SRC/build/benchmark)
#   RESULTS_DIR  where to save the per-query Parquet results (default: ./udr_results)
#   RTOL/ATOL    numeric tolerance for the comparison (default: 1e-6 / 1e-3)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
BIN_DIR="${BIN_DIR:-$GQE_SRC/build/benchmark}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/udr_results}"
RTOL="${RTOL:-1e-6}"
ATOL="${ATOL:-1e-3}"
CMP_PY="$GQE_SRC/tpch_bench/common/compare_parquet.py"
export GQE_LOG_LEVEL="${GQE_LOG_LEVEL:-info}"

# Make gqe's vendored nvcomp (5.2) resolvable at runtime (build-tree binaries lack it on RPATH).
while IFS= read -r _so; do
  _d="$(dirname "$_so")"
  case ":${LD_LIBRARY_PATH:-}:" in *":$_d:"*) ;; *) export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
done < <(find "$GQE_SRC/build" -name 'libnvcomp*.so*' 2>/dev/null)

DATA_DIR="${1:?usage: verify_udr.sh <tpcds_dataset_dir> [base_query ...]}"
shift || true
[[ -d "$DATA_DIR" ]] || { echo "ERROR: dataset dir not found: $DATA_DIR" >&2; exit 1; }

BASES=("$@")
if [[ ${#BASES[@]} -eq 0 ]]; then
  shopt -s nullglob
  for udr in "$BIN_DIR"/*_udr; do
    base="$(basename "${udr%_udr}")"
    [[ -x "$BIN_DIR/$base" ]] && BASES+=("$base")
  done
  shopt -u nullglob
fi
[[ ${#BASES[@]} -gt 0 ]] || { echo "No *_udr pairs found in $BIN_DIR (build them first)." >&2; exit 1; }

mkdir -p "$RESULTS_DIR"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# Run $1 (binary path), save its output.parquet to $2.
run_save() {
  local bin="$1" dest="$2"
  ( cd "$WORK" && rm -f output.parquet && "$bin" "$DATA_DIR" >/dev/null 2>&1 )
  if [[ -f "$WORK/output.parquet" ]]; then cp -f "$WORK/output.parquet" "$dest"; return 0; fi
  echo "  (no output.parquet produced by $bin)" >&2; return 1
}

echo "Saving results under: $RESULTS_DIR"
echo
rc=0
for base in "${BASES[@]}"; do
  bin_o="$BIN_DIR/$base"; bin_u="$BIN_DIR/${base}_udr"
  [[ -x "$bin_o" && -x "$bin_u" ]] || { echo "skip $base: missing binary"; continue; }

  out_o="$RESULTS_DIR/${base}.parquet"
  out_u="$RESULTS_DIR/${base}_udr.parquet"
  echo "== $base =="
  run_save "$bin_o" "$out_o" || { rc=1; continue; }
  run_save "$bin_u" "$out_u" || { rc=1; continue; }
  echo "  saved: $out_o"
  echo "  saved: $out_u"
  if python "$CMP_PY" "$out_o" "$out_u" --rtol "$RTOL" --atol "$ATOL" | sed 's/^/  /'; then
    :
  else
    rc=1
  fi
  echo
done

echo "Done. Results saved in $RESULTS_DIR (inspect with e.g. python -c \"import pandas,pyarrow.parquet as pq; print(pq.read_table('$RESULTS_DIR/q3.parquet').to_pandas())\")."
exit "$rc"
