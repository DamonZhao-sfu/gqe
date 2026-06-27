#!/usr/bin/env bash
# Compare the original hardcoded TPC-DS query vs its fused-kernel (UDR) variant.
#
# For each base query that has a <base>_udr counterpart, run both binaries several times on the
# same dataset, parse GQE's "Query execution time: N ms." log line, and report min/median + the
# speedup. If that log line isn't found, falls back to wall-clock timing of the whole process.
#
# Prereqs:
#   conda activate gqe
#   ./build_gqe.sh                # or build_query.sh for the specific targets
#   python ../common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
#
# Usage:
#   ./compare_udr.sh /data/tpcds_sf1                 # auto-detect all *_udr pairs that are built
#   ./compare_udr.sh /data/tpcds_sf1 q7 q43          # only these bases
#
# Env knobs:
#   GQE_SRC   gqe repo root        (default: repo containing this script)
#   BIN_DIR   benchmark bin dir    (default: $GQE_SRC/build/benchmark)
#   RUNS      timed runs each      (default: 5)
#   WARMUP    discarded warmups    (default: 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
BIN_DIR="${BIN_DIR:-$GQE_SRC/build/benchmark}"

# Make gqe's vendored nvcomp (5.2) resolvable at runtime (build-tree binaries lack it on RPATH).
while IFS= read -r _so; do
  _d="$(dirname "$_so")"
  case ":${LD_LIBRARY_PATH:-}:" in *":$_d:"*) ;; *) export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
done < <(find "$GQE_SRC/build" -name 'libnvcomp*.so*' 2>/dev/null)

RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
export GQE_LOG_LEVEL="${GQE_LOG_LEVEL:-info}"  # ensure the timing line is emitted

DATA_DIR="${1:?usage: compare_udr.sh <tpcds_dataset_dir> [base_query ...]}"
shift || true
[[ -d "$DATA_DIR" ]] || { echo "ERROR: dataset dir not found: $DATA_DIR" >&2; exit 1; }

# Determine which base queries to compare.
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

WORK="$(mktemp -d)"        # binaries write output.parquet/bandwidth.json into CWD
trap 'rm -rf "$WORK"' EXIT

# Run one binary once; echo elapsed ms (engine time if logged, else wall clock). On failure,
# stash the program output in LAST_ERR and return 1 (so callers can show WHY it failed).
LAST_ERR=""
run_once() {
  local bin="$1" t0 t1 out ms
  t0=$(date +%s%3N)
  if ! out=$(cd "$WORK" && "$bin" "$DATA_DIR" 2>&1); then LAST_ERR="$out"; return 1; fi
  t1=$(date +%s%3N)
  ms=$(printf '%s\n' "$out" | grep -oE 'Query execution time: [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
  [[ -z "$ms" ]] && ms=$(( t1 - t0 ))   # fallback: wall clock
  echo "$ms"
}

# min and median of the integers passed as args.
stats() {
  local sorted; sorted=$(printf '%s\n' "$@" | sort -n)
  local n; n=$(printf '%s\n' "$sorted" | wc -l)
  local min; min=$(printf '%s\n' "$sorted" | head -1)
  local med; med=$(printf '%s\n' "$sorted" | sed -n "$(( (n+1)/2 ))p")
  echo "$min $med"
}

printf '%-10s | %-16s | %-16s | %-8s | %-10s | %-7s\n' \
  "query" "orig min/med" "udr min/med" "speedup" "Δmin (ms)" "faster"
printf -- '-----------+------------------+------------------+----------+------------+--------\n'

rc=0
for base in "${BASES[@]}"; do
  bin_o="$BIN_DIR/$base"; bin_u="$BIN_DIR/${base}_udr"
  if [[ ! -x "$bin_o" || ! -x "$bin_u" ]]; then
    echo "skip $base: missing $bin_o or $bin_u" >&2; rc=1; continue
  fi

  # Smoke-test both binaries; if either fails to run, show the real error and skip (don't abort).
  if ! run_once "$bin_o" >/dev/null; then
    echo "skip $base: '$base' failed to run:" >&2
    printf '%s\n' "$LAST_ERR" | tail -3 | sed 's/^/    /' >&2; rc=1; continue
  fi
  if ! run_once "$bin_u" >/dev/null; then
    echo "skip $base: '${base}_udr' failed to run:" >&2
    printf '%s\n' "$LAST_ERR" | tail -3 | sed 's/^/    /' >&2; rc=1; continue
  fi

  for _ in $(seq 1 "$WARMUP"); do run_once "$bin_o" >/dev/null || true; run_once "$bin_u" >/dev/null || true; done

  o_times=(); u_times=()
  for _ in $(seq 1 "$RUNS"); do o_times+=("$(run_once "$bin_o" || echo 0)"); done
  for _ in $(seq 1 "$RUNS"); do u_times+=("$(run_once "$bin_u" || echo 0)"); done

  read -r o_min o_med < <(stats "${o_times[@]}")
  read -r u_min u_med < <(stats "${u_times[@]}")
  # speedup / absolute saving / percent faster, all on the min (best) times
  read -r speedup delta pct < <(awk -v a="$o_min" -v b="$u_min" 'BEGIN{
    if (b>0) printf "%.2fx %d %.1f%%", a/b, a-b, (1-b/a)*100; else print "n/a n/a n/a" }')

  printf '%-10s | %-16s | %-16s | %-8s | %-10s | %-7s\n' \
    "$base" "$o_min / $o_med" "$u_min / $u_med" "$speedup" "$delta" "$pct"
done

echo
echo "(min/median over $RUNS runs after $WARMUP warmup. speedup/Δmin/faster computed on the min.)"
echo "NOTE: end-to-end time includes Parquet read, identical for both, so at small scale factors"
echo "the join speedup is masked. To SEE the difference: use a bigger dataset (gen_tpcds_data.py"
echo "--sf 100), and/or isolate GPU kernel time with: tpch_bench/gqe/profile_udr.sh <data> <base>"
