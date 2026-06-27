#!/usr/bin/env bash
# Show the REAL speedup of the fused kernel by isolating GPU kernel time with Nsight Systems.
#
# End-to-end wall time is dominated by Parquet read (identical for both variants), so at small
# scale factors it hides the join speedup. This profiles each <base> and <base>_udr with nsys and
# compares total GPU kernel time (sum over all CUDA kernels) -- which is where the fused kernel
# actually wins. Saves the .nsys-rep files so you can also open them in the Nsight Systems GUI.
#
# Prereqs:
#   conda activate gqe
#   nsys (Nsight Systems). If missing: conda install -c nvidia nsight-systems
#   built binaries + a TPC-DS dataset (bigger SF shows a clearer gap)
#
# Usage:
#   ./profile_udr.sh /data/tpcds_sf10                # auto-detect built *_udr pairs
#   ./profile_udr.sh /data/tpcds_sf10 q7 q43         # only these bases
#
# Env knobs:
#   GQE_SRC, BIN_DIR (default $GQE_SRC/build/benchmark), OUTDIR (default ./udr_profiles)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
BIN_DIR="${BIN_DIR:-$GQE_SRC/build/benchmark}"
OUTDIR="${OUTDIR:-$PWD/udr_profiles}"
CSV_OUT="${CSV_OUT:-$PWD/udr_profile.csv}"   # kernel-time results also saved here
export GQE_LOG_LEVEL="${GQE_LOG_LEVEL:-warn}"

# Locate nsys: PATH, env var, conda, or a system CUDA / Nsight install.
NSYS="${NSYS:-}"
if [[ -z "$NSYS" ]]; then
  if command -v nsys >/dev/null 2>&1; then
    NSYS="$(command -v nsys)"
  else
    for c in "$CONDA_PREFIX/bin/nsys" /usr/local/cuda*/bin/nsys \
             /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/bin/nsys; do
      [[ -x "$c" ]] && { NSYS="$c"; break; }
    done
  fi
fi
if [[ -z "$NSYS" ]]; then
  echo "ERROR: nsys (Nsight Systems) not found. Options:" >&2
  echo "  - conda install -c nvidia nsight-systems" >&2
  echo "  - or point NSYS=/path/to/nsys ./profile_udr.sh ..." >&2
  echo "  (if you can't install nsys, use compare_udr.sh on a bigger dataset instead.)" >&2
  exit 1
fi
echo "==> using nsys: $NSYS"

# gqe's vendored nvcomp on the loader path (build-tree binaries lack it on RPATH).
while IFS= read -r _so; do
  _d="$(dirname "$_so")"
  case ":${LD_LIBRARY_PATH:-}:" in *":$_d:"*) ;; *) export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
done < <(find "$GQE_SRC/build" -name 'libnvcomp*.so*' 2>/dev/null)

DATA_DIR="${1:?usage: profile_udr.sh <tpcds_dataset_dir> [base_query ...]}"
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
[[ ${#BASES[@]} -gt 0 ]] || { echo "No *_udr pairs found in $BIN_DIR." >&2; exit 1; }

mkdir -p "$OUTDIR"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# Profile $1 (binary) into report $2; echo total GPU kernel time in ns. Never aborts the script:
# any failure (profiling blocked, unknown report name, etc.) yields 0.
LAST_NSYS_ERR=""
kernel_ns() {
  local bin="$1" rep="$2" csv="" report
  if ! ( cd "$WORK" && "$NSYS" profile -t cuda --sample=none --force-overwrite true \
           -o "$rep" "$bin" "$DATA_DIR" ) >"$WORK/nsys.log" 2>&1; then
    LAST_NSYS_ERR="$(tail -3 "$WORK/nsys.log")"; echo 0; return 0
  fi
  # Report name differs across nsys versions: cuda_gpu_kern_sum (new) vs gpukernsum (old).
  for report in cuda_gpu_kern_sum gpukernsum; do
    csv="$("$NSYS" stats --report "$report" --format csv "$rep.nsys-rep" 2>/dev/null || true)"
    [[ -n "$csv" ]] && break
  done
  if [[ -z "$csv" ]]; then LAST_NSYS_ERR="nsys stats produced no output"; echo 0; return 0; fi
  printf '%s\n' "$csv" | python3 -c '
import sys, csv
rows = list(csv.reader(sys.stdin))
hi = next((i for i,r in enumerate(rows) if any("Total Time" in c for c in r)), None)
if hi is None: print(0); sys.exit()
col = next((j for j,c in enumerate(rows[hi]) if c.strip().startswith("Total Time")), None)
if col is None: print(0); sys.exit()
tot = 0.0
for r in rows[hi+1:]:
    if len(r) > col:
        try: tot += float(r[col].replace(",", "").strip())
        except ValueError: pass
print(int(tot))
' 2>/dev/null || echo 0
}

echo "query,orig_kernel_ms,udr_kernel_ms,speedup,pct_faster" > "$CSV_OUT"

printf '%-10s | %-16s | %-16s | %-8s | %-7s\n' \
  "query" "orig kern (ms)" "udr kern (ms)" "speedup" "faster"
printf -- '-----------+------------------+------------------+----------+--------\n'

for base in "${BASES[@]}"; do
  bin_o="$BIN_DIR/$base"; bin_u="$BIN_DIR/${base}_udr"
  [[ -x "$bin_o" && -x "$bin_u" ]] || { echo "skip $base: missing binary"; continue; }

  o_ns="$(kernel_ns "$bin_o" "$OUTDIR/${base}")"
  u_ns="$(kernel_ns "$bin_u" "$OUTDIR/${base}_udr")"

  read -r o_ms u_ms speedup pct < <(awk -v a="$o_ns" -v b="$u_ns" 'BEGIN{
    om=a/1e6; um=b/1e6;
    if (b>0 && a>0) printf "%.2f %.2f %.2fx %.1f%%\n", om, um, a/b, (1-b/a)*100;
    else printf "%.2f %.2f n/a n/a\n", om, um }') || true
  printf '%-10s | %-16s | %-16s | %-8s | %-7s\n' "$base" "$o_ms" "$u_ms" "$speedup" "$pct"
  echo "$base,$o_ms,$u_ms,${speedup%x},${pct%\%}" >> "$CSV_OUT"
  if [[ "$o_ns" == "0" && "$u_ns" == "0" ]]; then
    echo "    (nsys could not extract kernel time; last error: ${LAST_NSYS_ERR:-unknown})" >&2
  fi
done

echo
echo "Saved CSV: $CSV_OUT"
echo "Total GPU kernel time (sum over all CUDA kernels), I/O excluded. Reports saved in $OUTDIR/"
echo "(*.nsys-rep) -- open in the Nsight Systems GUI to see the per-kernel breakdown."
