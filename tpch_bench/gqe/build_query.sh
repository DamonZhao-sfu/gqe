#!/usr/bin/env bash
# Compile ONE (or a few) benchmark target(s) incrementally -- e.g. q7_udr, q3_udr, q3.
#
# The benchmark programs are normal CMake targets under benchmark/ that link the gqe library.
# This script configures the gqe build dir once (if needed), then builds only the requested
# target(s) with Ninja -- so after the first full build_gqe.sh, recompiling a single query is
# a matter of seconds.
#
# Prereqs:
#   conda activate gqe
#   ./build_gqe.sh        # at least once, so libcudf + the gqe library exist
#
# Usage:
#   ./build_query.sh q7_udr
#   ./build_query.sh q3 q3_udr q7_udr
#   ./build_query.sh --list          # show available targets
#
# Env knobs:
#   GQE_SRC          path to gqe repo            (default: repo containing this script)
#   CUDA_ARCH        CUDA archs                  (default: native)
#   JOBS             parallel jobs               (default: nproc)
#   ENABLE_COMPILER  build MLIR query compiler   (default: OFF) -- only used on first configure
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
CUDA_ARCH="${CUDA_ARCH:-native}"
JOBS="${JOBS:-$(nproc)}"
ENABLE_COMPILER="${ENABLE_COMPILER:-OFF}"
BUILD_DIR="$GQE_SRC/build"

export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

# Known benchmark targets (from benchmark/CMakeLists.txt). Any name is accepted; this is just
# for --list and a friendly hint.
KNOWN_TARGETS=(tpc q3 q3_udr q6 q7 q7_udr q22 q38 q43 q48 load_tpcds)

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
  echo "Available benchmark targets:"; printf '  %s\n' "${KNOWN_TARGETS[@]}"
  exit 0
fi

TARGETS=("$@")
[[ ${#TARGETS[@]} -gt 0 ]] || {
  echo "usage: build_query.sh <target> [target ...]   (try --list)" >&2; exit 1; }

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the gqe conda env first: conda activate gqe" >&2
  exit 1
fi

# Configure the build dir once. We detect a prior configure by build.ninja; if libcudf has not
# been built yet, cmake configure will fail -- run build_gqe.sh first in that case.
if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  echo "==> configuring gqe build dir (first time): $BUILD_DIR"
  cmake -G Ninja -S "$GQE_SRC" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON \
    -DGQE_ENABLE_QUERY_COMPILER="$ENABLE_COMPILER" \
    -DGQE_ENABLE_RUST_CLI=OFF
else
  echo "==> using existing build dir: $BUILD_DIR"
fi

# If a new .cu/.cpp was added to benchmark/CMakeLists.txt since the last configure, Ninja will
# re-run CMake automatically when it sees the changed CMakeLists.

echo "==> building target(s): ${TARGETS[*]}  (-j $JOBS)"
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j "$JOBS"

echo
echo "Built:"
for t in "${TARGETS[@]}"; do
  bin="$BUILD_DIR/benchmark/$t"
  if [[ -x "$bin" ]]; then
    echo "  $bin"
  else
    echo "  (target '$t' built, but no binary at $bin -- check the target name)"
  fi
done
echo
echo "Run e.g.:  $BUILD_DIR/benchmark/${TARGETS[0]} /data/tpcds_sf1"
