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
# Env knobs (configure-time ones apply on first configure or with RECONFIGURE=1):
#   GQE_SRC          path to gqe repo            (default: repo containing this script)
#   BUILD_DIR        build directory             (default: $GQE_SRC/build)
#   CUDA_ARCH        CUDA archs                  (default: native -- only THIS GPU = fastest)
#   JOBS             parallel jobs               (default: nproc)
#   NVCC_THREADS     nvcc --threads             (default: 0 = all cores)
#   FAST             1 => -O0 (fast compile, slow runtime; good for correctness iteration)
#   RECONFIGURE      1 => re-run cmake to apply new flags to an existing build dir
#   ENABLE_COMPILER  build MLIR query compiler   (default: OFF)
#
# Auto-uses sccache/ccache as a compiler cache if either is on PATH.
#
# Fastest single-file iteration: build everything once for your GPU only, then edit + rebuild:
#   CUDA_ARCH=native ./build_gqe.sh           # one-time, native arch
#   ./build_query.sh q7_udr                    # seconds per edit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
CUDA_ARCH="${CUDA_ARCH:-native}"     # native = compile only for THIS GPU (fastest single-file)
JOBS="${JOBS:-$(nproc)}"
ENABLE_COMPILER="${ENABLE_COMPILER:-OFF}"
NVCC_THREADS="${NVCC_THREADS:-0}"    # 0 = let nvcc parallelize its compilation passes over all cores
FAST="${FAST:-0}"                    # 1 = -O0 device/host: much faster compile, slower runtime
RECONFIGURE="${RECONFIGURE:-0}"      # 1 = re-run cmake even if already configured (apply new flags)
BUILD_DIR="${BUILD_DIR:-$GQE_SRC/build}"

export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

# Auto-detect a compiler cache so unchanged translation units are not recompiled across rebuilds.
CACHE_ARGS=()
if command -v sccache >/dev/null 2>&1; then
  CACHE_ARGS=(-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache)
  echo "==> compiler cache: sccache"
elif command -v ccache >/dev/null 2>&1; then
  CACHE_ARGS=(-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  echo "==> compiler cache: ccache"
fi

# Extra compile flags: nvcc multi-threading, and optional -O0 fast-iteration mode.
CUDA_FLAGS="--threads ${NVCC_THREADS}"
CXX_FLAGS=""
if [[ "$FAST" == "1" ]]; then
  CUDA_FLAGS="$CUDA_FLAGS -O0"
  CXX_FLAGS="-O0"
  echo "==> FAST mode: -O0 (fast compile, slower runtime)"
fi

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
# Flags below (arch, cache, --threads, -O0) only take effect at CONFIGURE time. If the dir was
# already configured (e.g. by build_gqe.sh with multi-arch), pass RECONFIGURE=1 to re-apply them.
if [[ ! -f "$BUILD_DIR/build.ninja" || "$RECONFIGURE" == "1" ]]; then
  echo "==> configuring gqe build dir: $BUILD_DIR (arch=$CUDA_ARCH, nvcc --threads=$NVCC_THREADS)"
  cmake -G Ninja -S "$GQE_SRC" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON \
    -DGQE_ENABLE_QUERY_COMPILER="$ENABLE_COMPILER" \
    -DGQE_ENABLE_RUST_CLI=OFF \
    "${CACHE_ARGS[@]}"
else
  echo "==> using existing build dir: $BUILD_DIR"
  echo "    (it keeps its original arch/flags; pass RECONFIGURE=1 to apply native/cache/-O0)"
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
