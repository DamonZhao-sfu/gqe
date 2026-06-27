#!/usr/bin/env bash
# build_gqe.sh  v2  --  FAST build using the PREBUILT libcudf from conda.
#
# Why this is much faster than v1 (build_gqe.sh):
#   gqe's CMake does `rapids_find_package(cudf CONFIG REQUIRED)` -- it LINKS an installed libcudf,
#   it does not compile cudf itself. v1 compiles all of libcudf from source first (~30-60 min, the
#   dominant cost, multiplied by the number of CUDA archs). v2 installs the prebuilt RAPIDS libcudf
#   conda package instead, so only the gqe library + benchmarks + Rust CLI are compiled (minutes).
#
# Trade-off: the conda libcudf is the standard build (without the per-thread-default-stream "--ptds"
# tuning the upstream Dockerfile uses). gqe has no hard PTDS dependency, so this is functionally
# fine for the benchmarks / UDR comparison; absolute throughput may differ slightly from a --ptds
# libcudf. Use v1 if you need the exact upstream performance configuration.
#
# Prereqs:
#   conda activate gqe        # from: mamba env create -f ../env/gqe-env.yml
#
# Usage:
#   ./build_gqe_v2.sh
# Env knobs:
#   GQE_SRC          path to gqe repo            (default: repo containing this script)
#   CUDA_ARCH        CUDA archs                  (default: native -- only THIS GPU = fastest)
#   JOBS             parallel jobs               (default: nproc)
#   NVCC_THREADS     nvcc --threads             (default: 0 = all cores)
#   RAPIDS_VERSION   libcudf/librmm version     (default: 25.10 -- matches gqe's rapids-cmake pin)
#   ENABLE_COMPILER  build MLIR query compiler   (default: OFF)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
CUDA_ARCH="${CUDA_ARCH:-native}"
JOBS="${JOBS:-$(nproc)}"
NVCC_THREADS="${NVCC_THREADS:-0}"
RAPIDS_VERSION="${RAPIDS_VERSION:-25.10}"
ENABLE_COMPILER="${ENABLE_COMPILER:-OFF}"
BUILD_DIR="${BUILD_DIR:-$GQE_SRC/build}"

export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
echo "==> v2 fast build with $JOBS jobs (Ninja), arch=$CUDA_ARCH, libcudf=$RAPIDS_VERSION (prebuilt)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the gqe conda env first: conda activate gqe" >&2
  exit 1
fi

# Make conda's CUDA headers/libs (under targets/<arch>-linux/) visible to the compiler/linker.
_arch_triplet="$(uname -m)-linux"
if [[ -d "$CONDA_PREFIX/targets/$_arch_triplet/include" ]]; then
  export CPATH="$CONDA_PREFIX/targets/$_arch_triplet/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CONDA_PREFIX/targets/$_arch_triplet/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"

echo "==> [1/3] Installing prebuilt libcudf $RAPIDS_VERSION (+ deps) from conda"
# `libcudf` ships the headers + cudf-config.cmake (for find_package) + cudftestutil (the 'testing'
# component gqe requires). It pulls librmm/libkvikio/nvcomp as dependencies. libnvjitlink-dev /
# libcufile-dev are kept for any source we still compile. nvcomp for cudf 25.10 is 5.2.x, which is
# exactly what gqe wants, so there is no nvcomp version conflict.
conda install -y -c rapidsai -c conda-forge -c nvidia \
  "libcudf=${RAPIDS_VERSION}" "librmm=${RAPIDS_VERSION}" "libkvikio=${RAPIDS_VERSION}" \
  "cuda-version=12.9" libnvjitlink-dev libcufile-dev

# gqe vendors its own nvcomp 5.2 (cmake/nvcomp.cmake) with an extended manager API. Any conda
# nvcomp *dev headers* under $CONDA_PREFIX/include/nvcomp clash with it (multiple definition /
# wrong API: missing nvcompBitshuffleMode_t, different decompress signature). Remove just the
# headers package -- the conda nvcomp runtime lib that libcudf needs stays installed.
echo "    removing conda nvcomp dev headers (gqe uses its own vendored nvcomp 5.2)"
conda remove -y --force-remove libnvcomp-dev 2>/dev/null || true
if [[ -e "$CONDA_PREFIX/include/nvcomp/nvcompManager.hpp" ]]; then
  echo "    WARNING: conda nvcomp headers still present at $CONDA_PREFIX/include/nvcomp;" >&2
  echo "             the gqe build may clash. Identify the owning pkg with:" >&2
  echo "               conda list | grep -i nvcomp" >&2
fi

echo "==> [2/3] Configure + build the gqe library/benchmarks (compiler=$ENABLE_COMPILER)"
GQE_CACHE_ARGS=()
if command -v sccache >/dev/null 2>&1; then
  GQE_CACHE_ARGS=(-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache)
  echo "    compiler cache: sccache"
elif command -v ccache >/dev/null 2>&1; then
  GQE_CACHE_ARGS=(-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  echo "    compiler cache: ccache"
fi
cmake -G Ninja -S "$GQE_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_FLAGS="--threads ${NVCC_THREADS}" \
  -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON \
  -DGQE_ENABLE_QUERY_COMPILER="$ENABLE_COMPILER" \
  -DGQE_ENABLE_RUST_CLI=ON \
  "${GQE_CACHE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "==> [3/3] Build Rust gqe-cli"
pushd "$GQE_SRC/rust" >/dev/null
cargo build --release -p gqe-cli -j "$JOBS"
popd >/dev/null

echo
echo "Build complete (v2, prebuilt libcudf):"
echo "  benchmarks   : $BUILD_DIR/benchmark/   (q3, q3_udr, q7, q7_udr, q43, q43_udr, ...)"
echo "  node manager : $BUILD_DIR/src/node_manager/gqe_node_manager"
echo "  gqe-cli      : $GQE_SRC/rust/target/release/gqe-cli"
