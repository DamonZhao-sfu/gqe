#!/usr/bin/env bash
# Build GQE from source (C++ engine + Rust gqe-cli).
#
# Mirrors the steps in gqe/Dockerfile + CONTRIBUTING.md, but runs them inside
# the `gqe` conda env (tpch_bench/env/gqe-env.yml) instead of a container.
#
# Heavy: it builds libcudf (branch-25.10) from source, then GQE. The MLIR query
# compiler is OFF by default (set GQE_ENABLE_QUERY_COMPILER=ON to also build
# LLVM/MLIR 20.1.2 from source -- adds a long build).
#
# Prereqs:
#   conda activate gqe        # from: mamba env create -f ../env/gqe-env.yml
#   an NVIDIA GPU + driver; network access to clone libcudf
#
# Usage:
#   ./build_gqe.sh
# Env knobs:
#   GQE_SRC            path to gqe repo            (default: repo containing this script)
#   CUDF_SRC           where to clone/build libcudf (default: $HOME/cudf)
#   CUDA_ARCH          CUDA archs                  (default: 80-real;90-real;100-real;120)
#   JOBS               parallel build jobs         (default: nproc)
#   ENABLE_COMPILER    build MLIR query compiler   (default: OFF)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GQE_SRC="${GQE_SRC:-$(cd "$HERE/../.." && pwd)}"
CUDF_SRC="${CUDF_SRC:-$HOME/cudf}"
CUDA_ARCH="${CUDA_ARCH:-80-real;90-real;100-real;120}"
JOBS="${JOBS:-$(nproc)}"
ENABLE_COMPILER="${ENABLE_COMPILER:-OFF}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the gqe conda env first: conda activate gqe" >&2
  exit 1
fi

echo "==> [1/3] Build libcudf from source (branch-25.10) into $CUDF_SRC"
if [[ ! -d "$CUDF_SRC/.git" ]]; then
  git clone --branch branch-25.10 --depth 1 https://github.com/rapidsai/cudf.git "$CUDF_SRC"
fi
pushd "$CUDF_SRC" >/dev/null
# --ptds (per-thread default stream) is required by GQE for H2D/compute overlap.
# NOTE: cudf's build.sh greps for the literal quotes in --cmake-args="...", so the
# whole token is single-quoted here to keep the inner double-quotes intact.
CUDF_CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  ./build.sh libcudf --ptds \
  '--cmake-args="-DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=OFF -DCUDA_ENABLE_LINEINFO=ON"'
popd >/dev/null

echo "==> [2/3] Configure + build GQE (compiler=$ENABLE_COMPILER)"
BUILD_DIR="$GQE_SRC/build"
mkdir -p "$BUILD_DIR"
cmake -G Ninja -S "$GQE_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON \
  -DGQE_ENABLE_QUERY_COMPILER="$ENABLE_COMPILER" \
  -DGQE_ENABLE_RUST_CLI=ON
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "==> [3/3] Build Rust gqe-cli"
pushd "$GQE_SRC/rust" >/dev/null
cargo build --release -p gqe-cli
popd >/dev/null

echo
echo "Build complete:"
echo "  node manager : $BUILD_DIR/src/node_manager/gqe_node_manager"
echo "  task manager : $BUILD_DIR/src/task_manager/gqe_task_manager"
echo "  gqe-cli      : $GQE_SRC/rust/target/release/gqe-cli"
