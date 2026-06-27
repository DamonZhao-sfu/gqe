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

# Parallelism for ALL build stages:
#  - CMAKE_GENERATOR=Ninja      -> cudf's build.sh and our cmake both use Ninja
#  - CMAKE_BUILD_PARALLEL_LEVEL -> `cmake --build` uses $JOBS everywhere
#  - PARALLEL_LEVEL             -> the knob cudf's build.sh reads
#  - cargo gets -j "$JOBS" below
export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export PARALLEL_LEVEL="$JOBS"
echo "==> building with $JOBS parallel jobs (Ninja)"
# libcudf branch-25.10 must build against nvcomp 5.0.x (its nvcomp_adapter switch
# predates 5.2 enums and cudf compiles with -Werror=switch). See gqe/Dockerfile.
CUDF_NVCOMP_VERSION="${CUDF_NVCOMP_VERSION:-5.0.0.6}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the gqe conda env first: conda activate gqe" >&2
  exit 1
fi

# Conda puts the CUDA toolkit headers/libs under targets/<arch>-linux/ rather than the env root.
# Current libcudf (branch-25.10) has a runtime-compilation component (librtcx) that needs
# nvJitLink.h, so make those dirs visible to the compiler/linker.
_arch_triplet="$(uname -m)-linux"
if [[ -d "$CONDA_PREFIX/targets/$_arch_triplet/include" ]]; then
  export CPATH="$CONDA_PREFIX/targets/$_arch_triplet/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CONDA_PREFIX/targets/$_arch_triplet/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"

echo "==> [1/3] Build libcudf from source (branch-25.10) into $CUDF_SRC"
if [[ ! -d "$CUDF_SRC/.git" ]]; then
  git clone --depth 1 https://github.com/rapidsai/cudf.git "$CUDF_SRC"
fi

# Ensure nvJitLink dev headers (nvJitLink.h) are present for libcudf's librtcx component.
# (Kept after the build -- only nvcomp is removed below.)
echo "    ensuring libnvjitlink-dev is installed"
conda install -y -c rapidsai -c conda-forge -c nvidia libnvjitlink-dev

# Install nvcomp 5.0.x ONLY for the libcudf build, then remove it so it can't
# shadow the nvcomp 5.2 that GQE fetches itself (cmake/nvcomp.cmake). Mirrors
# gqe/Dockerfile. (Note: current cudf actually fetches its own proprietary nvcomp,
# so this is mostly belt-and-suspenders.) If a newer nvcomp is present it is downgraded.
echo "    installing nvcomp $CUDF_NVCOMP_VERSION for the libcudf build"
# Pass channels explicitly: `conda install` uses the global channel config
# (often just `defaults`), not the channels listed in the env's yml.
conda install -y -c rapidsai -c conda-forge -c nvidia \
  "libnvcomp-dev=$CUDF_NVCOMP_VERSION"

pushd "$CUDF_SRC" >/dev/null
# --ptds (per-thread default stream) is required by GQE for H2D/compute overlap.
# NOTE: cudf's build.sh greps for the literal quotes in --cmake-args="...", so the
# whole token is single-quoted here to keep the inner double-quotes intact.
PARALLEL_LEVEL="$JOBS" CUDF_CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  ./build.sh libcudf --ptds \
  '--cmake-args="-DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=OFF -DCUDA_ENABLE_LINEINFO=ON"'
popd >/dev/null

echo "    removing the temporary nvcomp (GQE will fetch its own 5.2)"
conda remove -y libnvcomp-dev libnvcomp || true

echo "==> [2/3] Configure + build GQE (compiler=$ENABLE_COMPILER)"
BUILD_DIR="$GQE_SRC/build"
mkdir -p "$BUILD_DIR"
# Auto-use a compiler cache (speeds up repeated full builds), and let nvcc multi-thread its
# per-arch passes. Tip: for a single-GPU box, CUDA_ARCH=native builds ~Nx faster than multi-arch.
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
  -DCMAKE_CUDA_FLAGS="--threads ${NVCC_THREADS:-0}" \
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
echo "Build complete:"
echo "  node manager : $BUILD_DIR/src/node_manager/gqe_node_manager"
echo "  task manager : $BUILD_DIR/src/task_manager/gqe_task_manager"
echo "  gqe-cli      : $GQE_SRC/rust/target/release/gqe-cli"
