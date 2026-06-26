#!/usr/bin/env bash
# Build libcudf + its TPC-H/NDS-H C++ programs from source.
#
# cudf ships TWO relevant sets of standalone TPC-H programs:
#   1. cpp/examples/tpch/        -> q1,q5,q6,q9,q10 ; each takes a dataset dir,
#                                   reads Parquet, prints/writes results.
#                                   Cleanest "one file per query" harness ->
#                                   best base for an Agent-codegen project.
#   2. cpp/benchmarks/ndsh/      -> NDS-H nvbench microbenchmarks (q1,q5,q6,q9,q10).
#
# This script builds the examples by default (BUILD_TARGET=examples) and can
# also build the benchmarks (BUILD_TARGET=benchmarks).
#
# Prereqs:
#   conda activate cudf-bench   # from: mamba env create -f ../env/cudf-env.yml
#   NVIDIA GPU + driver; network to clone cudf
#
# Usage:
#   ./build_cudf_ndsh.sh
# Env knobs:
#   CUDF_SRC       where to clone cudf        (default: $HOME/cudf-bench)
#   CUDF_BRANCH    cudf branch/tag            (default: branch-25.10)
#   CUDA_ARCH      CUDA archs                 (default: native)
#   JOBS           parallel jobs             (default: nproc)
#   BUILD_TARGET   examples | benchmarks | both (default: examples)
set -euo pipefail

CUDF_SRC="${CUDF_SRC:-$HOME/cudf-bench}"
CUDF_BRANCH="${CUDF_BRANCH:-branch-25.10}"
CUDA_ARCH="${CUDA_ARCH:-native}"
JOBS="${JOBS:-$(nproc)}"
BUILD_TARGET="${BUILD_TARGET:-examples}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the cudf-bench conda env first." >&2
  exit 1
fi

echo "==> [1/3] Clone cudf ($CUDF_BRANCH) into $CUDF_SRC"
if [[ ! -d "$CUDF_SRC/.git" ]]; then
  git clone --branch "$CUDF_BRANCH" --depth 1 https://github.com/rapidsai/cudf.git "$CUDF_SRC"
fi

# Prefer cudf's own, version-matched conda env file if you haven't made one.
BUNDLED_ENV=$(ls "$CUDF_SRC"/conda/environments/all_cuda-*_arch-"$(uname -m)".yaml 2>/dev/null | head -n1 || true)
if [[ -n "$BUNDLED_ENV" ]]; then
  echo "    (note) cudf's authoritative build env is: $BUNDLED_ENV"
fi

echo "==> [2/3] Build libcudf"
pushd "$CUDF_SRC" >/dev/null
CUDF_CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" ./build.sh libcudf -j"$JOBS"

echo "==> [3/3] Build TPC-H program(s): $BUILD_TARGET"
build_examples() {
  pushd "$CUDF_SRC/cpp/examples" >/dev/null
  ./build.sh   # builds all examples, including tpch/
  popd >/dev/null
}
build_benchmarks() {
  CUDF_CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" ./build.sh benchmarks -j"$JOBS"
}
case "$BUILD_TARGET" in
  examples)   build_examples ;;
  benchmarks) build_benchmarks ;;
  both)       build_examples; build_benchmarks ;;
  *) echo "ERROR: BUILD_TARGET must be examples|benchmarks|both" >&2; exit 1 ;;
esac
popd >/dev/null

echo
echo "Build complete. Likely artifact locations (verify on your tree):"
echo "  examples   : $CUDF_SRC/cpp/examples/tpch/build/tpch_q{1,5,6,9,10}"
echo "  benchmarks : $CUDF_SRC/cpp/build/benchmarks/  (NDSH_* nvbench binaries)"
