# TPC-H bring-up: GQE vs cudf (for an Agent-codegen-GPU project)

Scripts to build & run TPC-H on **GQE** and on **cudf's C++ TPC-H programs**,
plus conda envs and a DuckDB-based data generator. Intended as the evaluation
harness for an "Agent codegens GPU code per query" project.

## Which base to build the codegen Agent on?

| If the Agent generates...                              | Base on                | Why |
|--------------------------------------------------------|------------------------|-----|
| **libcudf operator C++** (one `.cpp` per query)        | **cudf `cpp/examples/tpch`** | standalone exe per query, secs to compile, easy to validate, no server |
| **hand-written fused CUDA kernels**                    | **GQE `user_defined_relation`** | engine hosts the kernel; `benchmark/hardcoded/q3_udr.cu` is the template; provides cuco / libperfect hash, data mgmt, multi-GPU |

Recommendation: prototype the codegen loop on **cudf examples** (fast, clean
compile→run→validate cycle), move to **GQE UDR** only when you need bespoke
fused kernels. GQE's build (libcudf-from-source + optional MLIR/LLVM + NVSHMEM
+ gRPC + Arrow Flight + a running server) is heavy for a high-throughput
compile-test loop. Neither ships TPC-DS; cudf's op-composition approach
generalizes to it more cheaply than writing 99 custom kernels.

## Layout

```
tpch_bench/
  env/gqe-env.yml           conda env to build/run GQE
  env/cudf-env.yml          conda env to build/run cudf TPC-H programs
  common/gen_tpch_data.py   DuckDB -> TPC-H Parquet + schema.sql + q*.sql
  common/gen_tpcds_data.py  DuckDB -> TPC-DS Parquet + q*.sql
  gqe/build_gqe.sh          build GQE (C++ engine + Rust gqe-cli)
  gqe/run_gqe_tpch.sh       TPC-H via the SERVER path (start server, load, query)
  gqe/run_gqe_benchmark.sh  TPC-DS via the standalone hardcoded benchmark programs
  cudf/build_cudf_ndsh.sh   build libcudf + tpch examples / ndsh benchmarks
  cudf/run_cudf_ndsh.sh     run cudf tpch_qN program(s)
```

## 1. Generate data (shared)

```bash
conda activate gqe          # duckdb is in this env (or pip install duckdb)
python tpch_bench/common/gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1 --queries
```

Produces `<outdir>/<table>/<table>.parquet`, `schema.sql`, and `queries/q*.sql`.

## 2. GQE

```bash
mamba env create -f tpch_bench/env/gqe-env.yml && conda activate gqe
./tpch_bench/gqe/build_gqe.sh                       # builds libcudf + GQE + gqe-cli
```

GQE has two ways to run, on two different benchmarks:

**(a) TPC-H, via the Flight SQL server path** (`scripts/load_tpch.py` + `run_tpch.py`):
```bash
./tpch_bench/gqe/run_gqe_tpch.sh /data/tpch_sf1 1   # or: 1 6 14  | all
```

**(b) TPC-DS, via the standalone hardcoded benchmark programs** (`benchmark/hardcoded/*`).
These are ALL TPC-DS queries (q3, q6, q7, q22, q38, q43, q48, and q3_udr -- the
custom-kernel variant of q3). They run in-process, no server:
```bash
python tpch_bench/common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
./tpch_bench/gqe/run_gqe_benchmark.sh /data/tpcds_sf1 3       # or: 3 6 22 | udr | all
```
> Heads-up: the query NUMBERS here (3/6/7/22/38/43/48) are **TPC-DS** query
> numbers, not TPC-H.

## 3. cudf

```bash
mamba env create -f tpch_bench/env/cudf-env.yml && conda activate cudf-bench
./tpch_bench/cudf/build_cudf_ndsh.sh                  # BUILD_TARGET=examples|benchmarks|both
./tpch_bench/cudf/run_cudf_ndsh.sh /data/tpch_sf1 1   # examples implement q1,5,6,9,10
```

## Caveats (verify against your checkout)

- **GQE deps are pinned & heavy.** Versions come from `gqe/conda/docker-x86_64.yml`
  and `gqe/Dockerfile` (libcudf `branch-25.10` built with `--ptds`; MLIR/LLVM
  `20.1.2` only if `GQE_ENABLE_QUERY_COMPILER=ON`). The container path
  (`docker build -t gqe .`) is the most reproducible; these scripts replicate it
  in a conda env.
- **DuckDB-dialect query SQL** may need minor edits for GQE's DataFusion/Substrait
  front-end. `run_tpch.py --validate <ref>` checks correctness.
- **cudf example CLI args vary by version.** `run_cudf_ndsh.sh` tries common forms;
  if a binary differs, run `tpch_qN --help` and adjust. ndsh nvbench binaries take
  `--scale-factor` instead of a dataset dir.
