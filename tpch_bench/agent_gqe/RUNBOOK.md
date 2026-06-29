# RUNBOOK — GQE codegen agent (TPC-H & TPC-DS)

End-to-end guide: build the engine, generate data, compute DuckDB references, then run the agent to
generate a whole GQE program (and a custom-kernel `*_udr.cu`) for a TPC-H or TPC-DS query and verify
it against DuckDB.

Inputs to the agent = **data files + database schema (auto from data) + a SQL query** (no baseline
C++). Output = a complete `.cpp` (its own `main`, like `q3.cpp`) — or a `.cu` custom kernel.

---

## 0. One-time setup

```bash
conda activate gqe
# thrift must be on PATH (gqe configure runs it); install if missing:
which thrift || conda install -y -c conda-forge thrift-compiler=0.19 libthrift=0.19

# Build the engine once (fast path: prebuilt libcudf from conda):
./tpch_bench/gqe/build_gqe_v2.sh
```

vLLM server (use a single GPU, e.g. GPU 1):
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve <model> --port 8000 > vllm.log 2>&1 &
```

---

## 1. Generate data (TPC-H and TPC-DS)

```bash
# TPC-DS  ->  <out>/<table>/*.parquet
python tpch_bench/common/gen_tpcds_data.py --sf 1 --outdir /localhome/hza214/gqe/tpcds_sf1
# TPC-H   ->  <out>/<table>/*.parquet
python tpch_bench/common/gen_tpch_data.py  --sf 1 --outdir /localhome/hza214/gqe/tpch_sf1
```
(Add `--queries` to also dump `queries/qN.sql`.)

---

## 2. Compute DuckDB CPU references (ground truth for comparison)

```bash
python tpch_bench/agent_gqe/gen_reference.py --bench tpcds --data /localhome/hza214/gqe/tpcds_sf1 --query all --outdir /localhome/hza214/gqe/tpcds_ref
python tpch_bench/agent_gqe/gen_reference.py --bench tpch  --data /localhome/hza214/gqe/tpch_sf1  --query all --outdir /localhome/hza214/gqe/tpch_ref
```
Produces `tpcds_ref/q{1..99}.parquet` (+ `.sql`) and `tpch_ref/q{1..22}.parquet`.

Phase-0 sanity (existing hardcoded GPU queries vs DuckDB):
```bash
./tpch_bench/gqe/crosscheck.sh /localhome/hza214/gqe/tpcds_sf1 3 7 43   # (note: crosscheck.sh lives in agent_gqe/)
./tpch_bench/agent_gqe/crosscheck.sh /localhome/hza214/gqe/tpcds_sf1 3 7 43
```

---

## 3. Run the agent — generate the WHOLE GQE program + compare

### TPC-DS (by query number)
```bash
python tpch_bench/agent_gqe/gqe_codegen.py \
  --query 3 --data /localhome/hza214/gqe/tpcds_sf1 \
  --ref-dir /localhome/hza214/gqe/tpcds_ref
```

### TPC-H (by query number)
```bash
python tpch_bench/agent_gqe/gqe_codegen.py \
  --tpch 6 --data /localhome/hza214/gqe/tpch_sf1 \
  --ref-dir /localhome/hza214/gqe/tpch_ref
```

### Any SQL file (TPC-DS data shown; for TPC-H add `--tpch`-style data and `--ref-file`)
```bash
python tpch_bench/agent_gqe/gqe_codegen.py \
  --data /localhome/hza214/gqe/tpcds_sf1 \
  --sql-file tpch_bench/agent_gqe/queries/tpcds_q3.sql --name tpcds_q3 \
  --ref-file /localhome/hza214/gqe/tpcds_ref/q3.parquet
```

What it prints/saves:
- `----- DuckDB CPU output -----` and `----- GPU (gqe) output -----` (first rows of each),
- `[time] DuckDB (CPU) X ms | GPU (gqe) Y ms | speedup Zx` (GPU = engine time from the
  "Query execution time" log; CPU = DuckDB query time),
- `vs DuckDB reference: MATCH/DIFFER` and `vs q<N>.parquet: ...`,
- success → `tpch_bench/agent_gqe/solution_<label>.cpp`,
- every attempt + logs → `tpch_bench/agent_gqe/agent_work/iter*`.

`--tpch` makes the generated `main` call `register_tpch(cat,data)`; otherwise `register_tpcds`.

### Forcing code-only output (reasoning models)
Reasoning/distilled models emit `<think>…</think>` and prose. The agent already (a) strips think
blocks, (b) takes the largest ```cpp block, (c) tolerates a truncated/unfenced reply. Extra knobs:
- `--max-tokens 12000` — raise if a whole-file reply gets truncated (reasoning eats the budget).
- `--guided` — vLLM guided decoding forces the reply to be exactly one ```cpp block (most robust;
  needs a guided-decoding-capable vLLM, the default). This mirrors how BespokeOLAP/GenDB guarantee
  code-only output via structured/tool outputs.

Model choice:
- **Coder models (e.g. Qwen3-Coder)** emit clean fenced code with little/no prose: just run with
  `--temperature 0`; usually no `--guided` needed. Raise `--max-tokens` if a whole-file reply is cut.
- **Reasoning/distilled models** emit `<think>` + prose: prefer `--guided` and a larger
  `--max-tokens` (thinking consumes the budget).

---

## 3b. (Optional) Compare against Sirius — the open-source GPU DuckDB engine

[Sirius](https://github.com/sirius-db/sirius) (Apache-2.0, NVIDIA + UW-Madison) is a GPU execution
engine that plugs into DuckDB; queries run on GPU automatically. Install (needs
[pixi](https://pixi.sh)):
```bash
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius && pixi run make          # builds build/release/duckdb + the sirius extension
```
Then pass `--sirius-home` (or set `SIRIUS_HOME`) and the agent also runs the same SQL on Sirius and
prints its GPU time + correctness:
```bash
python tpch_bench/agent_gqe/gqe_codegen.py --query 4 --data /localhome/hza214/gqe/tpcds_sf1 \
  --ref-dir /localhome/hza214/gqe/tpcds_ref --sirius-home /localhome/hza214/sirius
```
Output adds: `[time] Sirius (GPU): Z ms (vs DuckDB reference: MATCH/DIFFER)` and the per-run summary
becomes `DuckDB (CPU) X | GPU (gqe) Y | Sirius (GPU) Z`.
> Sirius is a strong, optimized GPU baseline — compare your generated GPU code against it, not only
> against DuckDB. At small scale factors all GPU engines (incl. Sirius) can lose to DuckDB.

## 4. Generate a custom fused-kernel program (q3_udr.cu style)

```bash
python tpch_bench/agent_gqe/udr_codegen.py \
  --query 3 --data /localhome/hza214/gqe/tpcds_sf1
# success -> tpch_bench/agent_gqe/solution_q3_udr.cu
```

---

## 5. Compare any two parquet results manually
```bash
python tpch_bench/common/compare_parquet.py \
  tpch_bench/agent_gqe/agent_work/output.parquet \
  /localhome/hza214/gqe/tpcds_ref/q3.parquet
```

---

## Outputs at a glance
| Path | What |
|------|------|
| `tpcds_ref/`, `tpch_ref/` | DuckDB CPU reference results (`qN.parquet`) |
| `agent_gqe/harness/gen_query.cpp` | live generated whole-file (default Q3; restored after each run) |
| `agent_gqe/agent_work/iter<N>_gen.cpp`, `iter<N>_build.log` | each attempt + compiler log |
| `agent_gqe/agent_work/output.parquet` | GPU output of the last successful run |
| `agent_gqe/solution_<label>.cpp` | winning whole-file program |
| `agent_gqe/solution_<label>_udr.cu` | winning custom-kernel program |

---

## Troubleshooting
- **`thrift ... no such file or directory`** during cmake: `conda install -c conda-forge thrift-compiler=0.19`; build with the env active.
- **`undefined symbol: ...nvcomp...LZ4CPUManager`** at runtime: stale conda nvcomp; `conda remove -y --force-remove libnvcomp libnvcomp-dev` (run scripts add gqe's vendored nvcomp to `LD_LIBRARY_PATH`).
- **BUILD FAILED every iteration**: the harness/DSL likely mismatches the real gqe API. First compile the default Q3 smoke test and read the errors:
  ```bash
  git checkout tpch_bench/agent_gqe/harness/gen_query.cpp
  ./tpch_bench/gqe/build_query.sh gqe_codegen_query 2>&1 | tail -80
  ```
- **TPC-H date predicates**: `l_shipdate` etc. are `TIMESTAMP_DAYS`; date literals are the trickiest
  part — expect a few extra agent iterations, or test with date-free queries first.
- **gqe path is TPC-DS/TPC-H** (engine-IR plan). The lighter **libcudf** agent is in `tpch_bench/agent/`.
