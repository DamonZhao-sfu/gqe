# GPU-codegen Agent (prototype)

A minimal agent where a **local vLLM-served LLM writes a libcudf TPC-H query**, which is then
compiled, run, and verified against a DuckDB reference — with errors fed back so the model retries.

```
 generate run_query.cpp ──▶ cmake build ──▶ run on dataset ──▶ diff vs DuckDB reference
        ▲                                                                  │
        └───────────────── feed back compile / runtime / diff error ◀──────┘
```

The baseline is "express the query as **libcudf operator calls**" (cudf NDS-H style). A fixed C++
harness supplies Parquet I/O + `main` + table loading, so the LLM only fills in one function —
keeping compiles reliable and the action space small.

## Files
```
agent/
  agent_codegen.py        the agent loop (stdlib-only LLM call; DuckDB reference; cmake build)
  harness/
    run_query.hpp         fixed: Table struct + the run_query() signature the LLM implements
    main.cpp              fixed: loads TPC-H parquet tables, calls run_query, writes output.parquet
    CMakeLists.txt        fixed: find_package(cudf) + build
  agent_work/             generated each run (run_query.cpp, build/, reference.parquet) [gitignored]
```

## Prereqs
1. **vLLM server** running an instruct/code model:
   ```bash
   pip install vllm
   vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000
   ```
2. **libcudf installed** in the conda env (the agent compiles against it):
   ```bash
   ./tpch_bench/gqe/build_gqe_v2.sh        # installs prebuilt libcudf, or just: conda install -c rapidsai libcudf=25.10
   ```
3. **TPC-H parquet data** + duckdb (in the gqe env):
   ```bash
   python tpch_bench/common/gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1
   ```

## Run
```bash
conda activate gqe
# Q6 (single-table filter+sum) and Q1 (single-table groupby) are the easiest to start with.
python tpch_bench/agent/agent_codegen.py --query 6 --data /data/tpch_sf1
python tpch_bench/agent/agent_codegen.py --query 1 --data /data/tpch_sf1 --max-iters 8

# point at a different server / model:
python tpch_bench/agent/agent_codegen.py --query 6 --data /data/tpch_sf1 \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-7B-Instruct
```
On success it prints the winning `agent_work/src/run_query.cpp` and writes `agent_work/output.parquet`.

## How it works (loop)
1. Fetch the query SQL from DuckDB's `tpch_queries()` and compute the **reference** result.
2. Prompt the LLM with the SQL + table schemas + the `run_query()` contract.
3. Write the returned code to `run_query.cpp`, `cmake --build`.
4. If it builds, run the binary → `output.parquet`.
5. Compare to the reference (`common/compare_parquet.py`, order-independent, numeric tolerance).
6. On any failure, append the **error** to the chat and loop (up to `--max-iters`).

## Notes / next steps
- Start with single-table queries (Q1, Q6); multi-table (Q3, joins) need a more capable model and
  more iterations. The harness already exposes all tables, so no harness change is needed.
- Reward signal for an RL/search loop is right here: `compile_ok`, `run_ok`, `result_match`, and you
  can add a timing axis (run the binary a few times, or wire in `nsys`).
- This uses pure cudf (its own conda nvcomp), so none of the gqe vendored-nvcomp runtime issues apply.
- To target **custom CUDA kernels** instead of libcudf composition, swap the harness/contract for the
  GQE `user_defined_relation` template (`benchmark/hardcoded/q3_udr.cu`) — heavier, but that's the
  path for agent-generated fused kernels.
