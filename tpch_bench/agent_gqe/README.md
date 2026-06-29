# GQE-style TPC-DS codegen agent

A local (vLLM) LLM generates a TPC-DS query **plan in GQE's logical IR** — using a name-based DSL —
which is compiled into the GQE engine, run, and checked against a DuckDB CPU reference, with errors
fed back so the model retries. See `PLAN.md` for the full design.

```
 generate build_plan_gen.cpp ─▶ ninja gqe_codegen_query ─▶ run ─▶ diff vs DuckDB reference
        ▲                                                                  │
        └──────────────────── feed back compile / runtime / diff ◀─────────┘
```

The LLM writes columns **by name**; the DSL (`harness/plan_builder.hpp`) resolves names→indices and
tracks the column layout — the hardest part of hand-building a GQE plan.

## Files
```
agent_gqe/
  PLAN.md                       design + decisions
  harness/
    plan_builder.hpp            name-based DSL (scan/filter/join/aggregate/sort/limit + exprs)
    build_plan.hpp              the build_plan() contract the LLM implements
    main.cpp                    fixed: pool + register all TPC-DS tables + optimize/exec + write parquet
    build_plan_gen.cpp          DEFAULT = TPC-DS Q3 in the DSL (agent overwrites at runtime; restored after)
  fewshot/q7_plan.cpp, q43_plan.cpp   worked DSL examples (used in the prompt)
  gen_reference.py              DuckDB CPU reference generator (tpcds extension)
  crosscheck.sh                 Phase 0: existing hardcoded qN vs DuckDB reference
  gqe_codegen.py                the agent loop
  agent_work/                   generated at runtime (reference, output) [gitignored]
```

## Phase 0 — trust the reference (run first)
Confirm the existing GPU hardcoded queries match the DuckDB reference:
```bash
conda activate gqe
./tpch_bench/gqe/build_gqe_v2.sh                       # build gqe (once)
python tpch_bench/common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
./tpch_bench/agent_gqe/crosscheck.sh /data/tpcds_sf1 3 7 43
```
MATCH on q3/q7/q43 means the reference + comparator are trustworthy.

## Inputs (like BespokeOLAP / GenDB)
The agent takes **data files + database schema + a SQL query** — no baseline C++:
- **data files**: `--data <dir>` where each `<dir>/<table>/*.parquet` becomes a table.
- **database schema**: auto-derived from the data (DuckDB views) and saved to
  `agent_work/<label>.input.schema.txt`.
- **SQL query**: one of `--sql "<...>"`, `--sql-file q.sql`, `--query <N>` (TPC-DS), `--tpch <N>`.

From these it generates a **complete GQE program** (a whole `.cpp` with its own `main`, like
`q3.cpp` — built on the name-based DSL + `gqe_runtime.hpp` helpers). The reference is the same SQL
run in DuckDB; the run prints the DuckDB output and the GPU output and diffs them.

The generated whole file lives at `harness/gen_query.cpp` (default = Q3 smoke test; the agent
overwrites it per run and restores the default afterward). On success it is saved as
`solution_<label>.cpp`. To also diff against a precomputed reference dir, pass
`--ref-dir /…/tpcds_ref` (uses `q<N>.parquet`) or `--ref-file <path>`.

## Phase 1 — run the agent
Start with the already-implemented queries (3, 7, 43):
```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000     # in another shell
# by TPC-DS number (convenience):
python tpch_bench/agent_gqe/gqe_codegen.py --query 3 --data /data/tpcds_sf1
# or by arbitrary SQL + your own data:
python tpch_bench/agent_gqe/gqe_codegen.py --data /data/tpcds_sf1 --sql-file myquery.sql --name myq
```
On success the winning plan is saved as `solution_qN.cpp`; the committed Q3 default in
`build_plan_gen.cpp` is restored afterward.

## Smoke test (no LLM)
The default `build_plan_gen.cpp` is Q3, so you can verify the harness end-to-end without the agent:
```bash
./tpch_bench/gqe/build_query.sh gqe_codegen_query      # or rebuild via build_gqe_v2.sh
LD_LIBRARY_PATH=... ./build/benchmark/gqe_codegen_query /data/tpcds_sf1   # writes output.parquet
python tpch_bench/common/compare_parquet.py output.parquet tpcds_ref/q3.parquet
```

## Notes
- Compiles into the GQE build tree (the target is added in `benchmark/CMakeLists.txt`, guarded by
  EXISTS). Uses the vendored-nvcomp `LD_LIBRARY_PATH` handling like the other run scripts.
- Scope today: scan / filter / join / aggregate / sort / limit (covers q3, q7, q43 and many more).
  Harder shapes (set/rollup → q22, intersect → q38, correlated subquery → q6) are Phase 2/3 and may
  need DSL additions (set_relation, in_predicate, window).
