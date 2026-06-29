# Example query files (agent input)

SQL query files used as input to the GQE codegen agent. The agent reads a `.sql` file (plus the
data files, from which it derives the schema) and generates GQE-template plan code.

- `tpcds_q3.sql`, `tpcds_q7.sql` — TPC-DS queries (the GQE harness registers TPC-DS tables).

## Get more query files (dump all from DuckDB)
```bash
# TPC-DS q1..q99 -> /data/tpcds_sf1/queries/qN.sql  (also generates the data)
python tpch_bench/common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1 --queries
# TPC-H  q1..q22 -> /data/tpch_sf1/queries/qN.sql
python tpch_bench/common/gen_tpch_data.py  --sf 1 --outdir /data/tpch_sf1  --queries
```

## Generate code from a query file
```bash
python tpch_bench/agent_gqe/gqe_codegen.py \
  --data /data/tpcds_sf1 \
  --sql-file tpch_bench/agent_gqe/queries/tpcds_q3.sql \
  --name tpcds_q3
# output: tpch_bench/agent_gqe/solution_tpcds_q3.cpp (on success)
```

Note: the GQE path generates a `build_plan` (logical-plan C++ compiled into the CUDA engine), not a
raw `.cu` kernel. Raw fused CUDA kernels are the UDR escalation (see PLAN_v2, Phase 3). TPC-H queries
should use the libcudf agent (`tpch_bench/agent/`), which loads TPC-H tables.
