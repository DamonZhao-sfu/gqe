# Plan: Agent that generates GQE-hardcoded-style GPU code for TPC-DS

Status: **PLAN ONLY — no implementation yet.** This documents the approach, the exact API the
generator must emit, and the validation strategy. Implementation happens after review.

## 1. Goal

An agent where a (local, vLLM-served) LLM generates **GQE-hardcoded-style C++** for a TPC-DS query
— i.e. it hand-builds a `gqe::logical::*` plan exactly like `benchmark/hardcoded/q3.cpp` — which is
then compiled into the GQE engine, run on TPC-DS parquet, and **verified against a DuckDB CPU
reference** (and, for the 7 already-implemented queries, cross-checked against the existing GPU
hardcoded output).

This is the *engine-IR* codegen path (contrast: the libcudf-composition path in `tpch_bench/agent/`,
which is lighter and better for a first prototype). The GQE path is the route toward later
agent-generated fused kernels (`user_defined_relation`, cf. `q3_udr.cu`).

## 2. What the generator must emit (target grammar)

### 2.1 Logical relations (`include/gqe/logical/*.hpp`)
| Relation | Constructor (abridged) |
|----------|------------------------|
| `read_relation` | `(subqueries, column_names, column_types, table_name, partial_filter)` |
| `filter_relation` | `(input, subqueries, condition, projection_indices)` |
| `project_relation` | `(input, subqueries, output_expressions)` |
| `join_relation` | `(left, right, subqueries, condition, join_type, projection_indices)` |
| `aggregate_relation` | `(input, subqueries, keys, measures=vector<pair<agg::Kind, expr>>)` |
| `sort_relation` | `(input, subqueries, orders, null_precedences, expressions)` |
| `fetch_relation` | `(input, offset, count)` |
| `set_relation` | `(lhs, rhs, set_operator_type)`  // union / union_all / intersect / minus |
| `window_relation` | `(input, subqueries, aggr, args, order_by, partition_by, dirs, lo, hi)` |
| `user_defined_relation` | `(children, task_functor, data_types, last_child_break_pipeline)` |

`join_type_type`: `inner, left, left_semi, left_anti, full, single`.

### 2.2 Expressions (`include/gqe/expression/*.hpp`)
- `column_reference_expression(idx)` — **0-based index into the CURRENT relation's columns**.
- `literal_expression<T>(value, is_null=false)` — `int32/int64/double/std::string/timestamp_D`.
- comparisons: `equal/not_equal/less/greater/less_equal/greater_equal/nulls_equal`.
- logical: `logical_and/logical_or`, `not_expression`.
- arithmetic: `add/subtract/multiply/divide`.
- `if_then_else_expression(if, then, else)` — CASE WHEN.
- `in_predicate_expression(needles, haystack_relation_index)` — IN / pushdown.
- `cast_expression(in, type)`, `is_null_expression(in)`.
- scalar fns: `datepart_expression`, `like_expression`, `substr_expression`, `round_expression`.

### 2.3 Aggregation kinds (`src/executor/aggregate.cpp`)
`SUM, PRODUCT, MIN, MAX, MEAN, COUNT_VALID, COUNT_ALL, VARIANCE, STD, MEDIAN, SUM_OF_SQUARES, ANY, ALL`.

### 2.4 Fixed main() skeleton (from q3/q7/q43.cpp)
memory pool (`get_memory_pool_size` + RMM pool) → `task_manager_context` + `catalog` + register
tables → build logical plan → `logical_optimizer` (rules: `uniqueness_propagation`,
`join_unique_keys`) → `physical_plan_builder` → `query_context` + `task_graph_builder` →
`execute_task_graph_single_gpu` → write `output.parquet` (+ `bandwidth.json`).

### 2.5 Schema source
`gqe::utility::tpcds::table_definitions()` (`src/utility/tpcds.cpp`) enumerates all TPC-DS tables
with column names+types and unique keys. Types: identifier/integer = INT64, decimal = FLOAT64,
string = STRING, date = TIMESTAMP_DAYS.

## 3. Key design choice: shrink the action space ("fill build_plan")

Mirror the libcudf agent's "fill one function" idea. The harness provides ALL boilerplate; the LLM
emits ONLY:

```cpp
std::shared_ptr<gqe::logical::relation>
build_plan(const gqe::catalog& cat,
           const std::function</*read_table helper*/>& read_table);
```

The harness:
- registers **every** TPC-DS table with its full schema (from `tpcds::table_definitions()`), so the
  LLM never writes registration boilerplate and can read any table/column by name;
- provides the `read_table(name, {cols}, [haystack], [partial_filter])` helper (as in q3.cpp);
- runs optimize → physical → task graph → execute → write output.parquet.

This removes ~80% of the code (and of the compile errors) and lets the LLM focus on the plan.

## 4. The hard part (call it out)

`column_reference_expression` is **by index**, and every `filter/join/project` reorders columns via
`projection_indices`. The LLM must track the column layout after each operator (the existing
hardcoded files do this with `// After this operation, columns are [...]` comments). This is the
main source of *logic* errors (not compile errors). Mitigations:
1. Few-shot with q3/q7/q43 (they show the index-tracking discipline).
2. Require the LLM to emit the running column-layout comment after each relation.
3. Future: add a thin **name-based** column helper layer so the LLM references columns by name, and
   a small pass resolves names→indices. (Optional; reduces the hardest error class.)

## 5. Validation / reference strategy

| Layer | Source | Role |
|-------|--------|------|
| **CPU ground truth** | DuckDB `tpcds` ext: `tpcds_queries()` SQL over the same parquet | authoritative reference |
| **GPU known-good** | existing GQE hardcoded q3,6,7,22,38,43,48 | cross-check harness + first targets |
| **Comparator** | `test/end_to_end/verify_parquet.py` (atol 1e-6) or `tpch_bench/common/compare_parquet.py` | diff output vs reference |

- Reference generation: register `<data>/<table>/*.parquet` as DuckDB views, run the TPC-DS SQL,
  `COPY (...) TO reference.parquet`. (`gen_tpcds_data.py` already uses the tpcds extension.)
- Compare positionally after sorting (gqe output columns are positional / default-named; DuckDB
  reference is in SELECT order). Numeric tolerance for decimal/float; date as days.
- **Phase 0 sanity**: confirm existing hardcoded q3/q7/q43 output ≈ DuckDB reference. This validates
  the reference pipeline on known-good GPU code before trusting it on agent output.

### Caveats
- DuckDB-dialect SQL is the spec; ensure the LLM targets that exact SQL.
- Some TPC-DS queries exceed the tractable subset (correlated subqueries → q6; INTERSECT → q38;
  rollup → q22). Scope the first targets to **filter/join/aggregate/sort** shapes.
- gqe runtime needs its vendored nvcomp on `LD_LIBRARY_PATH` (already handled by our run scripts).

## 6. Harness & loop (to build later)

Proposed layout `tpch_bench/agent_gqe/`:
```
PLAN.md                      (this file)
harness/
  build_plan.hpp             signature the LLM implements
  main.cpp                   pool + register-all-tpcds + read_table + optimize/exec + write parquet
  CMakeLists or target hook  add_executable linking `gqe` (built in the gqe build tree)
gqe_codegen.py               the agent loop (vLLM -> emit build_plan.cpp -> ninja -> run -> diff)
reference.py                 DuckDB tpcds reference generator (reuse gen_tpcds_data wiring)
fewshot/                     q3.cpp, q7.cpp, q43.cpp as exemplars
```
Loop = generate `build_plan.cpp` → incremental `ninja` in the gqe build tree (fast after first
build) → run binary (writes output.parquet) → `verify_parquet.py` vs DuckDB reference → feed
back compile/runtime/diff errors → retry (≤ max-iters). Reuse `build_query.sh` for incremental
compile and the nvcomp `LD_LIBRARY_PATH` handling from the run scripts.

## 7. Phased execution

- **Phase 0** — reference harness: DuckDB tpcds reference + compare; validate on existing q3/q7/q43.
- **Phase 1** — codegen harness ("fill build_plan") + loop; target the 7 already-implemented queries
  first (achievable in gqe; q3/q7/q43 simplest). Success metric: match DuckDB reference.
- **Phase 2** — new TPC-DS queries of filter/join/aggregate/sort shape.
- **Phase 3** — harder shapes (set/rollup/window) and the UDR/custom-kernel variant.

## 8. Open questions for the user (before implementing)
1. Compile into the **gqe build tree** (needs the full gqe build; heavy) vs a lighter path? (gqe
   isn't a conda package, so linking requires the build tree — confirms the heavier route.)
2. First targets: the 7 existing TPC-DS queries (safest), or jump to brand-new ones?
3. Name-based column helper now (less LLM error) or stay faithful to index-based as the hardcoded
   files do?
4. Keep using the local vLLM model + the same loop shape as `tpch_bench/agent/`?
