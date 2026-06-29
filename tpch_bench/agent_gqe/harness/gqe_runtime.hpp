// SPDX-License-Identifier: Apache-2.0
// Small runtime helpers so a generated WHOLE-FILE query program (with its own main, like q3.cpp)
// stays short: pool size, register all present TPC-DS tables, and optimize+execute+write a plan.
// The generated file uses these + the name-based DSL (plan_builder.hpp).
#pragma once

#include "plan_builder.hpp"

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/tpcds.hpp>
#include <gqe/utility/tpch.hpp>

#include <cudf/io/parquet.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <string>

inline std::size_t gqe_pool_size()
{
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  return total_mem / 284 * 256;  // ~90% of total, 256B-aligned
}

// Register every table from `defs` that exists under <data>/<table>/, with its full schema.
template <typename Defs>
inline void register_from(gqe::catalog& cat, std::string const& data, Defs const& defs)
{
  namespace fs = std::filesystem;
  for (auto const& [name, def] : defs) {
    if (!fs::is_directory(fs::path(data) / name)) continue;
    auto files = gqe::utility::get_parquet_files(data + "/" + name);
    if (files.empty()) continue;
    cat.register_table(name, def.columns, gqe::storage_kind::parquet_file{files},
                       gqe::partitioning_schema_kind::automatic{}, def.unique_keys);
  }
}

// Register all present TPC-DS / TPC-H tables. Call exactly ONE in main (table names overlap, e.g.
// `customer`, with different schemas across the two benchmarks).
inline void register_tpcds(gqe::catalog& cat, std::string const& data)
{
  register_from(cat, data, gqe::utility::tpcds::table_definitions());
}
inline void register_tpch(gqe::catalog& cat, std::string const& data)
{
  register_from(cat, data, gqe::utility::tpch::table_definitions());
}

// Optimize -> physical -> task graph -> execute -> write the result to `out`.
inline void gqe_run_and_write(gqe::task_manager_context& tm, gqe::catalog& cat, pb::Rel const& plan,
                              std::string const& out = "output.parquet")
{
  gqe::optimizer::optimization_configuration rules(
    {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys},
    {});
  gqe::optimizer::logical_optimizer optimizer(&rules, &cat);
  auto logical = optimizer.optimize(plan.node);

  gqe::physical_plan_builder ppb(&cat);
  auto physical = ppb.build(logical.get());

  gqe::query_context qctx(gqe::make_optimization_parameters());
  gqe::context_reference ctx_ref{&tm, &qctx};
  gqe::task_graph_builder gb(ctx_ref, &cat);
  auto tg = gb.build(physical.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, ctx_ref, tg.get());

  auto sink = cudf::io::sink_info(out);
  auto opts = cudf::io::parquet_writer_options::builder(sink, tg->root_tasks[0]->result().value());
  cudf::io::write_parquet(opts);
}
