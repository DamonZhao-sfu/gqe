// SPDX-License-Identifier: Apache-2.0
// Fixed harness for the GQE-style codegen agent. Sets up the memory pool, registers every TPC-DS
// table present in the dataset (full schema from tpcds::table_definitions), calls the generated
// build_plan(), optimizes + executes it, and writes output.parquet. The LLM only writes build_plan.
//
// NOTE: not compile-verified in this environment; expect to iterate on your GPU box.
#include "build_plan.hpp"

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

#include <cudf/io/parquet.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static std::size_t pool_size()
{
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  return total_mem / 284 * 256;  // ~90% of total, 256B-aligned (matches benchmark/utility.hpp)
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: gqe_codegen_query <tpcds_dataset_dir>\n";
    return 2;
  }
  std::string const data = argv[1];

  // Memory pool (single GPU).
  auto const ps = pool_size();
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, ps, ps};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register every TPC-DS table that exists in the dataset dir, with its full schema.
  gqe::task_manager_context task_manager_ctx;
  gqe::catalog cat{&task_manager_ctx};
  for (auto const& [name, def] : gqe::utility::tpcds::table_definitions()) {
    if (!fs::is_directory(fs::path(data) / name)) continue;
    auto files = gqe::utility::get_parquet_files(data + "/" + name);
    if (files.empty()) continue;
    cat.register_table(name, def.columns, gqe::storage_kind::parquet_file{files},
                       gqe::partitioning_schema_kind::automatic{}, def.unique_keys);
  }

  // Build the (generated) plan.
  pb::Ctx ctx{&cat};
  pb::Rel plan = build_plan(ctx);

  // Optimize -> physical -> task graph -> execute.
  gqe::optimizer::optimization_configuration rules(
    {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys},
    {});
  gqe::optimizer::logical_optimizer optimizer(&rules, &cat);
  auto logical = optimizer.optimize(plan.node);

  gqe::physical_plan_builder plan_builder(&cat);
  auto physical = plan_builder.build(logical.get());

  gqe::query_context query_ctx(gqe::make_optimization_parameters());
  gqe::context_reference ctx_ref{&task_manager_ctx, &query_ctx};
  gqe::task_graph_builder graph_builder(ctx_ref, &cat);
  auto task_graph = graph_builder.build(physical.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, ctx_ref, task_graph.get());

  // Write result.
  auto sink = cudf::io::sink_info("output.parquet");
  auto opts = cudf::io::parquet_writer_options::builder(
    sink, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(opts);
  std::cerr << "wrote output.parquet\n";
  return 0;
}
