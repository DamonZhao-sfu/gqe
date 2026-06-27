/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* TPC-DS Q43 -- fused-kernel prototype (cf. q3_udr.cu)
 *
 * SELECT s_store_name, s_store_id, sum(case when d_day_name='Sunday' ...), ...(7 days)
 * FROM   date_dim, store_sales, store
 * WHERE  d_date_sk = ss_sold_date_sk AND s_store_sk = ss_store_sk
 *    AND s_gmt_offset = -5 AND d_year = 2000
 * GROUP BY s_store_name, s_store_id ORDER BY ... LIMIT 100
 *
 * store_sales is the fact (probe) table; date_dim and store are both PK dimensions. This collapses
 * the two broadcast hash joins into one probe kernel: build a (key -> row) map for each filtered
 * dimension, probe both per store_sales row, and keep three index arrays (store, date, ss). The
 * join output [s_store_name, s_store_id, d_day_name, ss_sales_price] is then materialized once with
 * cudf::gather, and the original conditional-SUM aggregate / sort / limit run unchanged.
 */

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/helpers.hpp>

#include <cub/warp/warp_scan.cuh>
#include <cuco/static_map.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/scan.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

void print_usage()
{
  std::cout << "Run TPC-DS Q43 benchmark with customized fused-join kernel" << std::endl
            << "./q43_udr <path-to-dataset>" << std::endl;
}

std::shared_ptr<gqe::logical::read_relation> read_table(
  std::string table_name,
  std::vector<std::string> column_names,
  gqe::catalog const* tpcds_catalog,
  std::shared_ptr<gqe::logical::project_relation> partial_filter_haystack = nullptr,
  std::unique_ptr<gqe::expression> partial_filter                         = nullptr)
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    column_types.push_back(tpcds_catalog->column_type(table_name, column_name));
  return std::make_shared<gqe::logical::read_relation>(
    partial_filter
      ? std::vector<std::shared_ptr<gqe::logical::relation>>{std::move(partial_filter_haystack)}
      : std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    std::move(partial_filter));  // partial_filter
}

constexpr cuco::empty_key<int64_t> empty_key_sentinel(std::numeric_limits<int64_t>::max());
constexpr cuco::empty_value<int64_t> empty_value_sentinel(-1);
constexpr int warp_size = 32;

// Fused task for the two-way star join store_sales x {date_dim, store}.
class custom_task : public gqe::task {
 public:
  custom_task(gqe::context_reference ctx_ref,
              int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<gqe::task> date_dim_table,
              std::shared_ptr<gqe::task> store_table,
              std::shared_ptr<gqe::task> store_sales_table);

  void execute() override;
};

custom_task::custom_task(gqe::context_reference ctx_ref,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<gqe::task> date_dim_table,
                         std::shared_ptr<gqe::task> store_table,
                         std::shared_ptr<gqe::task> store_sales_table)
  : gqe::task(ctx_ref,
              task_id,
              stage_id,
              {std::move(date_dim_table), std::move(store_table), std::move(store_sales_table)},
              {})
{
}

/*
 * Probe both dimension maps in one pass. Probe `store` first (s_gmt_offset filter is selective),
 * then `date_dim` (already narrowed to d_year=2000 by predicate pushdown). Survivors record the
 * store row index, the date row index, and the store_sales row index.
 */
template <int block_size, typename DateMapRef, typename StoreMapRef>
__global__ void probe_hash_maps(DateMapRef date_map,
                                StoreMapRef store_map,
                                cudf::column_device_view ss_sold_date_sk_column,
                                cudf::column_device_view ss_store_sk_column,
                                cudf::size_type const in_rows_per_warp,
                                int64_t* date_indices,
                                int64_t* store_indices,
                                int64_t* ss_indices,
                                cudf::size_type* out_rows_per_warp)
{
  __shared__ cudf::size_type warp_out_row_idx[block_size / warp_size];
  __shared__
    typename cub::WarpScan<int8_t>::TempStorage warp_scan_temp_storage[block_size / warp_size];

  int const global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int const global_warp_id   = global_thread_id / warp_size;
  int const local_warp_id    = threadIdx.x / warp_size;
  int const warp_lane        = global_thread_id % warp_size;

  cudf::size_type const start_idx = global_warp_id * in_rows_per_warp;
  cudf::size_type const end_idx =
    min((global_warp_id + 1) * in_rows_per_warp, ss_store_sk_column.size());

  if (warp_lane == 0) warp_out_row_idx[local_warp_id] = start_idx;
  __syncwarp();

  for (cudf::size_type warp_row_idx = start_idx; warp_row_idx < end_idx;
       warp_row_idx += warp_size) {
    auto const thread_row_idx = warp_row_idx + warp_lane;
    int64_t store_idx         = -1;
    int64_t date_idx          = -1;

    bool pass = (thread_row_idx < end_idx);
    if (pass) {  // store -- most selective
      if (ss_store_sk_column.is_valid(thread_row_idx)) {
        auto const t = store_map.find(ss_store_sk_column.element<int64_t>(thread_row_idx));
        if (t != store_map.end()) store_idx = t->second;
      }
      pass = (store_idx != -1);
    }
    if (pass) {  // date_dim
      if (ss_sold_date_sk_column.is_valid(thread_row_idx)) {
        auto const t = date_map.find(ss_sold_date_sk_column.element<int64_t>(thread_row_idx));
        if (t != date_map.end()) date_idx = t->second;
      }
      pass = (date_idx != -1);
    }

    int8_t const num_matches = pass ? 1 : 0;

    int8_t out_offset;
    cub::WarpScan<int8_t>(warp_scan_temp_storage[local_warp_id])
      .ExclusiveSum(num_matches, out_offset);

    if (num_matches == 1) {
      auto const thread_out_row_idx     = warp_out_row_idx[local_warp_id] + out_offset;
      date_indices[thread_out_row_idx]  = date_idx;
      store_indices[thread_out_row_idx] = store_idx;
      ss_indices[thread_out_row_idx]    = thread_row_idx;
    }
    __syncwarp();

    if (warp_lane == warp_size - 1) warp_out_row_idx[local_warp_id] += (out_offset + num_matches);
    __syncwarp();
  }

  if (warp_lane == 0)
    out_rows_per_warp[global_warp_id] = warp_out_row_idx[local_warp_id] - start_idx;
}

__global__ void stream_compaction(int64_t const* in_array,
                                  int64_t* out_array,
                                  cudf::size_type const in_elements_per_chunk,
                                  cudf::size_type const* out_chunk_offsets,
                                  int num_chunks)
{
  for (int chunk_idx = blockIdx.x; chunk_idx < num_chunks; chunk_idx += gridDim.x) {
    auto const out_offset_start = chunk_idx == 0 ? 0 : out_chunk_offsets[chunk_idx - 1];
    auto const out_offset_end   = out_chunk_offsets[chunk_idx];
    auto const chunk_in_offset  = in_elements_per_chunk * chunk_idx;

    for (cudf::size_type out_offset = out_offset_start + threadIdx.x; out_offset < out_offset_end;
         out_offset += blockDim.x) {
      out_array[out_offset] = in_array[chunk_in_offset + out_offset - out_offset_start];
    }
  }
}

template <typename MapRef>
void insert_key_map(MapRef insert_ref, cudf::table_view build_table)
{
  auto key_column = cudf::column_device_view::create(build_table.column(0));
  thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                   thrust::make_counting_iterator<cudf::size_type>(key_column->size()),
                   [map = insert_ref, keys = *key_column] __device__(auto row_idx) mutable {
                     map.insert(
                       thrust::pair<int64_t, int64_t>(keys.element<int64_t>(row_idx), row_idx));
                   });
}

void custom_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();

  auto date_dim_table    = dependent_tasks[0]->result().value();  // [d_date_sk, d_day_name, d_year]
  auto store_table       = dependent_tasks[1]->result().value();  // [s_store_sk, s_store_id, s_store_name, s_gmt_offset]
  auto store_sales_table = dependent_tasks[2]->result().value();  // [ss_sold_date_sk, ss_store_sk, ss_sales_price]

  rmm::mr::polymorphic_allocator<cuco::pair<int64_t, int64_t>> polly_alloc;
  auto stream_alloc = rmm::mr::stream_allocator_adaptor(polly_alloc, cudf::get_default_stream());

  auto constexpr load_factor = 0.5;
  auto make_map              = [&](cudf::table_view t) {
    return cuco::static_map{static_cast<std::size_t>(std::ceil(t.num_rows() / load_factor)),
                            empty_key_sentinel,
                            empty_value_sentinel,
                            thrust::equal_to<int64_t>{},
                            cuco::linear_probing<1, cuco::default_hash_function<int64_t>>{},
                            {},
                            {},
                            stream_alloc};
  };
  auto date_map  = make_map(date_dim_table);
  auto store_map = make_map(store_table);
  insert_key_map(date_map.ref(cuco::insert), date_dim_table);
  insert_key_map(store_map.ref(cuco::insert), store_table);

  auto const ss_num_rows = store_sales_table.num_rows();
  rmm::device_uvector<int64_t> date_indices(ss_num_rows, cudf::get_default_stream());
  rmm::device_uvector<int64_t> store_indices(ss_num_rows, cudf::get_default_stream());
  rmm::device_uvector<int64_t> ss_indices(ss_num_rows, cudf::get_default_stream());

  // store_sales columns: 0 ss_sold_date_sk, 1 ss_store_sk, 2 ss_sales_price
  auto ss_sold_date_sk_column = cudf::column_device_view::create(store_sales_table.column(0));
  auto ss_store_sk_column     = cudf::column_device_view::create(store_sales_table.column(1));

  constexpr int block_size    = 128;
  constexpr int grid_size     = 1600;
  constexpr auto num_warps    = grid_size * (block_size / warp_size);
  auto const in_rows_per_warp = (ss_num_rows + num_warps - 1) / num_warps;
  rmm::device_uvector<cudf::size_type> out_rows_per_warp(num_warps, cudf::get_default_stream());

  probe_hash_maps<block_size><<<grid_size, block_size>>>(date_map.ref(cuco::find),
                                                         store_map.ref(cuco::find),
                                                         *ss_sold_date_sk_column,
                                                         *ss_store_sk_column,
                                                         in_rows_per_warp,
                                                         date_indices.data(),
                                                         store_indices.data(),
                                                         ss_indices.data(),
                                                         out_rows_per_warp.data());
  cudf::get_default_stream().synchronize();

  thrust::inclusive_scan(
    thrust::device, out_rows_per_warp.begin(), out_rows_per_warp.end(), out_rows_per_warp.begin());

  auto const out_rows_total = out_rows_per_warp.back_element(cudf::get_default_stream());
  rmm::device_uvector<int64_t> compact_date_indices(out_rows_total, cudf::get_default_stream());
  rmm::device_uvector<int64_t> compact_store_indices(out_rows_total, cudf::get_default_stream());
  rmm::device_uvector<int64_t> compact_ss_indices(out_rows_total, cudf::get_default_stream());

  stream_compaction<<<grid_size, block_size>>>(date_indices.data(),
                                               compact_date_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);
  stream_compaction<<<grid_size, block_size>>>(store_indices.data(),
                                               compact_store_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);
  stream_compaction<<<grid_size, block_size>>>(ss_indices.data(),
                                               compact_ss_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);
  cudf::get_default_stream().synchronize();

  auto date_indices_column =
    std::make_unique<cudf::column>(std::move(compact_date_indices), rmm::device_buffer{}, 0);
  auto store_indices_column =
    std::make_unique<cudf::column>(std::move(compact_store_indices), rmm::device_buffer{}, 0);
  auto ss_indices_column =
    std::make_unique<cudf::column>(std::move(compact_ss_indices), rmm::device_buffer{}, 0);

  auto materialize_column =
    [](cudf::table_view input_table, cudf::size_type column_idx, cudf::column_view gather_map) {
      auto gathered_column = cudf::gather(input_table.select({column_idx}), gather_map)->release();
      return std::move(gathered_column[0]);
    };

  // Output layout expected by the downstream aggregate (matches original q43 pre-aggregate cols):
  //   [s_store_name, s_store_id, d_day_name, ss_sales_price]
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  out_columns.push_back(materialize_column(store_table, 2, store_indices_column->view()));
  out_columns.push_back(materialize_column(store_table, 1, store_indices_column->view()));
  out_columns.push_back(materialize_column(date_dim_table, 1, date_indices_column->view()));
  out_columns.push_back(materialize_column(store_sales_table, 2, ss_indices_column->view()));

  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

std::vector<std::shared_ptr<gqe::task>> custom_relation_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  auto date_dim_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
  task_id++;
  auto store_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[1]);
  task_id++;

  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  for (auto const& sales_table : children_tasks[2]) {
    pipeline_results.push_back(std::make_shared<custom_task>(
      ctx_ref, task_id, stage_id, date_dim_table, store_table, sales_table));
    task_id++;
  }
  return pipeline_results;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    print_usage();
    return EXIT_FAILURE;
  }
  std::string const dataset_location(argv[1]);

  auto const pool_size = gqe::benchmark::get_memory_pool_size();
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, pool_size, pool_size};
  rmm::mr::set_current_device_resource(&pool_mr);

  gqe::task_manager_context task_manager_ctx;
  gqe::catalog tpcds_catalog{&task_manager_ctx};
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_day_name", cudf::data_type(cudf::type_id::STRING)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{},
                               {{"d_date_sk"}});  // PRIMARY KEY
  tpcds_catalog.register_table("store_sales",
                               {{"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_store_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "store",
    {{"s_store_sk", cudf::data_type(cudf::type_id::INT64)},
     {"s_store_id", cudf::data_type(cudf::type_id::STRING)},
     {"s_store_name", cudf::data_type(cudf::type_id::STRING)},
     {"s_gmt_offset", cudf::data_type(cudf::type_id::FLOAT64)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/store")},
    gqe::partitioning_schema_kind::automatic{},
    {{"s_store_sk"}});  // PRIMARY KEY

  // date_dim: filter d_year = 2000
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_day_name", "d_year"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<int64_t>>(2000)),
    std::vector<cudf::size_type>({0, 1, 2}));

  // store: filter s_gmt_offset = -5
  std::shared_ptr<gqe::logical::relation> store_table = read_table(
    "store", {"s_store_sk", "s_store_id", "s_store_name", "s_gmt_offset"}, &tpcds_catalog);
  store_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(store_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(3),
                                            std::make_shared<gqe::literal_expression<double>>(-5)),
    std::vector<cudf::size_type>({0, 1, 2, 3}));

  // Predicate pushdown: restrict store_sales reads to ss_sold_date_sk in the filtered date keys.
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
  auto const partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table, std::vector<std::shared_ptr<gqe::logical::relation>>(), std::move(col_0_exprs));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(0)},  // ss_sold_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_sold_date_sk", "ss_store_sk", "ss_sales_price"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // Fuse the two-way star join into a single user-defined relation.
  // Children order MUST match custom_relation_generate_tasks / custom_task:
  //   [date_dim, store, store_sales]
  // Output columns: [s_store_name, s_store_id, d_day_name, ss_sales_price]
  store_sales_table = std::make_shared<gqe::logical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(
      {std::move(date_dim_table), std::move(store_table), std::move(store_sales_table)}),
    custom_relation_generate_tasks,
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::STRING),
                                  cudf::data_type(cudf::type_id::STRING),
                                  cudf::data_type(cudf::type_id::STRING),
                                  cudf::data_type(cudf::type_id::FLOAT64)}),
    false);

  // GROUP BY s_store_name, s_store_id ; SUM(case when d_day_name=<day> then ss_sales_price end)
  std::vector<std::unique_ptr<gqe::expression>> agg_keys;
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));

  auto day_expr         = std::make_shared<gqe::column_reference_expression>(2);
  auto sales_price_expr = std::make_shared<gqe::column_reference_expression>(3);
  auto null_expr        = std::make_shared<gqe::literal_expression<double>>(0, true);

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> agg_measures;
  for (auto const day :
       {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}) {
    agg_measures.emplace_back(
      cudf::aggregation::SUM,
      std::make_unique<gqe::if_then_else_expression>(
        std::make_shared<gqe::equal_expression>(
          day_expr, std::make_shared<gqe::literal_expression<std::string>>(day)),
        sales_price_expr,
        null_expr));
  }

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(agg_keys),
    std::move(agg_measures));

  // ORDER BY all 9 columns
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  for (cudf::size_type col = 0; col < 9; ++col)
    sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(col));
  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>(9, cudf::order::ASCENDING),
    std::vector<cudf::null_order>(9, cudf::null_order::BEFORE),
    std::move(sort_exprs));

  // LIMIT 100
  store_sales_table =
    std::make_shared<gqe::logical::fetch_relation>(std::move(store_sales_table), 0, 100);

  auto logical_plan = std::move(store_sales_table);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::query_context query_ctx(gqe::make_optimization_parameters());
  gqe::context_reference ctx_ref{&task_manager_ctx, &query_ctx};

  gqe::task_graph_builder graph_builder(ctx_ref, &tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, ctx_ref, task_graph.get());

  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  std::ofstream out;
  out.open("bandwidth.json");
  out << query_ctx.disk_timer.to_string();
  out << query_ctx.h2d_timer.to_string();
  out << query_ctx.decomp_timer.to_string();
  out << query_ctx.decode_timer.to_string();

  return 0;
}
