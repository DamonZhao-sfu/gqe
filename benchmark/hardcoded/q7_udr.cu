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

/* TPC-DS Q7 -- fused-kernel prototype (cf. q3_udr.cu)
 *
 * SELECT i_item_id, avg(ss_quantity), avg(ss_list_price), avg(ss_coupon_amt), avg(ss_sales_price)
 * FROM   store_sales, customer_demographics, date_dim, item, promotion
 * WHERE  ss_sold_date_sk = d_date_sk AND ss_item_sk = i_item_sk
 *    AND ss_cdemo_sk = cd_demo_sk AND ss_promo_sk = p_promo_sk
 *    AND cd_gender='M' AND cd_marital_status='S' AND cd_education_status='College'
 *    AND (p_channel_email='N' OR p_channel_event='N') AND d_year=2000
 * GROUP BY i_item_id ORDER BY i_item_id LIMIT 100
 *
 * store_sales is the single fact (probe) table; the four dimensions all join PK = FK.
 * Three of them (customer_demographics, date_dim, promotion) contribute NO output columns --
 * they are pure semi-join filters. Only `item` supplies a payload (i_item_id) for the group-by.
 *
 * Instead of running four separate broadcast hash joins (each materializing a full intermediate
 * table), this prototype builds one hash map per (pre-filtered) dimension and probes all four in a
 * SINGLE fused kernel. A store_sales row survives only if it matches every dimension; we then keep
 * just two index arrays -- the store_sales row index (for the four measures) and the item row index
 * (for i_item_id) -- and materialize the join output once at the end.
 */

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
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
  std::cout << "Run TPC-DS Q7 benchmark with customized fused-join kernel" << std::endl
            << "./q7_udr <path-to-dataset>" << std::endl;
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

// The join keys (surrogate keys) are never max int64_t in the TPC-DS data.
constexpr cuco::empty_key<int64_t> empty_key_sentinel(std::numeric_limits<int64_t>::max());
// Row indices are non-negative in nature.
constexpr cuco::empty_value<int64_t> empty_value_sentinel(-1);

// Number of threads in a warp
constexpr int warp_size = 32;

// A customized task that fuses the four-way star join (store_sales x {customer_demographics,
// date_dim, promotion, item}) into a single probe kernel.
class custom_task : public gqe::task {
 public:
  custom_task(gqe::context_reference ctx_ref,
              int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<gqe::task> cd_table,
              std::shared_ptr<gqe::task> date_dim_table,
              std::shared_ptr<gqe::task> promotion_table,
              std::shared_ptr<gqe::task> item_table,
              std::shared_ptr<gqe::task> store_sales_table);

  void execute() override;
};

custom_task::custom_task(gqe::context_reference ctx_ref,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<gqe::task> cd_table,
                         std::shared_ptr<gqe::task> date_dim_table,
                         std::shared_ptr<gqe::task> promotion_table,
                         std::shared_ptr<gqe::task> item_table,
                         std::shared_ptr<gqe::task> store_sales_table)
  : gqe::task(ctx_ref,
              task_id,
              stage_id,
              {std::move(cd_table),
               std::move(date_dim_table),
               std::move(promotion_table),
               std::move(item_table),
               std::move(store_sales_table)},
              {})
{
}

/*
 * Fused kernel for the four-way star join. Warp `idx` processes store_sales rows in
 * [idx * in_rows_per_warp, (idx + 1) * in_rows_per_warp) and writes survivors into *item_indices*
 * and *ss_indices* starting at `idx * in_rows_per_warp`. Because every dimension key is a primary
 * key, each probe yields at most one match, so a warp never writes beyond its slot. The number of
 * survivors per warp is stored in *out_rows_per_warp* for a later stream-compaction pass.
 *
 * Probe order is chosen by selectivity (cheapest-to-fail first): customer_demographics is the most
 * selective filter, then promotion, then date_dim, and finally item (unfiltered, but needed for the
 * i_item_id payload). The short-circuit && means most rows touch only one hash map.
 */
template <int block_size,
          typename CdMapRef,
          typename PromoMapRef,
          typename DateMapRef,
          typename ItemMapRef>
__global__ void probe_hash_maps(CdMapRef cd_map,
                                PromoMapRef promo_map,
                                DateMapRef date_map,
                                ItemMapRef item_map,
                                cudf::column_device_view ss_cdemo_sk_column,
                                cudf::column_device_view ss_promo_sk_column,
                                cudf::column_device_view ss_sold_date_sk_column,
                                cudf::column_device_view ss_item_sk_column,
                                cudf::size_type const in_rows_per_warp,
                                int64_t* item_indices,
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
    min((global_warp_id + 1) * in_rows_per_warp, ss_item_sk_column.size());

  if (warp_lane == 0) warp_out_row_idx[local_warp_id] = start_idx;
  __syncwarp();

  for (cudf::size_type warp_row_idx = start_idx; warp_row_idx < end_idx;
       warp_row_idx += warp_size) {
    // Note: this block contains a whole-warp scan, so every thread in the warp must reach it.
    auto const thread_row_idx = warp_row_idx + warp_lane;
    int64_t item_idx          = -1;

    bool pass = (thread_row_idx < end_idx);
    // Inner joins drop NULL foreign keys, so each probe is guarded by is_valid().
    if (pass)  // 1) customer_demographics -- most selective
      pass = ss_cdemo_sk_column.is_valid(thread_row_idx) &&
             cd_map.find(ss_cdemo_sk_column.element<int64_t>(thread_row_idx)) != cd_map.end();
    if (pass)  // 2) promotion
      pass = ss_promo_sk_column.is_valid(thread_row_idx) &&
             promo_map.find(ss_promo_sk_column.element<int64_t>(thread_row_idx)) != promo_map.end();
    if (pass)  // 3) date_dim
      pass =
        ss_sold_date_sk_column.is_valid(thread_row_idx) &&
        date_map.find(ss_sold_date_sk_column.element<int64_t>(thread_row_idx)) != date_map.end();
    if (pass) {  // 4) item -- unfiltered, but we need its row index for i_item_id
      if (ss_item_sk_column.is_valid(thread_row_idx)) {
        auto const item_tuple = item_map.find(ss_item_sk_column.element<int64_t>(thread_row_idx));
        if (item_tuple != item_map.end()) item_idx = item_tuple->second;
      }
      pass = (item_idx != -1);
    }

    int8_t const num_matches = pass ? 1 : 0;

    // Whole-warp exclusive scan to compute each survivor's output slot.
    int8_t out_offset;
    cub::WarpScan<int8_t>(warp_scan_temp_storage[local_warp_id])
      .ExclusiveSum(num_matches, out_offset);

    if (num_matches == 1) {
      auto const thread_out_row_idx    = warp_out_row_idx[local_warp_id] + out_offset;
      item_indices[thread_out_row_idx] = item_idx;
      ss_indices[thread_out_row_idx]   = thread_row_idx;
    }
    __syncwarp();

    if (warp_lane == warp_size - 1) warp_out_row_idx[local_warp_id] += (out_offset + num_matches);
    __syncwarp();
  }

  if (warp_lane == 0)
    out_rows_per_warp[global_warp_id] = warp_out_row_idx[local_warp_id] - start_idx;
}

/*
 * Compacts the gappy per-warp index slots into a dense array. For warp `idx` it copies from
 * `in_array + in_elements_per_chunk * idx` to `out_array + out_chunk_offsets[idx - 1]`.
 */
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

// Insert (key -> row_index) for column 0 of `build_table` into an already-constructed cuco map.
// (The map is constructed in-place at the call site -- cf. q3_udr.cu -- because the pinned cuco
// version's static_map is not returned by value.)
template <typename MapRef>
void insert_key_map(MapRef insert_ref, cudf::table_view build_table)
{
  auto key_column = cudf::column_device_view::create(build_table.column(0));
  thrust::for_each(thrust::make_counting_iterator<cudf::size_type>(0),
                   thrust::make_counting_iterator<cudf::size_type>(key_column->size()),
                   [map = insert_ref, keys = *key_column] __device__(auto row_idx) mutable {
                     // Surrogate keys (PKs) are non-NULL per the TPC-DS spec.
                     map.insert(
                       thrust::pair<int64_t, int64_t>(keys.element<int64_t>(row_idx), row_idx));
                   });
}

void custom_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();

  auto cd_table          = dependent_tasks[0]->result().value();  // [cd_demo_sk, ...] (filtered)
  auto date_dim_table    = dependent_tasks[1]->result().value();  // [d_date_sk, ...]  (filtered)
  auto promotion_table   = dependent_tasks[2]->result().value();  // [p_promo_sk, ...] (filtered)
  auto item_table        = dependent_tasks[3]->result().value();  // [i_item_sk, i_item_id]
  auto store_sales_table = dependent_tasks[4]->result().value();  // 8 columns, see main()

  // One allocator shared by all four hash maps, bound to the default stream.
  rmm::mr::polymorphic_allocator<cuco::pair<int64_t, int64_t>> polly_alloc;
  auto stream_alloc = rmm::mr::stream_allocator_adaptor(polly_alloc, cudf::get_default_stream());

  // Build one (key -> row_idx) map per (pre-filtered) dimension. The dimensions are tiny relative
  // to store_sales, so a broadcast build is cheap; only `item` actually needs its payload row idx.
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
  auto cd_map    = make_map(cd_table);
  auto promo_map = make_map(promotion_table);
  auto date_map  = make_map(date_dim_table);
  auto item_map  = make_map(item_table);
  insert_key_map(cd_map.ref(cuco::insert), cd_table);
  insert_key_map(promo_map.ref(cuco::insert), promotion_table);
  insert_key_map(date_map.ref(cuco::insert), date_dim_table);
  insert_key_map(item_map.ref(cuco::insert), item_table);

  // Output size is unknown a priori but bounded by the number of store_sales rows.
  auto const ss_num_rows = store_sales_table.num_rows();
  rmm::device_uvector<int64_t> item_indices(ss_num_rows, cudf::get_default_stream());
  rmm::device_uvector<int64_t> ss_indices(ss_num_rows, cudf::get_default_stream());

  // store_sales column layout (see register/read order in main):
  //   0 ss_quantity, 1 ss_list_price, 2 ss_coupon_amt, 3 ss_sales_price,
  //   4 ss_sold_date_sk, 5 ss_item_sk, 6 ss_cdemo_sk, 7 ss_promo_sk
  auto ss_cdemo_sk_column     = cudf::column_device_view::create(store_sales_table.column(6));
  auto ss_promo_sk_column     = cudf::column_device_view::create(store_sales_table.column(7));
  auto ss_sold_date_sk_column = cudf::column_device_view::create(store_sales_table.column(4));
  auto ss_item_sk_column      = cudf::column_device_view::create(store_sales_table.column(5));

  constexpr int block_size    = 128;  // must be a multiple of warp_size
  constexpr int grid_size     = 1600;  // chosen empirically, as in q3_udr
  constexpr auto num_warps    = grid_size * (block_size / warp_size);
  auto const in_rows_per_warp = (ss_num_rows + num_warps - 1) / num_warps;
  rmm::device_uvector<cudf::size_type> out_rows_per_warp(num_warps, cudf::get_default_stream());

  probe_hash_maps<block_size><<<grid_size, block_size>>>(cd_map.ref(cuco::find),
                                                         promo_map.ref(cuco::find),
                                                         date_map.ref(cuco::find),
                                                         item_map.ref(cuco::find),
                                                         *ss_cdemo_sk_column,
                                                         *ss_promo_sk_column,
                                                         *ss_sold_date_sk_column,
                                                         *ss_item_sk_column,
                                                         in_rows_per_warp,
                                                         item_indices.data(),
                                                         ss_indices.data(),
                                                         out_rows_per_warp.data());
  cudf::get_default_stream().synchronize();

  // Per-warp survivor counts -> exclusive/inclusive offsets for compaction.
  thrust::inclusive_scan(
    thrust::device, out_rows_per_warp.begin(), out_rows_per_warp.end(), out_rows_per_warp.begin());

  auto const out_rows_total = out_rows_per_warp.back_element(cudf::get_default_stream());
  rmm::device_uvector<int64_t> compact_item_indices(out_rows_total, cudf::get_default_stream());
  rmm::device_uvector<int64_t> compact_ss_indices(out_rows_total, cudf::get_default_stream());

  // FIXME: replace with CUB DeviceBatchMemcpy once available.
  stream_compaction<<<grid_size, block_size>>>(item_indices.data(),
                                               compact_item_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);
  stream_compaction<<<grid_size, block_size>>>(ss_indices.data(),
                                               compact_ss_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);
  cudf::get_default_stream().synchronize();

  auto item_indices_column =
    std::make_unique<cudf::column>(std::move(compact_item_indices), rmm::device_buffer{}, 0);
  auto ss_indices_column =
    std::make_unique<cudf::column>(std::move(compact_ss_indices), rmm::device_buffer{}, 0);

  // Materialize the join output via a single gather per output column.
  auto materialize_column =
    [](cudf::table_view input_table, cudf::size_type column_idx, cudf::column_view gather_map) {
      auto gathered_column = cudf::gather(input_table.select({column_idx}), gather_map)->release();
      return std::move(gathered_column[0]);
    };

  // Output layout expected by the downstream aggregate (matches the original q7 pre-aggregate cols):
  //   [ss_quantity, ss_list_price, ss_coupon_amt, ss_sales_price, i_item_id]
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  out_columns.push_back(materialize_column(store_sales_table, 0, ss_indices_column->view()));
  out_columns.push_back(materialize_column(store_sales_table, 1, ss_indices_column->view()));
  out_columns.push_back(materialize_column(store_sales_table, 2, ss_indices_column->view()));
  out_columns.push_back(materialize_column(store_sales_table, 3, ss_indices_column->view()));
  out_columns.push_back(materialize_column(item_table, 1, item_indices_column->view()));

  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

// Functor that turns the child relations' tasks into our fused custom_task(s). The four dimensions
// are concatenated into single broadcast tasks; we emit one custom_task per store_sales partition.
std::vector<std::shared_ptr<gqe::task>> custom_relation_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::context_reference ctx_ref,
  int32_t& task_id,
  int32_t stage_id)
{
  auto cd_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[0]);
  task_id++;
  auto date_dim_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[1]);
  task_id++;
  auto promotion_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[2]);
  task_id++;
  auto item_table =
    std::make_shared<gqe::concatenate_task>(ctx_ref, task_id, stage_id, children_tasks[3]);
  task_id++;

  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  for (auto const& sales_table : children_tasks[4]) {
    pipeline_results.push_back(std::make_shared<custom_task>(
      ctx_ref, task_id, stage_id, cd_table, date_dim_table, promotion_table, item_table,
      sales_table));
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

  // Configure the memory pool (single GPU).
  auto const pool_size = gqe::benchmark::get_memory_pool_size();
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, pool_size, pool_size};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the input tables.
  gqe::task_manager_context task_manager_ctx;
  gqe::catalog tpcds_catalog{&task_manager_ctx};
  tpcds_catalog.register_table("store_sales",
                               {{"ss_quantity", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_coupon_amt", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_cdemo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_promo_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("customer_demographics",
                               {{"cd_demo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"cd_gender", cudf::data_type(cudf::type_id::STRING)},
                                {"cd_marital_status", cudf::data_type(cudf::type_id::STRING)},
                                {"cd_education_status", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/customer_demographics")},
                               gqe::partitioning_schema_kind::automatic{},
                               {{"cd_demo_sk"}});  // PRIMARY KEY
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{},
                               {{"d_date_sk"}});  // PRIMARY KEY
  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT64)},
     {"i_item_id", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/item")},
    gqe::partitioning_schema_kind::automatic{},
    {{"i_item_sk"}});  // PRIMARY KEY
  tpcds_catalog.register_table("promotion",
                               {{"p_promo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"p_channel_email", cudf::data_type(cudf::type_id::STRING)},
                                {"p_channel_event", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/promotion")},
                               gqe::partitioning_schema_kind::automatic{},
                               {{"p_promo_sk"}});  // PRIMARY KEY

  // date_dim: filter d_year = 2000
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(2000)),
    std::vector<cudf::size_type>({0, 1}));

  // Predicate pushdown: restrict store_sales reads to ss_sold_date_sk in the filtered date keys.
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
  auto const partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table, std::vector<std::shared_ptr<gqe::logical::relation>>(), std::move(col_0_exprs));
  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(4)},  // ss_sold_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_quantity",
                "ss_list_price",
                "ss_coupon_amt",
                "ss_sales_price",
                "ss_sold_date_sk",
                "ss_item_sk",
                "ss_cdemo_sk",
                "ss_promo_sk"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // customer_demographics: filter gender='M' AND marital='S' AND education='College'
  std::shared_ptr<gqe::logical::relation> customer_demographics_table =
    read_table("customer_demographics",
               {"cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"},
               &tpcds_catalog);
  customer_demographics_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(customer_demographics_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<std::string>>("M")),
      std::make_shared<gqe::logical_and_expression>(
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(2),
          std::make_shared<gqe::literal_expression<std::string>>("S")),
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(3),
          std::make_shared<gqe::literal_expression<std::string>>("College")))),
    std::vector<cudf::size_type>({0, 1, 2, 3}));

  // item: no filter
  std::shared_ptr<gqe::logical::relation> item_table =
    read_table("item", {"i_item_sk", "i_item_id"}, &tpcds_catalog);

  // promotion: filter (p_channel_email='N' OR p_channel_event='N')
  std::shared_ptr<gqe::logical::relation> promotion_table =
    read_table("promotion", {"p_promo_sk", "p_channel_email", "p_channel_event"}, &tpcds_catalog);
  promotion_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(promotion_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_or_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<std::string>>("N")),
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<std::string>>("N"))),
    std::vector<cudf::size_type>({0, 1, 2}));

  // Fuse the four-way star join into a single user-defined relation.
  // Children order MUST match custom_relation_generate_tasks / custom_task:
  //   [customer_demographics, date_dim, promotion, item, store_sales]
  // Output columns: [ss_quantity, ss_list_price, ss_coupon_amt, ss_sales_price, i_item_id]
  store_sales_table = std::make_shared<gqe::logical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>({std::move(customer_demographics_table),
                                                          std::move(date_dim_table),
                                                          std::move(promotion_table),
                                                          std::move(item_table),
                                                          std::move(store_sales_table)}),
    custom_relation_generate_tasks,
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::INT64),
                                  cudf::data_type(cudf::type_id::FLOAT64),
                                  cudf::data_type(cudf::type_id::FLOAT64),
                                  cudf::data_type(cudf::type_id::FLOAT64),
                                  cudf::data_type(cudf::type_id::STRING)}),
    false);

  // GROUP BY i_item_id, AVG(ss_quantity/list_price/coupon_amt/sales_price)
  std::vector<std::unique_ptr<gqe::expression>> groupby_keys;
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(4));  // i_item_id

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> groupby_values;
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(0)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(1)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(2)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(3)));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby_keys),
    std::move(groupby_values));

  // ORDER BY i_item_id
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
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

  // Write the result to disk.
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  // Write bandwidth/timing info.
  std::ofstream out;
  out.open("bandwidth.json");
  out << query_ctx.disk_timer.to_string();
  out << query_ctx.h2d_timer.to_string();
  out << query_ctx.decomp_timer.to_string();
  out << query_ctx.decode_timer.to_string();

  return 0;
}
