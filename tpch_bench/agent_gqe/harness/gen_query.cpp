// SPDX-License-Identifier: Apache-2.0
// WHOLE-FILE example: TPC-DS Q3 as a complete standalone program (its own main), using the
// name-based DSL (plan_builder.hpp) + runtime helpers (gqe_runtime.hpp). This is the shape the
// agent generates: includes, main(argv[1]=dataset dir), pool, register tables, build plan, execute.
//
// TPC-DS Q3: store_sales x item x date_dim; d_moy=11, i_manufact_id=128;
//   group by d_year,i_brand_id,i_brand sum(ss_ext_sales_price); order by d_year, sum desc, brand_id; limit 100.
#include "gqe_runtime.hpp"
#include "plan_builder.hpp"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: q3 <tpcds_dataset_dir>\n";
    return 2;
  }
  std::string const data = argv[1];

  // Memory pool (single GPU).
  rmm::mr::cuda_memory_resource cuda_mr;
  auto const ps = gqe_pool_size();
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&cuda_mr, ps, ps};
  rmm::mr::set_current_device_resource(&pool);

  // Catalog with all present TPC-DS tables registered.
  gqe::task_manager_context tm;
  gqe::catalog cat{&tm};
  register_tpcds(cat, data);

  // Build the plan with the name-based DSL.
  pb::Ctx c{&cat};
  using namespace pb;

  auto date_dim = filter(scan(c, "date_dim", {"d_date_sk", "d_year", "d_moy"}),
                         eq(col("d_moy"), lit<int64_t>(11)));
  auto item = filter(scan(c, "item", {"i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"}),
                     eq(col("i_manufact_id"), lit<int64_t>(128)));
  auto store_sales = scan(c, "store_sales", {"ss_item_sk", "ss_sold_date_sk", "ss_ext_sales_price"});

  auto j1 = join(store_sales, item, "ss_item_sk", "i_item_sk", gqe::join_type_type::inner,
                 {"ss_sold_date_sk", "ss_ext_sales_price", "i_brand_id", "i_brand"});
  auto j2 = join(j1, date_dim, "ss_sold_date_sk", "d_date_sk", gqe::join_type_type::inner,
                 {"ss_ext_sales_price", "i_brand_id", "i_brand", "d_year"});

  auto g = aggregate(j2, {"d_year", "i_brand_id", "i_brand"},
                     {{cudf::aggregation::SUM, col("ss_ext_sales_price"), "sum_agg"}});
  auto s = sort(g, {{"d_year", cudf::order::ASCENDING},
                    {"sum_agg", cudf::order::DESCENDING},
                    {"i_brand_id", cudf::order::ASCENDING}});
  pb::Rel plan = limit(s, 100);

  gqe_run_and_write(tm, cat, plan);  // writes output.parquet
  std::cerr << "wrote output.parquet\n";
  return 0;
}
