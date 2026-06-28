// SPDX-License-Identifier: Apache-2.0
// DEFAULT / SEED plan: TPC-DS Q3 expressed with the name-based DSL (plan_builder.hpp).
// The agent OVERWRITES this file per query at runtime; `git checkout` restores this Q3 default.
// Doubles as a smoke test for the harness: build target `gqe_codegen_query`, run on a TPC-DS
// dataset, and it should match the DuckDB Q3 reference.
//
// TPC-DS Q3: store_sales x item x date_dim; d_moy=11, i_manufact_id=128;
//   group by d_year,i_brand_id,i_brand sum(ss_ext_sales_price); order by d_year, sum desc, brand_id; limit 100.
#include "build_plan.hpp"

pb::Rel build_plan(pb::Ctx const& c)
{
  using namespace pb;

  auto date_dim = filter(scan(c, "date_dim", {"d_date_sk", "d_year", "d_moy"}),
                         eq(col("d_moy"), lit<int64_t>(11)));

  auto item = filter(scan(c, "item", {"i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"}),
                     eq(col("i_manufact_id"), lit<int64_t>(128)));

  auto store_sales = scan(c, "store_sales", {"ss_item_sk", "ss_sold_date_sk", "ss_ext_sales_price"});

  // store_sales JOIN item ON ss_item_sk = i_item_sk
  auto j1 = join(store_sales, item, "ss_item_sk", "i_item_sk", gqe::join_type_type::inner,
                 {"ss_sold_date_sk", "ss_ext_sales_price", "i_brand_id", "i_brand"});

  // ... JOIN date_dim ON ss_sold_date_sk = d_date_sk
  auto j2 = join(j1, date_dim, "ss_sold_date_sk", "d_date_sk", gqe::join_type_type::inner,
                 {"ss_ext_sales_price", "i_brand_id", "i_brand", "d_year"});

  auto g = aggregate(j2, {"d_year", "i_brand_id", "i_brand"},
                     {{cudf::aggregation::SUM, col("ss_ext_sales_price"), "sum_agg"}});

  auto s = sort(g, {{"d_year", cudf::order::ASCENDING},
                    {"sum_agg", cudf::order::DESCENDING},
                    {"i_brand_id", cudf::order::ASCENDING}});

  return limit(s, 100);
}
