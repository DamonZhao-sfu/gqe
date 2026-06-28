// SPDX-License-Identifier: Apache-2.0
// Few-shot example: TPC-DS Q43 in the name-based DSL. Two-way star join + conditional SUMs (CASE WHEN).
#include "build_plan.hpp"

pb::Rel build_plan(pb::Ctx const& c)
{
  using namespace pb;

  auto date_dim = filter(scan(c, "date_dim", {"d_date_sk", "d_day_name", "d_year"}),
                         eq(col("d_year"), lit<int64_t>(2000)), {"d_date_sk", "d_day_name"});

  auto store = filter(scan(c, "store", {"s_store_sk", "s_store_id", "s_store_name", "s_gmt_offset"}),
                      eq(col("s_gmt_offset"), lit<double>(-5)),
                      {"s_store_sk", "s_store_id", "s_store_name"});

  auto store_sales = scan(c, "store_sales", {"ss_sold_date_sk", "ss_store_sk", "ss_sales_price"});

  auto j1 = join(store_sales, date_dim, "ss_sold_date_sk", "d_date_sk", gqe::join_type_type::inner,
                 {"ss_store_sk", "ss_sales_price", "d_day_name"});
  auto j2 = join(j1, store, "ss_store_sk", "s_store_sk", gqe::join_type_type::inner,
                 {"s_store_name", "s_store_id", "d_day_name", "ss_sales_price"});

  // sum(case when d_day_name = <day> then ss_sales_price else null end) per day-of-week
  auto day_sum = [&](char const* day, char const* name) -> Agg {
    return Agg{cudf::aggregation::SUM,
               if_else(eq(col("d_day_name"), lit<std::string>(day)), col("ss_sales_price"),
                       lit<double>(0, /*is_null=*/true)),
               name};
  };

  auto g = aggregate(j2, {"s_store_name", "s_store_id"},
                     {day_sum("Sunday", "sun_sales"), day_sum("Monday", "mon_sales"),
                      day_sum("Tuesday", "tue_sales"), day_sum("Wednesday", "wed_sales"),
                      day_sum("Thursday", "thu_sales"), day_sum("Friday", "fri_sales"),
                      day_sum("Saturday", "sat_sales")});

  auto s = sort(g, {{"s_store_name", cudf::order::ASCENDING},
                    {"s_store_id", cudf::order::ASCENDING},
                    {"sun_sales", cudf::order::ASCENDING},
                    {"mon_sales", cudf::order::ASCENDING},
                    {"tue_sales", cudf::order::ASCENDING},
                    {"wed_sales", cudf::order::ASCENDING},
                    {"thu_sales", cudf::order::ASCENDING},
                    {"fri_sales", cudf::order::ASCENDING},
                    {"sat_sales", cudf::order::ASCENDING}});
  return limit(s, 100);
}
