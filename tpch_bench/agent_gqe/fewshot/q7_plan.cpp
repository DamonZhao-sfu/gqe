// SPDX-License-Identifier: Apache-2.0
// Few-shot example: TPC-DS Q7 in the name-based DSL. Four-way star join + AVG group-by.
// (Filter-only dimensions keep just their join key.)
#include "build_plan.hpp"

pb::Rel build_plan(pb::Ctx const& c)
{
  using namespace pb;

  auto date_dim = filter(scan(c, "date_dim", {"d_date_sk", "d_year"}),
                         eq(col("d_year"), lit<int64_t>(2000)), {"d_date_sk"});

  auto cd = filter(
    scan(c, "customer_demographics",
         {"cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"}),
    and_(eq(col("cd_gender"), lit<std::string>("M")),
         and_(eq(col("cd_marital_status"), lit<std::string>("S")),
              eq(col("cd_education_status"), lit<std::string>("College")))),
    {"cd_demo_sk"});

  auto promo = filter(scan(c, "promotion", {"p_promo_sk", "p_channel_email", "p_channel_event"}),
                      or_(eq(col("p_channel_email"), lit<std::string>("N")),
                          eq(col("p_channel_event"), lit<std::string>("N"))),
                      {"p_promo_sk"});

  auto item = scan(c, "item", {"i_item_sk", "i_item_id"});

  auto store_sales = scan(c, "store_sales",
                          {"ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price",
                           "ss_sold_date_sk", "ss_item_sk", "ss_cdemo_sk", "ss_promo_sk"});

  auto j1 = join(store_sales, cd, "ss_cdemo_sk", "cd_demo_sk", gqe::join_type_type::inner,
                 {"ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price",
                  "ss_sold_date_sk", "ss_item_sk", "ss_promo_sk"});
  auto j2 = join(j1, date_dim, "ss_sold_date_sk", "d_date_sk", gqe::join_type_type::inner,
                 {"ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "ss_item_sk",
                  "ss_promo_sk"});
  auto j3 = join(j2, promo, "ss_promo_sk", "p_promo_sk", gqe::join_type_type::inner,
                 {"ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "ss_item_sk"});
  auto j4 = join(j3, item, "ss_item_sk", "i_item_sk", gqe::join_type_type::inner,
                 {"ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "i_item_id"});

  auto g = aggregate(j4, {"i_item_id"},
                     {{cudf::aggregation::MEAN, col("ss_quantity"), "agg1"},
                      {cudf::aggregation::MEAN, col("ss_list_price"), "agg2"},
                      {cudf::aggregation::MEAN, col("ss_coupon_amt"), "agg3"},
                      {cudf::aggregation::MEAN, col("ss_sales_price"), "agg4"}});

  auto s = sort(g, {{"i_item_id", cudf::order::ASCENDING}});
  return limit(s, 100);
}
