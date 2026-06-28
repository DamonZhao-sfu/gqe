// SPDX-License-Identifier: Apache-2.0
//
// Name-based plan-building DSL over GQE's logical API.
//
// GQE's logical relations reference columns by 0-based INDEX, and every filter/join/project
// reshuffles columns -- which is the hardest thing for a code generator to track. This thin layer
// lets the generator build a plan using COLUMN NAMES; it resolves names->indices and tracks the
// running column layout for you.
//
// NOTE: not compile-verified in this environment; expect to iterate on your GPU box.
#pragma once

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/types.hpp>

#include <cudf/aggregation.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pb {

// An Expr is a thunk: given the current relation's column names, build a gqe expression with
// indices resolved by name. (Lets the same expression be materialized against any context.)
using Expr = std::function<std::unique_ptr<gqe::expression>(std::vector<std::string> const&)>;

inline int index_of(std::vector<std::string> const& cols, std::string const& name)
{
  auto it = std::find(cols.begin(), cols.end(), name);
  if (it == cols.end()) throw std::runtime_error("plan_builder: unknown column '" + name + "'");
  return static_cast<int>(std::distance(cols.begin(), it));
}

// ----------------------------------------------------------------- expression builders (by name)
inline Expr col(std::string name)
{
  return [name = std::move(name)](std::vector<std::string> const& ctx) {
    return std::make_unique<gqe::column_reference_expression>(
      static_cast<cudf::size_type>(index_of(ctx, name)));
  };
}
template <typename T>
inline Expr lit(T value, bool is_null = false)
{
  return [value, is_null](std::vector<std::string> const&) -> std::unique_ptr<gqe::expression> {
    return std::make_unique<gqe::literal_expression<T>>(value, is_null);
  };
}

// Generic binary builder; concrete ops below.
template <typename ExprT>
inline Expr binop(Expr a, Expr b)
{
  return [a = std::move(a), b = std::move(b)](
           std::vector<std::string> const& ctx) -> std::unique_ptr<gqe::expression> {
    std::shared_ptr<gqe::expression> l = a(ctx);
    std::shared_ptr<gqe::expression> r = b(ctx);
    return std::make_unique<ExprT>(l, r);
  };
}
inline Expr eq(Expr a, Expr b) { return binop<gqe::equal_expression>(std::move(a), std::move(b)); }
inline Expr ne(Expr a, Expr b) { return binop<gqe::not_equal_expression>(std::move(a), std::move(b)); }
inline Expr lt(Expr a, Expr b) { return binop<gqe::less_expression>(std::move(a), std::move(b)); }
inline Expr gt(Expr a, Expr b) { return binop<gqe::greater_expression>(std::move(a), std::move(b)); }
inline Expr le(Expr a, Expr b) { return binop<gqe::less_equal_expression>(std::move(a), std::move(b)); }
inline Expr ge(Expr a, Expr b) { return binop<gqe::greater_equal_expression>(std::move(a), std::move(b)); }
inline Expr and_(Expr a, Expr b) { return binop<gqe::logical_and_expression>(std::move(a), std::move(b)); }
inline Expr or_(Expr a, Expr b) { return binop<gqe::logical_or_expression>(std::move(a), std::move(b)); }
inline Expr add(Expr a, Expr b) { return binop<gqe::add_expression>(std::move(a), std::move(b)); }
inline Expr sub(Expr a, Expr b) { return binop<gqe::subtract_expression>(std::move(a), std::move(b)); }
inline Expr mul(Expr a, Expr b) { return binop<gqe::multiply_expression>(std::move(a), std::move(b)); }
inline Expr div(Expr a, Expr b) { return binop<gqe::divide_expression>(std::move(a), std::move(b)); }

inline Expr if_else(Expr c, Expr t, Expr e)
{
  return [c = std::move(c), t = std::move(t), e = std::move(e)](
           std::vector<std::string> const& ctx) -> std::unique_ptr<gqe::expression> {
    std::shared_ptr<gqe::expression> ce = c(ctx), te = t(ctx), ee = e(ctx);
    return std::make_unique<gqe::if_then_else_expression>(ce, te, ee);
  };
}
inline Expr cast(Expr a, cudf::data_type ty)
{
  return [a = std::move(a), ty](std::vector<std::string> const& ctx) -> std::unique_ptr<gqe::expression> {
    std::shared_ptr<gqe::expression> ae = a(ctx);
    return std::make_unique<gqe::cast_expression>(ae, ty);
  };
}

// ----------------------------------------------------------------- relations (carry column names)
struct Ctx {
  gqe::catalog const* cat;
};

struct Rel {
  std::shared_ptr<gqe::logical::relation> node;
  std::vector<std::string> cols;  // current output column names, in order
};

struct Agg {
  cudf::aggregation::Kind kind;
  Expr expr;
  std::string name;  // output column name
};

// SELECT <columns> FROM <table>
inline Rel scan(Ctx const& c, std::string const& table, std::vector<std::string> columns)
{
  std::vector<cudf::data_type> types;
  types.reserve(columns.size());
  for (auto const& col_name : columns) types.push_back(c.cat->column_type(table, col_name));
  auto node = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{}, columns, std::move(types), table,
    std::unique_ptr<gqe::expression>{});
  return Rel{std::move(node), std::move(columns)};
}

// WHERE <cond>, optionally projecting to `keep` (defaults to all current columns).
inline Rel filter(Rel in, Expr cond, std::vector<std::string> keep = {})
{
  if (keep.empty()) keep = in.cols;
  std::vector<cudf::size_type> proj;
  for (auto const& k : keep) proj.push_back(static_cast<cudf::size_type>(index_of(in.cols, k)));
  auto node = std::make_shared<gqe::logical::filter_relation>(
    in.node, std::vector<std::shared_ptr<gqe::logical::relation>>{}, cond(in.cols), std::move(proj));
  return Rel{std::move(node), std::move(keep)};
}

// SELECT <name = expr, ...> (compute/rename columns).
inline Rel project(Rel in, std::vector<std::pair<std::string, Expr>> exprs)
{
  std::vector<std::unique_ptr<gqe::expression>> outs;
  std::vector<std::string> names;
  for (auto& [name, e] : exprs) {
    outs.push_back(e(in.cols));
    names.push_back(name);
  }
  auto node = std::make_shared<gqe::logical::project_relation>(
    in.node, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outs));
  return Rel{std::move(node), std::move(names)};
}

// l JOIN r ON l.<left_key> = r.<right_key>. Output = `keep` names (resolved over [l.cols, r.cols]);
// defaults to all of l's columns followed by all of r's columns.
inline Rel join(Rel l, Rel r, std::string const& left_key, std::string const& right_key,
                gqe::join_type_type type, std::vector<std::string> keep = {})
{
  std::vector<std::string> concat = l.cols;
  concat.insert(concat.end(), r.cols.begin(), r.cols.end());

  auto const li = index_of(l.cols, left_key);
  auto const ri = static_cast<int>(l.cols.size()) + index_of(r.cols, right_key);
  std::shared_ptr<gqe::expression> a =
    std::make_shared<gqe::column_reference_expression>(static_cast<cudf::size_type>(li));
  std::shared_ptr<gqe::expression> b =
    std::make_shared<gqe::column_reference_expression>(static_cast<cudf::size_type>(ri));
  auto cond = std::make_unique<gqe::equal_expression>(a, b);

  if (keep.empty()) keep = concat;
  std::vector<cudf::size_type> proj;
  for (auto const& k : keep) proj.push_back(static_cast<cudf::size_type>(index_of(concat, k)));

  auto node = std::make_shared<gqe::logical::join_relation>(
    l.node, r.node, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(cond), type,
    std::move(proj));
  return Rel{std::move(node), std::move(keep)};
}

// GROUP BY <keys> producing keys followed by the aggregate measures (gqe groupby emits keys first).
inline Rel aggregate(Rel in, std::vector<std::string> keys, std::vector<Agg> measures)
{
  std::vector<std::unique_ptr<gqe::expression>> key_exprs;
  for (auto const& k : keys)
    key_exprs.push_back(std::make_unique<gqe::column_reference_expression>(
      static_cast<cudf::size_type>(index_of(in.cols, k))));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> ms;
  std::vector<std::string> out = keys;
  for (auto& m : measures) {
    ms.emplace_back(m.kind, m.expr(in.cols));
    out.push_back(m.name);
  }
  auto node = std::make_shared<gqe::logical::aggregate_relation>(
    in.node, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(key_exprs),
    std::move(ms));
  return Rel{std::move(node), std::move(out)};
}

// ORDER BY <col asc/desc, ...> (NULLS FIRST).
inline Rel sort(Rel in, std::vector<std::pair<std::string, cudf::order>> by)
{
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> nulls;
  for (auto const& [name, ord] : by) {
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(
      static_cast<cudf::size_type>(index_of(in.cols, name))));
    orders.push_back(ord);
    nulls.push_back(cudf::null_order::BEFORE);
  }
  auto node = std::make_shared<gqe::logical::sort_relation>(
    in.node, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(orders),
    std::move(nulls), std::move(exprs));
  return Rel{std::move(node), in.cols};
}

// LIMIT <count> OFFSET <offset>
inline Rel limit(Rel in, int64_t count, int64_t offset = 0)
{
  auto node = std::make_shared<gqe::logical::fetch_relation>(in.node, offset, count);
  return Rel{std::move(node), in.cols};
}

}  // namespace pb
