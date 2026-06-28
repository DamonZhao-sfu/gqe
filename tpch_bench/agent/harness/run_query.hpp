// SPDX-License-Identifier: Apache-2.0
// Fixed scaffold for the GPU-codegen agent. The LLM only implements run_query() in run_query.cpp;
// everything else (parquet I/O, main, table loading) is provided by the harness, which keeps the
// LLM's job small and the compile reliable.
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// A loaded input table plus its column names (cudf tables are positional, so we keep names here).
struct Table {
  cudf::table_view view;
  std::vector<std::string> names;

  // Access a column by its name, e.g. t.col("l_discount").
  [[nodiscard]] cudf::column_view col(std::string const& name) const
  {
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) throw std::runtime_error("no such column: " + name);
    return view.column(static_cast<cudf::size_type>(std::distance(names.begin(), it)));
  }
};

// THE FUNCTION THE AGENT IMPLEMENTS.
// Input: map from table name (e.g. "lineitem") to a loaded Table.
// Output: the query result as a cudf::table (one column per SELECT item, in SELECT order).
//         For a scalar result, return a single-row, single-column table.
std::unique_ptr<cudf::table> run_query(std::map<std::string, Table> const& tables);
