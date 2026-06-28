// SPDX-License-Identifier: Apache-2.0
// Fixed harness: load the TPC-H parquet tables present in <dataset_dir>, call the agent-generated
// run_query(), and write the result to output.parquet. The agent never edits this file.
#include "run_query.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<std::string> parquet_files(fs::path const& dir)
{
  std::vector<std::string> files;
  if (!fs::is_directory(dir)) return files;
  for (auto const& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".parquet") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  return files;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: agent_query <tpch_dataset_dir>\n";
    return 2;
  }
  fs::path const data = argv[1];

  // Standard TPC-H tables; only those present in the dataset dir are loaded.
  static constexpr char const* kTables[] = {"region",   "nation", "supplier", "customer",
                                            "part",     "partsupp", "orders",  "lineitem"};

  std::vector<std::unique_ptr<cudf::table>> owners;  // keep tables alive while views are used
  std::map<std::string, Table> tables;

  for (auto const* name : kTables) {
    auto files = parquet_files(data / name);
    if (files.empty()) continue;
    auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{files}).build();
    auto res  = cudf::io::read_parquet(opts);
    std::vector<std::string> names;
    names.reserve(res.metadata.schema_info.size());
    for (auto const& s : res.metadata.schema_info) names.push_back(s.name);
    owners.push_back(std::move(res.tbl));
    tables[name] = Table{owners.back()->view(), std::move(names)};
  }

  if (tables.empty()) {
    std::cerr << "ERROR: no TPC-H parquet tables found under " << data << "\n";
    return 2;
  }

  auto result = run_query(tables);

  auto sink = cudf::io::sink_info("output.parquet");
  auto wopt = cudf::io::parquet_writer_options::builder(sink, result->view()).build();
  cudf::io::write_parquet(wopt);
  std::cerr << "wrote output.parquet (" << result->num_rows() << " rows, "
            << result->num_columns() << " cols)\n";
  return 0;
}
