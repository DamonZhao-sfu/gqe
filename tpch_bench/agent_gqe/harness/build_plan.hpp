// SPDX-License-Identifier: Apache-2.0
// The single function the code generator implements (in build_plan_gen.cpp), using the name-based
// DSL in plan_builder.hpp. `ctx.cat` is a catalog with all present TPC-DS tables registered.
#pragma once

#include "plan_builder.hpp"

// Return the root relation of the query plan. Its `cols` are the final output column names.
pb::Rel build_plan(pb::Ctx const& ctx);
