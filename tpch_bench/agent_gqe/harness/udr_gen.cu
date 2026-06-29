// SPDX-License-Identifier: Apache-2.0
// Placeholder for the UDR (custom fused-kernel) codegen target. The agent (udr_codegen.py)
// OVERWRITES this with a complete standalone program like benchmark/hardcoded/q3_udr.cu, then
// restores this placeholder. Kept tiny so the normal build compiles it instantly.
#include <iostream>
int main(int /*argc*/, char** /*argv*/)
{
  std::cerr << "udr_gen placeholder. Run tpch_bench/agent_gqe/udr_codegen.py to generate a kernel.\n";
  return 2;
}
