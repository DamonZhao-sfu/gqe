#!/usr/bin/env python3
"""UDR codegen agent: a local (vLLM) LLM writes a COMPLETE custom fused-kernel program
(like benchmark/hardcoded/q3_udr.cu) for a TPC-DS query, compiled into the GQE engine, run, and
checked vs a DuckDB reference, with errors fed back so it retries.

This is the "customize the kernel" path (cf. the plan-only path in gqe_codegen.py). The model emits a
whole standalone .cu: register tables, build a logical plan that uses a `user_defined_relation` with a
hand-written CUDA kernel for the hot join, then aggregate/sort/limit and write output.parquet.

Inputs (same as gqe_codegen.py): data files + schema (auto) + a SQL query.

Usage:
    python tpch_bench/agent_gqe/udr_codegen.py --query 3 --data /data/tpcds_sf1
    python tpch_bench/agent_gqe/udr_codegen.py --data /data/tpcds_sf1 --sql-file q.sql --name myq
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import gqe_codegen as g  # reuse: duck, resolve_sql, schemas_for, chat, pick_model, extract_cpp, nvcomp_ld, tail

HERE = Path(__file__).resolve().parent
GQE_SRC = HERE.parent.parent
UDR_GEN = HERE / "harness" / "udr_gen.cu"
FEWSHOT_UDR = [GQE_SRC / "benchmark" / "hardcoded" / f for f in ("q3_udr.cu", "q7_udr.cu", "q43_udr.cu")]
CMP_PY = GQE_SRC / "tpch_bench" / "common" / "compare_parquet.py"

SYSTEM = """You are an expert NVIDIA CUDA + GQE engineer. You write a COMPLETE standalone C++/CUDA
program (a .cu file) that runs ONE TPC-DS query on the GQE engine using a CUSTOM fused kernel for
the hot join, exactly in the style of the worked examples below (q3_udr.cu / q7_udr.cu / q43_udr.cu).

Your program MUST:
- have `int main(int argc, char* argv[])` taking argv[1] = dataset dir;
- set up the RMM pool, a task_manager_context and catalog, and register ONLY the tables this query
  needs (with the columns it needs), reading parquet via gqe::utility::get_parquet_files(dir+"/"+t);
- build the logical plan, using a `gqe::logical::user_defined_relation` whose custom_task runs a
  hand-written __global__ kernel (cuco hash maps + cub warp-scan + stream compaction, then
  cudf::gather) to fuse the multi-way join -- mirror the example structure closely;
- finish with aggregate/sort/fetch as the query requires, build+run the task graph
  (execute_task_graph_single_gpu), and write the result to "output.parquet".

Output ONLY the complete .cu inside ONE ```cpp code block (all #includes, kernels, main). No prose.
"""


def build():
    bdir = GQE_SRC / "build"
    if not (bdir / "build.ninja").exists() and not (bdir / "Makefile").exists():
        return False, "GQE build dir not configured; run build_gqe_v2.sh first"
    p = subprocess.run(["cmake", "--build", str(bdir), "--target", "udr_codegen_query",
                        "-j", str(os.cpu_count() or 4)], capture_output=True, text=True)
    return p.returncode == 0, p.stdout + p.stderr


def run(workdir, data):
    binp = GQE_SRC / "build" / "benchmark" / "udr_codegen_query"
    if not binp.exists():
        return False, f"binary not found: {binp}"
    env = g.nvcomp_ld(dict(os.environ))
    env["GQE_LOG_LEVEL"] = env.get("GQE_LOG_LEVEL", "warn")
    p = subprocess.run([str(binp), str(data)], cwd=str(workdir), capture_output=True, text=True, env=env)
    return p.returncode == 0, p.stdout + p.stderr


def system_prompt():
    ex = []
    for f in FEWSHOT_UDR:
        if f.exists():
            ex.append(f"// Worked example {f.name}:\n{f.read_text()}")
    return SYSTEM + "\n\nExamples:\n\n" + "\n\n".join(ex[:2])  # 2 examples keep the prompt reasonable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sql", default="")
    ap.add_argument("--sql-file", dest="sql_file", default="")
    ap.add_argument("--query", type=int)
    ap.add_argument("--tpch", type=int)
    ap.add_argument("--name", default="")
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workdir", default=str(HERE / "agent_work"))
    args = ap.parse_args()

    model = g.pick_model(args.endpoint, args.model)
    con, tables = g.duck(args.data)
    sql, label = g.resolve_sql(con, args)
    schemas = g.schemas_for(con, tables, sql)
    print(f"[udr] endpoint={args.endpoint} model={model} query={label}")

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    ref = work / f"reference_{label}.parquet"
    print("[udr] computing DuckDB reference ...")
    con.execute(f"COPY ({sql}) TO '{ref}' (FORMAT parquet)")

    backup = UDR_GEN.read_text() if UDR_GEN.exists() else None
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": f"Write a custom fused-kernel .cu for TPC-DS query "
                                             f"{label}.\n\nSQL:\n{sql}\n\nTables/columns:\n{schemas}\n\n"
                                             f"Write udr_gen.cu now."}]
    try:
        for it in range(1, args.max_iters + 1):
            print(f"\n===== iteration {it}/{args.max_iters} =====")
            code = g.extract_cpp(g.chat(args.endpoint, model, messages, args.temperature))
            UDR_GEN.write_text(code)
            (work / f"udr_iter{it}.cu").write_text(code)
            messages.append({"role": "assistant", "content": f"```cpp\n{code}\n```"})

            ok, log = build()
            if not ok:
                (work / f"udr_iter{it}.build.log").write_text(log)
                print("[udr] BUILD FAILED:\n" + "\n".join(log.splitlines()[-25:]))
                messages.append({"role": "user", "content": "Compilation failed. Fix it. Errors:\n" + g.tail(log)})
                continue
            ok, log = run(work, args.data)
            if not ok:
                print("[udr] RUNTIME FAILED:\n" + "\n".join(log.splitlines()[-15:]))
                messages.append({"role": "user", "content": "Compiled but crashed. Fix it. Output:\n" + g.tail(log)})
                continue
            p = subprocess.run([sys.executable, str(CMP_PY), str(work / "output.parquet"), str(ref)],
                               capture_output=True, text=True)
            print("[udr] result:", (p.stdout + p.stderr).strip())
            if p.returncode == 0:
                sol = HERE / f"solution_{label}_udr.cu"
                sol.write_text(code)
                print(f"\n✅ SUCCESS iter {it}. Kernel saved: {sol}")
                return 0
            messages.append({"role": "user", "content": "Ran but wrong result vs SQL reference. Fix it. Diff:\n" + g.tail((p.stdout + p.stderr).strip())})
        print(f"\n❌ did not converge in {args.max_iters} iters. Attempts in {work}/udr_iter*")
        return 1
    finally:
        if backup is not None:
            UDR_GEN.write_text(backup)  # restore the placeholder


if __name__ == "__main__":
    raise SystemExit(main())
