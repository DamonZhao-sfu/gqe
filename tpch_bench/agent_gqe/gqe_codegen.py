#!/usr/bin/env python3
"""GQE-style GPU codegen agent: a local (vLLM) LLM writes a TPC-DS query plan with the name-based
DSL (plan_builder.hpp); it is compiled into the GQE engine, run, and checked vs a DuckDB reference,
with errors fed back so the model retries.

    generate build_plan_gen.cpp -> ninja gqe_codegen_query -> run -> diff vs DuckDB reference
              ^                                                                  |
              +----------------------- feed back compile/run/diff error ---------+

Prereqs:
    - vLLM server (e.g. `vllm serve <model> --port 8000`)
    - GQE built once (build/benchmark exists) -- e.g. ./tpch_bench/gqe/build_gqe_v2.sh
    - TPC-DS parquet data: python tpch_bench/common/gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1
    - python duckdb in the gqe env.

Usage (start with already-implemented queries: 3, 7, 43):
    python tpch_bench/agent_gqe/gqe_codegen.py --query 3 --data /data/tpcds_sf1
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GQE_SRC = HERE.parent.parent
HARNESS = HERE / "harness"
GEN_FILE = HARNESS / "build_plan_gen.cpp"
FEWSHOT = HERE / "fewshot"
CMP_PY = GQE_SRC / "tpch_bench" / "common" / "compare_parquet.py"


def http_json(url, payload=None, method="GET", timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def pick_model(endpoint, model):
    if model:
        return model
    try:
        return http_json(f"{endpoint}/models")["data"][0]["id"]
    except Exception as e:
        sys.exit(f"--model not given and could not query {endpoint}/models ({e})")


def chat(endpoint, model, messages, temperature):
    r = http_json(f"{endpoint}/chat/completions",
                  {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": 4096}, method="POST")
    return r["choices"][0]["message"]["content"]


def extract_cpp(text):
    m = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


# --------------------------------------------------------------------------- duckdb reference
def duck(data):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds;")
    tables = []
    for t in ["call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
              "customer_address", "customer_demographics", "date_dim", "household_demographics",
              "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
              "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
              "web_sales", "web_site"]:
        files = sorted((Path(data) / t).glob("*.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet({[str(f) for f in files]})")
            tables.append(t)
    return con, tables


def get_sql(con, n):
    row = con.execute("SELECT query FROM tpcds_queries() WHERE query_nr = ?", [n]).fetchone()
    if not row:
        sys.exit(f"no TPC-DS query #{n}")
    return row[0].strip().rstrip(";")


def schemas_for(con, tables, sql):
    out = []
    for t in tables:
        if re.search(rf"\b{t}\b", sql):
            cols = con.execute(f"DESCRIBE {t}").fetchall()
            out.append(f"{t}(" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")")
    return "\n".join(out)


# --------------------------------------------------------------------------- build + run
def nvcomp_ld(env):
    dirs = set()
    for so in (GQE_SRC / "build").rglob("libnvcomp*.so*"):
        dirs.add(str(so.parent))
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(list(dirs) + [env.get("LD_LIBRARY_PATH", "")])
    return env


def build():
    bdir = GQE_SRC / "build"
    # ensure the target exists (it is guarded by EXISTS in CMake; reconfigure once to pick it up)
    if not (bdir / "build.ninja").exists() and not (bdir / "Makefile").exists():
        return False, "GQE build dir not configured; run build_gqe_v2.sh first"
    rc = subprocess.run(["cmake", str(bdir)], capture_output=True, text=True)  # reconfigure
    p = subprocess.run(["cmake", "--build", str(bdir), "--target", "gqe_codegen_query",
                        "-j", str(os.cpu_count() or 4)], capture_output=True, text=True)
    return p.returncode == 0, (rc.stdout + rc.stderr + p.stdout + p.stderr)


def run(workdir, data):
    binp = GQE_SRC / "build" / "benchmark" / "gqe_codegen_query"
    if not binp.exists():
        return False, f"binary not found: {binp}"
    env = nvcomp_ld(dict(os.environ))
    env["GQE_LOG_LEVEL"] = env.get("GQE_LOG_LEVEL", "warn")
    p = subprocess.run([str(binp), str(data)], cwd=str(workdir), capture_output=True, text=True, env=env)
    return p.returncode == 0, p.stdout + p.stderr


def compare(out, ref):
    p = subprocess.run([sys.executable, str(CMP_PY), str(out), str(ref)], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


# --------------------------------------------------------------------------- prompt
DSL_CHEATSHEET = """You implement ONE function using the name-based plan DSL (namespace pb):

    pb::Rel build_plan(pb::Ctx const& c);

Relations (each carries column names; reference columns BY NAME):
  scan(c, "table", {"col1","col2",...})                 -> read those columns
  filter(rel, <expr>, {keep...})                        -> WHERE; keep defaults to all columns
  project(rel, {{"out", <expr>}, ...})                  -> compute/rename columns
  join(l, r, "left_key", "right_key", gqe::join_type_type::inner, {keep...})
        keep names resolve over [l.cols ++ r.cols]; defaults to all of both
  aggregate(rel, {"key1",...}, { {cudf::aggregation::SUM, <expr>, "out"}, ... })
        output columns = keys first, then the measures
  sort(rel, {{"col", cudf::order::ASCENDING}, {"c2", cudf::order::DESCENDING}})
  limit(rel, count, offset=0)

Expressions:
  col("name"), lit<int64_t>(11), lit<double>(-5), lit<std::string>("M"), lit<double>(0,true)=NULL
  eq ne lt gt le ge  and_ or_  add sub mul div  if_else(cond, then, else)  cast(e, type)
  join types: inner, left, left_semi, left_anti, full

Rules:
- Reference every column by its name; track which columns survive each step (use `keep` to drop
  columns you no longer need, especially join keys).
- aggregate kinds: SUM, MEAN(=AVG), MIN, MAX, COUNT_ALL, COUNT_VALID.
- Output ONLY one complete build_plan_gen.cpp inside a single ```cpp block: it must
  #include "build_plan.hpp" and define pb::Rel build_plan(pb::Ctx const& c). No prose.
"""


def system_prompt():
    examples = []
    for f in [GEN_FILE, FEWSHOT / "q7_plan.cpp", FEWSHOT / "q43_plan.cpp"]:
        if f.exists():
            examples.append(f"// Example ({f.name}):\n{f.read_text()}")
    return DSL_CHEATSHEET + "\n\nWorked examples:\n\n" + "\n\n".join(examples)


def user_prompt(n, sql, schemas):
    return (f"Implement TPC-DS query {n} as build_plan_gen.cpp using the DSL.\n\nSQL:\n{sql}\n\n"
            f"Tables/columns available (use scan/col with these names):\n{schemas}\n\n"
            f"Write build_plan_gen.cpp now.")


def tail(s, n=2000):
    return s[-n:] if len(s) > n else s


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workdir", default=str(HERE / "agent_work"))
    args = ap.parse_args()

    model = pick_model(args.endpoint, args.model)
    print(f"[agent] endpoint={args.endpoint} model={model} TPC-DS Q{args.query}")

    con, tables = duck(args.data)
    sql = get_sql(con, args.query)
    schemas = schemas_for(con, tables, sql)
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    ref = work / f"reference_q{args.query}.parquet"
    print("[agent] computing DuckDB reference ...")
    con.execute(f"COPY ({sql}) TO '{ref}' (FORMAT parquet)")

    # keep a backup of the default (q3) seed so we can restore it after the run
    backup = GEN_FILE.read_text() if GEN_FILE.exists() else None

    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(args.query, sql, schemas)}]
    try:
        for it in range(1, args.max_iters + 1):
            print(f"\n===== iteration {it}/{args.max_iters} =====")
            code = extract_cpp(chat(args.endpoint, model, messages, args.temperature))
            GEN_FILE.write_text(code)
            (work / f"iter{it}_build_plan.cpp").write_text(code)  # keep every attempt
            messages.append({"role": "assistant", "content": f"```cpp\n{code}\n```"})

            ok, log = build()
            if not ok:
                (work / f"iter{it}_build.log").write_text(log)
                print("[agent] BUILD FAILED. Tail of compiler output:")
                print("\n".join(log.splitlines()[-25:]))
                print(f"[agent] full log: {work / f'iter{it}_build.log'}  code: {work / f'iter{it}_build_plan.cpp'}")
                messages.append({"role": "user", "content": "Compilation failed. Fix it. Errors:\n" + tail(log)})
                continue
            ok, log = run(work, args.data)
            if not ok:
                (work / f"iter{it}_run.log").write_text(log)
                print("[agent] RUNTIME FAILED:")
                print("\n".join(log.splitlines()[-15:]))
                messages.append({"role": "user", "content": "Compiled but crashed. Fix it. Output:\n" + tail(log)})
                continue
            ok, detail = compare(work / "output.parquet", ref)
            print("[agent] result:", detail)
            if ok:
                sol = HERE / f"solution_q{args.query}.cpp"
                sol.write_text(code)
                print(f"\n✅ SUCCESS iter {it}. Solution saved: {sol}")
                return 0
            messages.append({"role": "user", "content": "Ran but wrong result vs SQL reference. Fix the plan. Diff:\n" + tail(detail)})
        last = work / f"last_attempt_q{args.query}.cpp"
        last.write_text(GEN_FILE.read_text())
        print(f"\n❌ did not converge in {args.max_iters} iters.")
        print(f"   all attempts + logs: {work}/iter*  | last attempt: {last}")
        print("   TIP: first confirm the harness itself compiles with the default Q3:")
        print("        git checkout tpch_bench/agent_gqe/harness/build_plan_gen.cpp && \\")
        print("        ./tpch_bench/gqe/build_query.sh gqe_codegen_query")
        return 1
    finally:
        if backup is not None:
            GEN_FILE.write_text(backup)  # restore the committed q3 default


if __name__ == "__main__":
    raise SystemExit(main())
