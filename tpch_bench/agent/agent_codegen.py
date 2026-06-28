#!/usr/bin/env python3
"""A tiny GPU-codegen Agent: a local (vLLM-served) LLM writes a libcudf TPC-H query, which is then
compiled, run, and checked against a DuckDB reference -- with errors fed back so the model retries.

Agentic loop:
    generate run_query.cpp  ->  cmake build  ->  run on dataset  ->  diff vs DuckDB reference
                ^                                                              |
                +------------------ feed back compile/run/diff error ---------+

Why this shape (see tpch_bench discussion): the baseline is "express the query as libcudf operator
calls" (cudf NDS-H style). A fixed C++ harness provides Parquet I/O + main + table loading, so the
model only fills one function -- which keeps compiles reliable and the action space small.

Prereqs:
    - a vLLM server, e.g.:  vllm serve <model> --port 8000
    - conda env with libcudf installed (build_gqe_v2.sh installs it) + a TPC-H parquet dataset:
          python tpch_bench/common/gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1
    - python: duckdb (in the gqe env). No other pip deps (LLM call uses urllib).

Usage:
    python agent_codegen.py --query 6 --data /data/tpch_sf1
    python agent_codegen.py --query 1 --data /data/tpch_sf1 --endpoint http://localhost:8000/v1
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
HARNESS = HERE / "harness"
CMP_PY = HERE.parent / "common" / "compare_parquet.py"


# ----------------------------------------------------------------------------- LLM (vLLM / OpenAI)
def http_json(url, payload=None, method="GET", timeout=600):
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
        data = http_json(f"{endpoint}/models")
        return data["data"][0]["id"]
    except Exception as e:
        sys.exit(f"--model not given and could not query {endpoint}/models ({e})")


def chat(endpoint, model, messages, temperature):
    resp = http_json(f"{endpoint}/chat/completions", {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": 4096,
    }, method="POST")
    return resp["choices"][0]["message"]["content"]


def extract_cpp(text):
    m = re.search(r"```(?:cpp|c\+\+|cuda)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


# ----------------------------------------------------------------------------- DuckDB reference
def duckdb_setup(data):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    present = []
    for t in ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]:
        files = sorted((Path(data) / t).glob("*.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet({[str(f) for f in files]})")
            present.append(t)
    return con, present


def query_sql(con, n):
    row = con.execute("SELECT query FROM tpch_queries() WHERE query_nr = ?", [n]).fetchone()
    if not row:
        sys.exit(f"no TPC-H query #{n}")
    # one statement only; strip trailing ';'
    return row[0].strip().rstrip(";")


def schema_text(con, tables):
    lines = []
    for t in tables:
        cols = con.execute(f"DESCRIBE {t}").fetchall()  # (name, type, ...)
        lines.append(f"{t}(" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")")
    return "\n".join(lines)


def write_reference(con, sql, out):
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet)")


# ----------------------------------------------------------------------------- build + run + check
def prepare_workdir(workdir):
    src = workdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    for f in ["main.cpp", "run_query.hpp", "CMakeLists.txt"]:
        shutil.copy(HARNESS / f, src / f)
    return src


def configure(src):
    build = src / "build"
    env = dict(os.environ)
    prefix = env.get("CONDA_PREFIX", "")
    cmd = ["cmake", "-S", str(src), "-B", str(build),
           "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=native"]
    if prefix:
        cmd.append(f"-DCMAKE_PREFIX_PATH={prefix}")
    if shutil.which("ninja"):
        cmd += ["-G", "Ninja"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return build, p.returncode == 0, p.stdout + p.stderr


def build(build_dir):
    p = subprocess.run(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 4)],
                       capture_output=True, text=True)
    return p.returncode == 0, p.stdout + p.stderr


def run_binary(build_dir, workdir, data):
    binp = build_dir / "agent_query"
    p = subprocess.run([str(binp), str(data)], cwd=str(workdir), capture_output=True, text=True)
    return p.returncode == 0, p.stdout + p.stderr


def compare(out, ref):
    p = subprocess.run([sys.executable, str(CMP_PY), str(out), str(ref)],
                       capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip() + p.stderr.strip()


# ----------------------------------------------------------------------------- prompt
SYSTEM = """You are an expert NVIDIA libcudf (RAPIDS cuDF C++) engineer. You write GPU query code.

You implement exactly ONE function in run_query.cpp:

    std::unique_ptr<cudf::table> run_query(std::map<std::string, Table> const& tables);

`Table` (from run_query.hpp) is:
    struct Table { cudf::table_view view; std::vector<std::string> names;
                   cudf::column_view col(std::string const& name) const; };
Access input columns by name, e.g.  tables.at("lineitem").col("l_discount").

Rules:
- Use ONLY libcudf C++ APIs (cudf::ast, cudf::compute_column, cudf::binary_operation,
  cudf::apply_boolean_mask, cudf::groupby, cudf::reduce, cudf::sort_by_key, cudf::gather,
  cudf::table, cudf::scalar, etc.) plus rmm. No external libraries.
- Return the result as a cudf::table with one column per SELECT item, in SELECT order.
  For a scalar result, return a single-row, single-column table.
- Output ONLY a complete run_query.cpp inside ONE ```cpp code block: the necessary #includes
  (always #include "run_query.hpp"), any helper functions, and run_query(). No prose.
- Decimal columns in the data are DECIMAL/INT; cast as needed. Dates are TIMESTAMP_DAYS.
"""


def user_prompt(sql, schemas):
    return (f"Implement this TPC-H query with libcudf.\n\nSQL:\n{sql}\n\n"
            f"Available input tables and columns (load via tables.at(\"<table>\").col(\"<col>\")):\n"
            f"{schemas}\n\nWrite run_query.cpp now.")


# ----------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=int, required=True, help="TPC-H query number (1-22)")
    ap.add_argument("--data", required=True, help="TPC-H parquet dataset dir (<dir>/<table>/*.parquet)")
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workdir", default=str(HERE / "agent_work"))
    args = ap.parse_args()

    model = pick_model(args.endpoint, args.model)
    print(f"[agent] endpoint={args.endpoint} model={model} query=Q{args.query}")

    con, tables = duckdb_setup(args.data)
    sql = query_sql(con, args.query)
    schemas = schema_text(con, tables)
    workdir = Path(args.workdir); workdir.mkdir(parents=True, exist_ok=True)
    ref = workdir / "reference.parquet"
    print("[agent] computing DuckDB reference ...")
    write_reference(con, sql, str(ref))

    src = prepare_workdir(workdir)
    build_dir, ok, log = configure(src)
    if not ok:
        sys.exit(f"[agent] cmake configure failed (is libcudf installed?):\n{log[-2000:]}")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(sql, schemas)}]

    for it in range(1, args.max_iters + 1):
        print(f"\n===== iteration {it}/{args.max_iters} =====")
        code = extract_cpp(chat(args.endpoint, model, messages, args.temperature))
        (src / "run_query.cpp").write_text(code)
        messages.append({"role": "assistant", "content": f"```cpp\n{code}\n```"})

        ok, log = build(build_dir)
        if not ok:
            print("[agent] BUILD FAILED")
            messages.append({"role": "user",
                             "content": "Compilation failed. Fix it. Errors:\n" + tail(log)})
            continue

        ok, log = run_binary(build_dir, workdir, args.data)
        if not ok:
            print("[agent] RUNTIME FAILED")
            messages.append({"role": "user",
                             "content": "It compiled but crashed at runtime. Fix it. Output:\n" + tail(log)})
            continue

        ok, detail = compare(workdir / "output.parquet", ref)
        print("[agent] result check:", detail)
        if ok:
            print(f"\n✅ SUCCESS on iteration {it}. Solution: {src / 'run_query.cpp'}")
            print(f"   result: {workdir / 'output.parquet'}  reference: {ref}")
            return 0
        messages.append({"role": "user",
                         "content": "It ran but the result is wrong vs the SQL reference. Fix the "
                                    "query logic. Diff:\n" + tail(detail)})

    print(f"\n❌ did not converge in {args.max_iters} iterations. Last code: {src / 'run_query.cpp'}")
    return 1


def tail(s, n=1800):
    return s[-n:] if len(s) > n else s


if __name__ == "__main__":
    raise SystemExit(main())
