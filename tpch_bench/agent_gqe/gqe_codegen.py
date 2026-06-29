#!/usr/bin/env python3
"""GQE-style GPU codegen agent: a local (vLLM) LLM writes a TPC-DS query plan with the name-based
DSL (plan_builder.hpp); it is compiled into the GQE engine, run, and checked vs a DuckDB reference,
with errors fed back so the model retries.

    generate gen_query.cpp -> ninja gqe_codegen_query -> run -> diff vs DuckDB reference
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
GEN_FILE = HARNESS / "gen_query.cpp"   # the whole generated program (its own main)
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


def chat(endpoint, model, messages, temperature, max_tokens=8192, guided=False):
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens}
    if guided:
        # vLLM guided decoding: force the answer to be a single ```cpp ... ``` block.
        payload["guided_regex"] = r"```cpp\n[\s\S]+\n```"
    return http_json(f"{endpoint}/chat/completions", payload, method="POST")["choices"][0]["message"]["content"]


def extract_cpp(text):
    """Return ONLY the C++ source, even from reasoning models that emit prose / <think> blocks."""
    # 1) drop chain-of-thought wrappers some reasoning models emit
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.S | re.I)
    # 2) prefer fenced code blocks; pick the LARGEST (the full program, not a snippet)
    blocks = re.findall(r"```(?:cpp|c\+\+|cuda|c)?\s*(.*?)```", text, re.S | re.I)
    if blocks:
        return max(blocks, key=len).strip()
    # 3) unclosed fence (truncated output): take everything after the opening fence
    m = re.search(r"```(?:cpp|c\+\+|cuda|c)?\s*(.*)$", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    # 4) no fences at all: slice from the first #include to the end
    m = re.search(r"(#include[\s\S]*)$", text)
    return (m.group(1) if m else text).strip()


# --------------------------------------------------------------------------- inputs (schema + data)
# INPUTS, like BespokeOLAP / GenDB: (1) data files, (2) database schema, (3) SQL query.
# The schema is derived from the data files: every <data>/<name>/*.parquet becomes table <name>.
def duck(data):
    import duckdb
    con = duckdb.connect()
    for ext in ("tpcds", "tpch"):  # only needed for the --query/--tpch convenience SQL lookups
        try:
            con.execute(f"INSTALL {ext}; LOAD {ext};")
        except Exception:
            pass
    tables = []
    for sub in sorted(Path(data).iterdir() if Path(data).is_dir() else []):
        if not sub.is_dir():
            continue
        files = sorted(sub.glob("*.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {sub.name} AS SELECT * FROM read_parquet({[str(f) for f in files]})")
            tables.append(sub.name)
    if not tables:
        sys.exit(f"no <table>/*.parquet found under {data}")
    return con, tables


# Resolve the SQL query input: explicit --sql / --sql-file, or the --query/--tpch convenience lookups.
def resolve_sql(con, args):
    if args.sql:
        return args.sql.strip().rstrip(";"), (args.name or "query")
    if args.sql_file:
        return Path(args.sql_file).read_text().strip().rstrip(";"), (args.name or Path(args.sql_file).stem)
    if args.query is not None:
        row = con.execute("SELECT query FROM tpcds_queries() WHERE query_nr = ?", [args.query]).fetchone()
        if not row:
            sys.exit(f"no TPC-DS query #{args.query}")
        return row[0].strip().rstrip(";"), (args.name or f"q{args.query}")
    if args.tpch is not None:
        row = con.execute("SELECT query FROM tpch_queries() WHERE query_nr = ?", [args.tpch]).fetchone()
        if not row:
            sys.exit(f"no TPC-H query #{args.tpch}")
        return row[0].strip().rstrip(";"), (args.name or f"tpch_q{args.tpch}")
    sys.exit("provide a SQL query: --sql '<...>' | --sql-file <path> | --query <N> | --tpch <N>")


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
    if not (bdir / "build.ninja").exists() and not (bdir / "Makefile").exists():
        return False, "GQE build dir not configured; run build_gqe_v2.sh first"
    # Just build the target. Ninja auto-regenerates build.ninja if a CMakeLists changed (one-time);
    # we do NOT force a reconfigure every iteration -- that is slow and re-runs the thrift/Substrait
    # codegen steps each time. (We only rewrite gen_query.cpp, which never re-triggers configure.)
    p = subprocess.run(["cmake", "--build", str(bdir), "--target", "gqe_codegen_query",
                        "-j", str(os.cpu_count() or 4)], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr)


def run(workdir, data):
    binp = GQE_SRC / "build" / "benchmark" / "gqe_codegen_query"
    if not binp.exists():
        return False, f"binary not found: {binp}", None
    env = nvcomp_ld(dict(os.environ))
    env["GQE_LOG_LEVEL"] = "info"  # so the "Query execution time: N ms." line is emitted
    import time
    t0 = time.perf_counter()
    p = subprocess.run([str(binp), str(data)], cwd=str(workdir), capture_output=True, text=True, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    out = p.stdout + p.stderr
    m = re.search(r"Query execution time:\s*([0-9]+)\s*ms", out)
    gpu_ms = float(m.group(1)) if m else wall_ms  # engine time if logged, else wall clock
    return p.returncode == 0, out, gpu_ms


def compare(out, ref):
    p = subprocess.run([sys.executable, str(CMP_PY), str(out), str(ref)], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def table_files(data):
    d = {}
    for sub in sorted(p for p in Path(data).iterdir() if p.is_dir()):
        fs = sorted(sub.glob("*.parquet"))
        if fs:
            d[sub.name] = [str(f) for f in fs]
    return d


def run_sirius(home, data, sql, out, config="", gpu=""):
    """Run the same SQL on the open-source Sirius GPU DuckDB extension. Returns (ok, ms, log)."""
    import time
    home = Path(home)
    binp = home / "build" / "release" / "duckdb"
    if not binp.exists():
        return False, None, f"sirius duckdb not at {binp} (build: cd {home} && pixi run make)"
    # Sirius's own ./build/release/duckdb AUTO-LOADS the extension at startup (configured via
    # SIRIUS_CONFIG_FILE); a manual LOAD would re-register and throw "gpu_execution already exists".
    lines = []
    for name, files in table_files(data).items():
        lines.append(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({files});")
    lines += [".timer on", f"COPY ({sql}) TO '{out}' (FORMAT parquet);"]
    script = "\n".join(lines) + "\n"
    # Run with a clean library path so the conda (gqe) libs don't shadow Sirius's own (pixi) libs.
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    # Sirius reserves 95% of GPU memory at LOAD unless SIRIUS_CONFIG_FILE caps it -> set a config to
    # avoid OOM, and/or pin a free GPU (vLLM may occupy one).
    if config:
        env["SIRIUS_CONFIG_FILE"] = config
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    t0 = time.perf_counter()
    p = subprocess.run([str(binp), "-unsigned"], input=script, capture_output=True, text=True, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    m = re.search(r"real\s+([0-9.]+)", p.stdout + p.stderr)  # duckdb .timer: "Run Time (s): real X"
    ms = float(m.group(1)) * 1000.0 if m else wall_ms
    return (p.returncode == 0 and Path(out).exists()), ms, (p.stdout + p.stderr)


def show_parquet(path, label, n=10):
    try:
        import pandas as pd, pyarrow.parquet as pq
        df = pq.read_table(str(path)).to_pandas()
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\n----- {label}  ({df.shape[0]} rows x {df.shape[1]} cols) -----")
            print(df.head(n).to_string(index=False))
    except Exception as e:
        print(f"[show] could not read {path}: {e}")


# --------------------------------------------------------------------------- prompt
DSL_CHEATSHEET = """You write a COMPLETE standalone C++ program (its own main), exactly like the
worked example, that runs ONE TPC-DS query on the GQE engine using the name-based plan DSL.

Structure of the whole file (copy the example's skeleton):
  #include "gqe_runtime.hpp"   // gqe_pool_size(); register_tpcds OR register_tpch (cat,data); gqe_run_and_write(tm,cat,plan)
  #include "plan_builder.hpp"  // the DSL (namespace pb)
  #include <rmm/...>; <iostream>
  int main(int argc, char** argv) {
    std::string data = argv[1];
    // RMM pool, set_current_device_resource
    gqe::task_manager_context tm; gqe::catalog cat{&tm}; register_tpcds(cat, data);
    pb::Ctx c{&cat}; using namespace pb;
    /* build the plan with the DSL */
    pb::Rel plan = ...;
    gqe_run_and_write(tm, cat, plan);   // writes output.parquet
  }

DSL relations (each carries column names; reference columns BY NAME):
  scan(c, "table", {"col1",...})                  filter(rel, <expr>, {keep...})
  project(rel, {{"out", <expr>}, ...})            join(l, r, "lkey", "rkey", gqe::join_type_type::inner, {keep...})
  aggregate(rel, {"key",...}, {{cudf::aggregation::SUM, <expr>, "out"}, ...})  // keys first, then measures
  sort(rel, {{"col", cudf::order::ASCENDING}})    limit(rel, count, offset=0)
DSL expressions:
  col("name"), lit<int64_t>(11), lit<double>(-5), lit<std::string>("M"), lit<double>(0,true)=NULL
  eq ne lt gt le ge  and_ or_  add sub mul div  if_else(cond,then,else)  cast(e,type)
  join types: inner, left, left_semi, left_anti, full ; aggs: SUM, MEAN(=AVG), MIN, MAX, COUNT_ALL, COUNT_VALID

Rules:
- Reference columns by name; track which columns survive each step (use `keep` to drop columns you
  no longer need, especially join keys).
- Output ONLY the complete .cpp inside ONE ```cpp block (includes + main). No prose.
"""


def system_prompt():
    ex = (FEWSHOT / "q3_full.cpp")
    example = f"// Worked whole-file example (q3_full.cpp):\n{ex.read_text()}" if ex.exists() else ""
    return DSL_CHEATSHEET + "\n\n" + example


def user_prompt(n, sql, schemas, register_fn):
    note = ""
    if register_fn == "register_tpch":
        note = ("\nNOTE: This is TPC-H. In main call register_tpch(cat, data) (NOT register_tpcds). "
                "Date columns (e.g. l_shipdate) are TIMESTAMP_DAYS; build date literals with "
                "lit<cudf::timestamp_D>(cudf::timestamp_D{cudf::duration_D{days_since_epoch}}).")
    else:
        note = "\nNOTE: This is TPC-DS. In main call register_tpcds(cat, data)."
    return (f"Write the COMPLETE program (whole .cpp with main) for this query.\n\nSQL:\n{sql}\n\n"
            f"Tables/columns available (use scan/col with these names):\n{schemas}\n{note}\n\n"
            f"Write the whole .cpp now.")


def tail(s, n=2000):
    return s[-n:] if len(s) > n else s


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # INPUTS: data files (--data) + database schema (auto-derived from data) + a SQL query (one of):
    ap.add_argument("--data", required=True, help="data files dir: <data>/<table>/*.parquet")
    ap.add_argument("--sql", default="", help="SQL query text")
    ap.add_argument("--sql-file", dest="sql_file", default="", help="path to a .sql file")
    ap.add_argument("--query", type=int, help="convenience: TPC-DS query number (uses duckdb tpcds)")
    ap.add_argument("--tpch", type=int, help="convenience: TPC-H query number (uses duckdb tpch)")
    ap.add_argument("--name", default="", help="label for outputs (default derived from the source)")
    ap.add_argument("--ref-dir", dest="ref_dir", default="",
                    help="dir of precomputed DuckDB results (e.g. /.../tpcds_ref); compares vs <ref-dir>/q<N>.parquet")
    ap.add_argument("--ref-file", dest="ref_file", default="", help="explicit reference parquet to compare against")
    ap.add_argument("--sirius-home", dest="sirius_home", default=os.environ.get("SIRIUS_HOME", ""),
                    help="path to a built Sirius repo (github.com/sirius-db/sirius); also runs the query on Sirius GPU")
    ap.add_argument("--sirius-config", dest="sirius_config", default=os.environ.get("SIRIUS_CONFIG_FILE", ""),
                    help="Sirius YAML config (cap GPU memory; default reserves 95%% -> OOM)")
    ap.add_argument("--sirius-gpu", dest="sirius_gpu", default="",
                    help="GPU index for Sirius (CUDA_VISIBLE_DEVICES); use a GPU not busy with vLLM")
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--duckdb-threads", dest="duckdb_threads", type=int, default=0,
                    help="DuckDB CPU baseline threads (0=default/all cores; 1=single-thread like BespokeOLAP/GenDB)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="LLM max output tokens (raise for whole-file)")
    ap.add_argument("--guided", action="store_true",
                    help="force the LLM output to be a single ```cpp block via vLLM guided decoding")
    ap.add_argument("--workdir", default=str(HERE / "agent_work"))
    args = ap.parse_args()

    model = pick_model(args.endpoint, args.model)

    # Inputs: data files -> database schema (views); SQL query.
    con, tables = duck(args.data)
    if args.duckdb_threads and args.duckdb_threads > 0:
        con.execute(f"SET threads={args.duckdb_threads}")
    duckdb_threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    sql, label = resolve_sql(con, args)
    schemas = schemas_for(con, tables, sql)
    register_fn = "register_tpch" if args.tpch is not None else "register_tpcds"
    print(f"[agent] endpoint={args.endpoint} model={model} query={label} ({register_fn})")

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    # Persist the resolved inputs so the run is self-describing/reproducible.
    (work / f"{label}.input.sql").write_text(sql + "\n")
    (work / f"{label}.input.schema.txt").write_text(schemas + "\n")
    ref = work / f"reference_{label}.parquet"
    print("[agent] computing DuckDB reference ...")
    import time
    t0 = time.perf_counter()
    con.execute(f"SELECT count(*) FROM ({sql}) t").fetchall()  # CPU execution timing
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    con.execute(f"COPY ({sql}) TO '{ref}' (FORMAT parquet)")
    show_parquet(ref, "DuckDB CPU output")
    print(f"[time] DuckDB (CPU, threads={duckdb_threads}) query time: {cpu_ms:.1f} ms")

    # Optional external reference (e.g. a precomputed tpcds_ref dir).
    extra_ref = None
    if args.ref_file:
        extra_ref = Path(args.ref_file)
    elif args.ref_dir and (args.query is not None or args.tpch is not None):
        qn = args.query if args.query is not None else args.tpch
        cand = Path(args.ref_dir) / f"q{qn}.parquet"
        if cand.exists():
            extra_ref = cand
    if extra_ref and extra_ref.exists():
        print(f"[agent] will also compare against {extra_ref}")

    # Run the same query on Sirius (open-source GPU DuckDB extension) once, for a GPU baseline.
    sirius_ms = None
    if args.sirius_home:
        print("[agent] running Sirius (GPU) baseline ...")
        s_out = work / f"sirius_{label}.parquet"
        ok_s, sirius_ms, slog = run_sirius(args.sirius_home, args.data, sql, s_out,
                                           args.sirius_config, args.sirius_gpu)
        if ok_s:
            show_parquet(s_out, "Sirius (GPU) output")
            _, sd = compare(s_out, ref)
            print(f"[time] Sirius (GPU): {sirius_ms:.1f} ms   (vs DuckDB reference: {sd})")
        else:
            print("[agent] Sirius skipped/failed:", "\n".join(slog.splitlines()[-3:]))
            sirius_ms = None

    # keep a backup of the default (q3) seed so we can restore it after the run
    backup = GEN_FILE.read_text() if GEN_FILE.exists() else None

    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(label, sql, schemas, register_fn)}]
    try:
        for it in range(1, args.max_iters + 1):
            print(f"\n===== iteration {it}/{args.max_iters} =====")
            code = extract_cpp(chat(args.endpoint, model, messages, args.temperature,
                                    args.max_tokens, args.guided))
            GEN_FILE.write_text(code)
            (work / f"iter{it}_gen.cpp").write_text(code)  # keep every attempt
            messages.append({"role": "assistant", "content": f"```cpp\n{code}\n```"})

            ok, log = build()
            if not ok:
                (work / f"iter{it}_build.log").write_text(log)
                print("[agent] BUILD FAILED. Tail of compiler output:")
                print("\n".join(log.splitlines()[-25:]))
                print(f"[agent] full log: {work / f'iter{it}_build.log'}  code: {work / f'iter{it}_gen.cpp'}")
                messages.append({"role": "user", "content": "Compilation failed. Fix it. Errors:\n" + tail(log)})
                continue
            ok, log, gpu_ms = run(work, args.data)
            if not ok:
                (work / f"iter{it}_run.log").write_text(log)
                print("[agent] RUNTIME FAILED:")
                print("\n".join(log.splitlines()[-15:]))
                messages.append({"role": "user", "content": "Compiled but crashed. Fix it. Output:\n" + tail(log)})
                continue
            # Show both outputs, timings, and compare.
            show_parquet(work / "output.parquet", "GPU (gqe) output")
            speedup = (cpu_ms / gpu_ms) if gpu_ms else float("nan")
            sir = f"  |  Sirius (GPU) {sirius_ms:.1f} ms" if sirius_ms else ""
            print(f"[time] DuckDB (CPU, t={duckdb_threads}) {cpu_ms:.1f} ms  |  GPU (gqe) {gpu_ms:.1f} ms  "
                  f"|  speedup(vs CPU) {speedup:.2f}x{sir}")
            ok, detail = compare(work / "output.parquet", ref)
            print("[agent] vs DuckDB reference:", detail)
            if extra_ref and extra_ref.exists():
                ok2, detail2 = compare(work / "output.parquet", extra_ref)
                print(f"[agent] vs {extra_ref.name}:", detail2)
            if ok:
                sol = HERE / f"solution_{label}.cpp"
                sol.write_text(code)
                print(f"\n✅ SUCCESS iter {it}. Solution saved: {sol}")
                return 0
            messages.append({"role": "user", "content": "Ran but wrong result vs SQL reference. Fix the plan. Diff:\n" + tail(detail)})
        last = work / f"last_attempt_{label}.cpp"
        last.write_text(GEN_FILE.read_text())
        print(f"\n❌ did not converge in {args.max_iters} iters.")
        print(f"   all attempts + logs: {work}/iter*  | last attempt: {last}")
        print("   TIP: first confirm the harness itself compiles with the default Q3:")
        print("        git checkout tpch_bench/agent_gqe/harness/gen_query.cpp && \\")
        print("        ./tpch_bench/gqe/build_query.sh gqe_codegen_query")
        return 1
    finally:
        if backup is not None:
            GEN_FILE.write_text(backup)  # restore the committed q3 default


if __name__ == "__main__":
    raise SystemExit(main())
