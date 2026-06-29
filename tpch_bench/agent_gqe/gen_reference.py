#!/usr/bin/env python3
"""Run TPC-H / TPC-DS queries in DuckDB over a parquet dataset and save each result locally.

These are the CPU ground-truth results the GPU output is compared against later.
Registers <data>/<table>/*.parquet as DuckDB views, then runs the benchmark query SQL and writes
<outdir>/qN.parquet (and <outdir>/qN.sql).

Usage:
    # all TPC-DS queries:
    python gen_reference.py --bench tpcds --data /data/tpcds_sf1 --query all --outdir tpcds_ref
    # all TPC-H queries:
    python gen_reference.py --bench tpch  --data /data/tpch_sf1  --query all --outdir tpch_ref
    # a single query:
    python gen_reference.py --bench tpcds --data /data/tpcds_sf1 --query 3
"""
import argparse
import sys
from pathlib import Path


def connect(data):
    import duckdb
    con = duckdb.connect()
    for ext in ("tpch", "tpcds"):
        try:
            con.execute(f"INSTALL {ext}; LOAD {ext};")
        except Exception:
            pass
    tables = []
    for sub in sorted(p for p in Path(data).iterdir() if p.is_dir()):
        files = sorted(sub.glob("*.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {sub.name} AS SELECT * FROM read_parquet({[str(f) for f in files]})")
            tables.append(sub.name)
    if not tables:
        sys.exit(f"no <table>/*.parquet under {data}")
    return con


def query_numbers(con, bench, which):
    fn = "tpch_queries" if bench == "tpch" else "tpcds_queries"
    if which != "all":
        return [int(which)], fn
    rows = con.execute(f"SELECT query_nr FROM {fn}() ORDER BY query_nr").fetchall()
    return [r[0] for r in rows], fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=["tpch", "tpcds"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--query", default="all", help="query number or 'all'")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    con = connect(args.data)
    nums, fn = query_numbers(con, args.bench, args.query)
    out = Path(args.outdir or f"{args.bench}_ref"); out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for n in nums:
        row = con.execute(f"SELECT query FROM {fn}() WHERE query_nr = ?", [n]).fetchone()
        if not row:
            print(f"[ref] {args.bench} q{n}: no such query", file=sys.stderr); continue
        sql = row[0].strip().rstrip(";")
        (out / f"q{n}.sql").write_text(sql + "\n")
        try:
            con.execute(f"COPY ({sql}) TO '{out / f'q{n}.parquet'}' (FORMAT parquet)")
            print(f"[ref] wrote {out / f'q{n}.parquet'}")
            ok += 1
        except Exception as e:
            print(f"[ref] {args.bench} q{n} failed: {str(e).splitlines()[0]}", file=sys.stderr)
    print(f"[ref] done: {ok}/{len(nums)} results in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
