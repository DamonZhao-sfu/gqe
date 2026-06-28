#!/usr/bin/env python3
"""Generate DuckDB CPU reference results for TPC-DS queries over a parquet dataset.

Registers <data>/<table>/*.parquet as DuckDB views, runs the TPC-DS query SQL from DuckDB's tpcds
extension, and writes reference/qN.parquet. This is the ground truth the GPU output is checked against.

Usage:
    python gen_reference.py --data /data/tpcds_sf1 --query 3
    python gen_reference.py --data /data/tpcds_sf1 --query all --outdir ./tpcds_ref
"""
import argparse
import sys
from pathlib import Path

TPCDS_TABLES = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
    "customer_address", "customer_demographics", "date_dim", "household_demographics",
    "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
    "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]


def connect(data):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds;")
    for t in TPCDS_TABLES:
        files = sorted((Path(data) / t).glob("*.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet({[str(f) for f in files]})")
    return con


def query_sql(con, n):
    row = con.execute("SELECT query FROM tpcds_queries() WHERE query_nr = ?", [n]).fetchone()
    if not row:
        sys.exit(f"no TPC-DS query #{n}")
    return row[0].strip().rstrip(";")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--query", required=True, help="query number or 'all'")
    ap.add_argument("--outdir", default="./tpcds_ref")
    args = ap.parse_args()

    con = connect(args.data)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    nums = range(1, 100) if args.query == "all" else [int(args.query)]
    for n in nums:
        try:
            sql = query_sql(con, n)
            dest = out / f"q{n}.parquet"
            con.execute(f"COPY ({sql}) TO '{dest}' (FORMAT parquet)")
            print(f"[ref] wrote {dest}")
        except Exception as e:
            print(f"[ref] q{n} failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
