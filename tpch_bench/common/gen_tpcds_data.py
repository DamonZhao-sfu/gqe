#!/usr/bin/env python3
"""Generate TPC-DS data (Parquet) + query SQL using DuckDB.

The GQE standalone benchmark programs under benchmark/hardcoded/ (q3, q6, q7,
q22, q38, q43, q48, q3_udr) are ALL TPC-DS queries. They read per-table Parquet
from a dataset directory laid out as:

    <outdir>/
        store_sales/store_sales.parquet
        date_dim/date_dim.parquet
        item/item.parquet
        customer/customer.parquet
        ... (one subdir per TPC-DS table)
        queries/q1.sql ... q99.sql        # (with --queries)

No TPC dsdgen binary is required -- DuckDB's bundled tpcds extension generates
the data.

Usage:
    python gen_tpcds_data.py --sf 1 --outdir /data/tpcds_sf1 --queries
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", type=float, default=1.0, help="scale factor (default 1)")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--queries", action="store_true",
                    help="also dump q1.sql..q99.sql under <outdir>/queries")
    ap.add_argument("--threads", type=int, default=0,
                    help="DuckDB threads (0 = auto)")
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb not installed. `conda install -c conda-forge "
              "python-duckdb` or `pip install duckdb`.", file=sys.stderr)
        return 1

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    con = duckdb.connect()
    if args.threads:
        con.execute(f"PRAGMA threads={args.threads}")
    con.execute("INSTALL tpcds; LOAD tpcds;")
    print(f"[gen] dsdgen sf={args.sf} (this can take a while) ...", flush=True)
    con.execute(f"CALL dsdgen(sf={args.sf})")

    # Discover every generated table rather than hardcoding the 24 names.
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()]
    # dsdgen creates a metadata table we don't want as a 'data' dir.
    tables = [t for t in tables if t != "dbgen_version"]

    for t in tables:
        tdir = os.path.join(outdir, t)
        os.makedirs(tdir, exist_ok=True)
        dest = os.path.join(tdir, f"{t}.parquet")
        print(f"[gen] writing {dest}", flush=True)
        con.execute(f"COPY (SELECT * FROM {t}) TO '{dest}' (FORMAT parquet)")

    if args.queries:
        qdir = os.path.join(outdir, "queries")
        os.makedirs(qdir, exist_ok=True)
        try:
            rows = con.execute(
                "SELECT query_nr, query FROM tpcds_queries() ORDER BY query_nr"
            ).fetchall()
        except Exception as e:
            print(f"[gen] WARN: could not dump queries via tpcds_queries(): {e}",
                  file=sys.stderr)
            rows = []
        for nr, sql in rows:
            with open(os.path.join(qdir, f"q{nr}.sql"), "w") as f:
                f.write(sql.strip() + "\n")
        if rows:
            print(f"[gen] wrote {len(rows)} query files under {qdir} "
                  f"(DuckDB dialect -- adjust per engine if needed)", flush=True)

    print(f"[gen] done. {len(tables)} tables under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
