#!/usr/bin/env python3
"""Generate TPC-H data (Parquet) + schema.sql + query SQL using DuckDB.

This produces a directory layout that GQE's scripts/load_tpch.py expects:

    <outdir>/
        schema.sql                # CREATE TABLE statements
        customer/customer.parquet
        lineitem/lineitem.parquet
        nation/nation.parquet
        orders/orders.parquet
        part/part.parquet
        partsupp/partsupp.parquet
        region/region.parquet
        supplier/supplier.parquet
        queries/q1.sql ... q22.sql   # (with --queries)

The same Parquet files can be fed to cudf TPC-H example programs that take a
dataset directory. No TPC dbgen binary is required -- DuckDB's bundled tpch
extension generates the data.

Usage:
    python gen_tpch_data.py --sf 1 --outdir /data/tpch_sf1 --queries
"""
import argparse
import os
import sys

TABLES = [
    "region", "nation", "supplier", "customer",
    "part", "partsupp", "orders", "lineitem",
]

# Standard TPC-H DDL. Types kept to widely-parseable SQL so both the GQE
# DataFusion/Substrait front-end and other engines accept it.
SCHEMA_SQL = """\
CREATE TABLE region (
  r_regionkey  BIGINT,
  r_name       VARCHAR,
  r_comment    VARCHAR
);
CREATE TABLE nation (
  n_nationkey  BIGINT,
  n_name       VARCHAR,
  n_regionkey  BIGINT,
  n_comment    VARCHAR
);
CREATE TABLE supplier (
  s_suppkey    BIGINT,
  s_name       VARCHAR,
  s_address    VARCHAR,
  s_nationkey  BIGINT,
  s_phone      VARCHAR,
  s_acctbal    DECIMAL(15,2),
  s_comment    VARCHAR
);
CREATE TABLE customer (
  c_custkey    BIGINT,
  c_name       VARCHAR,
  c_address    VARCHAR,
  c_nationkey  BIGINT,
  c_phone      VARCHAR,
  c_acctbal    DECIMAL(15,2),
  c_mktsegment VARCHAR,
  c_comment    VARCHAR
);
CREATE TABLE part (
  p_partkey     BIGINT,
  p_name        VARCHAR,
  p_mfgr        VARCHAR,
  p_brand       VARCHAR,
  p_type        VARCHAR,
  p_size        INTEGER,
  p_container   VARCHAR,
  p_retailprice DECIMAL(15,2),
  p_comment     VARCHAR
);
CREATE TABLE partsupp (
  ps_partkey    BIGINT,
  ps_suppkey    BIGINT,
  ps_availqty   INTEGER,
  ps_supplycost DECIMAL(15,2),
  ps_comment    VARCHAR
);
CREATE TABLE orders (
  o_orderkey      BIGINT,
  o_custkey       BIGINT,
  o_orderstatus   VARCHAR,
  o_totalprice    DECIMAL(15,2),
  o_orderdate     DATE,
  o_orderpriority VARCHAR,
  o_clerk         VARCHAR,
  o_shippriority  INTEGER,
  o_comment       VARCHAR
);
CREATE TABLE lineitem (
  l_orderkey      BIGINT,
  l_partkey       BIGINT,
  l_suppkey       BIGINT,
  l_linenumber    INTEGER,
  l_quantity      DECIMAL(15,2),
  l_extendedprice DECIMAL(15,2),
  l_discount      DECIMAL(15,2),
  l_tax           DECIMAL(15,2),
  l_returnflag    VARCHAR,
  l_linestatus    VARCHAR,
  l_shipdate      DATE,
  l_commitdate    DATE,
  l_receiptdate   DATE,
  l_shipinstruct  VARCHAR,
  l_shipmode      VARCHAR,
  l_comment       VARCHAR
);
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf", type=float, default=1.0, help="scale factor (default 1)")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--queries", action="store_true",
                    help="also dump q1.sql..q22.sql under <outdir>/queries")
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
    con.execute("INSTALL tpch; LOAD tpch;")
    print(f"[gen] dbgen sf={args.sf} ...", flush=True)
    con.execute(f"CALL dbgen(sf={args.sf})")

    for t in TABLES:
        tdir = os.path.join(outdir, t)
        os.makedirs(tdir, exist_ok=True)
        dest = os.path.join(tdir, f"{t}.parquet")
        print(f"[gen] writing {dest}", flush=True)
        con.execute(
            f"COPY (SELECT * FROM {t}) TO '{dest}' (FORMAT parquet)"
        )

    with open(os.path.join(outdir, "schema.sql"), "w") as f:
        f.write(SCHEMA_SQL)
    print(f"[gen] wrote {os.path.join(outdir, 'schema.sql')}", flush=True)

    if args.queries:
        qdir = os.path.join(outdir, "queries")
        os.makedirs(qdir, exist_ok=True)
        rows = con.execute(
            "SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr"
        ).fetchall()
        for nr, sql in rows:
            with open(os.path.join(qdir, f"q{nr}.sql"), "w") as f:
                f.write(sql.strip() + "\n")
        print(f"[gen] wrote {len(rows)} query files under {qdir} "
              f"(DuckDB dialect -- adjust per engine if needed)", flush=True)

    print("[gen] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
