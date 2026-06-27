#!/usr/bin/env python3
"""Compare two Parquet result files for (approximate) equality.

Used to check that a fused-kernel (UDR) query produces the same result as the
original. Rows are sorted before comparison (order-independent); numeric columns
are compared with a tolerance (floating-point SUM/AVG can differ in the last
bits depending on reduction order), other columns exactly.

Usage:
    compare_parquet.py A.parquet B.parquet [--rtol 1e-6] [--atol 1e-3]
Exit code 0 if they match, 1 if they differ (or on error).
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"ERROR: need pandas/pyarrow/numpy ({e})", file=sys.stderr)
        return 1

    try:
        a = pq.read_table(args.a).to_pandas()
        b = pq.read_table(args.b).to_pandas()
    except Exception as e:
        print(f"DIFFER: failed to read parquet ({e})")
        return 1

    if a.shape != b.shape:
        print(f"DIFFER: shape {a.shape} vs {b.shape} "
              f"(rows {a.shape[0]} vs {b.shape[0]}, cols {a.shape[1]} vs {b.shape[1]})")
        return 1

    if a.shape[0] == 0:
        print("MATCH: both empty")
        return 0

    # Sort each independently by all columns (positional), so row order doesn't matter.
    a = a.sort_values(list(a.columns)).reset_index(drop=True)
    b = b.sort_values(list(b.columns)).reset_index(drop=True)

    total_bad = 0
    details = []
    for i in range(a.shape[1]):
        ca, cb = a.iloc[:, i], b.iloc[:, i]
        name = str(a.columns[i])
        if np.issubdtype(ca.dtype, np.number) and np.issubdtype(cb.dtype, np.number):
            close = np.isclose(ca.to_numpy(dtype="float64"),
                               cb.to_numpy(dtype="float64"),
                               rtol=args.rtol, atol=args.atol, equal_nan=True)
            nbad = int((~close).sum())
            if nbad:
                diff = np.abs(ca.to_numpy(dtype="float64") - cb.to_numpy(dtype="float64"))
                details.append(f"  col[{i}] {name}: {nbad} numeric mismatch, max|Δ|={np.nanmax(diff):.6g}")
        else:
            eq = ca.astype("string").fillna("\x00") == cb.astype("string").fillna("\x00")
            nbad = int((~eq).sum())
            if nbad:
                details.append(f"  col[{i}] {name}: {nbad} value mismatch")
        total_bad += nbad

    if total_bad == 0:
        print(f"MATCH: {a.shape[0]} rows x {a.shape[1]} cols identical "
              f"(rtol={args.rtol}, atol={args.atol})")
        return 0
    print(f"DIFFER: {total_bad} mismatching cells over {a.shape[0]} rows")
    print("\n".join(details))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
