"""Word-for-word comparison of two wrfout files (every variable, every element).

Usage: python tools/metal/wrfout_bitwise.py A B [--json out.json]
Exit status 0 only when every common variable is bit-identical and both files
hold the same variables.  NaNs compare by bit pattern.
"""
import argparse
import json
import sys

import netCDF4
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    a = ap.parse_args()
    da, db = netCDF4.Dataset(a.a), netCDF4.Dataset(a.b)
    for d in (da, db):
        d.set_auto_maskandscale(False)
    va, vb = set(da.variables), set(db.variables)
    rows, total_words, total_diff = [], 0, 0
    for name in sorted(va & vb):
        x, y = np.asarray(da.variables[name][:]), np.asarray(db.variables[name][:])
        if x.shape != y.shape or x.dtype != y.dtype:
            rows.append({"var": name, "error": f"shape/dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}"})
            total_diff += 1
            continue
        if x.dtype.kind in "fiu":
            xb = x.view(f"u{x.dtype.itemsize}") if x.dtype.kind == "f" else x
            yb = y.view(f"u{y.dtype.itemsize}") if y.dtype.kind == "f" else y
            n = int(np.count_nonzero(xb != yb))
        else:
            n = int(np.count_nonzero(x != y))
        total_words += x.size
        total_diff += n
        if n:
            rows.append({"var": name, "differing_words": n, "size": int(x.size)})
    out = {"a": a.a, "b": a.b, "variables_compared": len(va & vb), "only_in_a": sorted(va - vb),
           "only_in_b": sorted(vb - va), "words_compared": total_words, "differing_words": total_diff,
           "differing_variables": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "differing_variables"}))
    for r in rows[:40]:
        print("  ", r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if total_diff == 0 and not (va ^ vb) else 1


if __name__ == "__main__":
    sys.exit(main())
