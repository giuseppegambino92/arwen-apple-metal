"""Compare two wrfout files (candidate vs reference): bias, RMSE, correlation per field.

    python tools/metal/compare_wrfout.py CANDIDATE REFERENCE [--json OUT]
Surface verification fields first (T2, 10 m wind speed, PSFC, total rain), then
the 3-D prognostics.  Bias and RMSE are candidate - reference.
"""
import argparse
import json
import sys

import netCDF4
import numpy as np

SURFACE = ("T2", "WS10", "U10", "V10", "PSFC", "RAIN", "Q2", "TSK", "HFX", "LH", "SWDOWN", "GLW", "PBLH")
VOLUME = ("T", "U", "V", "W", "PH", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "P", "MU")


def field(ds, name):
    if name == "WS10":
        return np.hypot(field(ds, "U10"), field(ds, "V10"))
    if name == "RAIN":
        return field(ds, "RAINNC") + field(ds, "RAINC")
    v = ds.variables[name][...]
    return np.asarray(v, dtype=np.float64).squeeze()


def stats(c, r):
    d = c - r
    finite = np.isfinite(c) & np.isfinite(r)
    cf, rf = c[finite], r[finite]
    corr = float(np.corrcoef(cf.ravel(), rf.ravel())[0, 1]) if cf.size > 1 and cf.std() > 0 and rf.std() > 0 else None
    return {"bias": float(np.mean(d[finite])), "rmse": float(np.sqrt(np.mean(d[finite] ** 2))),
            "max_abs": float(np.max(np.abs(d[finite]), initial=0.0)), "corr": corr,
            "ref_std": float(rf.std()), "identical_fraction": float(np.mean(c == r)),
            "nonfinite_candidate": int(np.count_nonzero(~np.isfinite(c)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    with netCDF4.Dataset(a.candidate) as c, netCDF4.Dataset(a.reference) as r:
        for name in SURFACE + VOLUME:
            try:
                fc, fr = field(c, name), field(r, name)
            except KeyError:
                continue
            if fc.shape != fr.shape:
                out[name] = {"error": f"shape {fc.shape} vs {fr.shape}"}
                continue
            out[name] = stats(fc, fr)
    for k, v in out.items():
        if "error" in v:
            print(f"{k:7s} {v['error']}")
        else:
            print(f"{k:7s} bias={v['bias']:+.4g} rmse={v['rmse']:.4g} max|d|={v['max_abs']:.4g} "
                  f"corr={v['corr'] if v['corr'] is None else round(v['corr'], 6)} ref_std={v['ref_std']:.4g} "
                  f"identical={100 * v['identical_fraction']:.1f}% nonfinite={v['nonfinite_candidate']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
