"""Build every scheme-composed ArWen unit on Metal through its PRODUCTION loader.

compile_all_units.py assembles each gpuwm/core/kernels/*.cu the generic way
(kernels.module_source / module_options).  About thirty units are composed by
a scheme's own loader instead (chained translation units, per-device direct
NVRTC loaders, integer tiers, *_default sources).  This gate calls those very
loaders with GPUWM_BACKEND=metal, so the CUDA source, options and composition
are exactly the ones a forecast compiles, and each build goes through the
CUDA -> MSL translator, `xcrun metal` and the Metal library load.

    python tools/metal/compile_loader_units.py [--jobs 6] [--exact] [--only LABEL ...]

``--exact`` sets GPUWM_WRF_EXACT=1 first (the strict-chain sources).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def jobs():
    """(label, 'module:function', args) for every production loader."""
    from gpuwm import kernel_prewarm as kp
    from gpuwm.core.preflight import PRODUCTION_LOADED_KERNEL_UNITS
    out = []
    for name, (mod, fn) in kp._COMPOSED_LOADERS.items():
        out.append((name, "gpuwm.kernel_prewarm:_warm_composed", (name,)))
    for name in kp._DEVICE_LOADERS:
        out.append((name, "gpuwm.kernel_prewarm:_warm_device_loader", (name,)))
    for name, trig in kp._KF_TRIGGERS.items():
        out.append((name, "gpuwm.kernel_prewarm:_warm_kf", (trig,)))
    for name, ref in PRODUCTION_LOADED_KERNEL_UNITS.items():
        out.append((name, ref, ()))
    dev = (0,)
    for name in ("sfire_atm", "sfire_core", "sfire_moisture", "sfire_phys", "sfire_spotting", "sfire_wind"):
        out.append((name, f"gpuwm.core.{name}:_module", dev))
    out += [
        ("snow_papers", "gpuwm.core.snow_papers:_module", dev),
        ("snow_lap", "gpuwm.core.snow_papers:_lap_module", dev),
        ("mynnsfc_b66187f", "gpuwm.core.mynnsfc_b66187f:module", ()),
        ("p3_zdiag", "gpuwm.core.p3_device:p3_reflectivity_module", ()),
        ("vortex_init", "gpuwm.tc.vortex:_module", ()),
        ("eeps[EEPS_NZ=44]", "gpuwm.core.kernels:load_module_int_defines", ("eeps", (("EEPS_NZ", 44),))),
        ("flg_column[FLG_NV1=45]", "gpuwm.core.kernels:load_module_unflushed", ("flg_column", (("FLG_NV1", 45),))),
        ("rrtmgp_rte[RRTMGP_MAX_LAYERS=62]", "gpuwm.core.kernels:load_module_int_defines",
         ("rrtmgp_rte", (("RRTMGP_MAX_LAYERS", 62),))),
        ("mskf[GPUWM_MSKF_NO_AEROSOL=1]", "gpuwm.core.kernels.no_ftz:load_module_no_ftz",
         ("mskf", (("GPUWM_MSKF_NO_AEROSOL", 1),))),
    ]
    # the *_default units are these modules' default sources
    for name in ("rrtmgp_gas", "ysu", "nssl2_driver_support", "urban_ucm480"):
        out.append((name, "gpuwm.core.kernels:load_module", (name,)))
    return out


def one(job):
    label, ref, args = job
    os.environ["GPUWM_BACKEND"] = "metal"
    t0 = time.time()
    try:
        import importlib
        mod, fn = ref.split(":")
        f = getattr(importlib.import_module(mod), fn)
        f = getattr(f, "__wrapped__", f)          # the uncached body, as the pre-warm calls it
        f(*args)
        return dict(label=label, ok=True, s=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        first = [line for line in msg.splitlines() if " error: " in line][:2]
        return dict(label=label, ok=False, s=round(time.time() - t0, 1),
                    error=("\n".join(first) + "\n" if first else "") + msg[:700],
                    where=traceback.format_exc().splitlines()[-6:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.exact:
        os.environ["GPUWM_WRF_EXACT"] = "1"
    os.environ["GPUWM_BACKEND"] = "metal"
    js = [j for j in jobs() if not a.only or j[0] in a.only]
    t0 = time.time()
    res = []
    with cf.ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(one, js):
            res.append(r)
            print(f"{r['label']:40s} {'ok' if r['ok'] else 'FAIL'} {r['s']:7.1f}s "
                  + ("" if r["ok"] else r["error"][:400].replace("\n", " | ")), flush=True)
    n_ok = sum(r["ok"] for r in res)
    print(f"loaders={len(res)} ok={n_ok} fail={len(res) - n_ok} exact={a.exact} wall={time.time() - t0:.0f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
