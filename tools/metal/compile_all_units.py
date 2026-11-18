"""Translate and compile EVERY ArWen CUDA translation unit to Metal.

The recorded corpus (compile_corpus.py) holds what one configuration compiled.
This gate takes each unit in gpuwm/core/kernels/*.cu, assembles it exactly as
gpuwm's own loader does (gpuwm.core.kernels.module_source / module_options:
preamble, prepended headers, defines, NVRTC options), and runs it through the
CUDA -> MSL translator and `xcrun metal`.  Units the generic loader refuses
(they are composed by a scheme-specific loader) are listed as such.

    python tools/metal/compile_all_units.py [--jobs 8] [--out DIR] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def assemble(name):
    """Source and NVRTC options exactly as the unit's runtime loader builds them."""
    from gpuwm.core import kernels as K
    if name == "eeps":
        # gpuwm.core.eeps_pbl: an integer tier sized by the run's level count
        # (the WRF default e_vert 45 -> 44 mass levels)
        return K.module_source_int_defines("eeps", (("EEPS_NZ", 44),)), tuple(K.module_options("eeps"))
    if name == "fast_sbm":
        # gpuwm.core.fast_sbm_cuda.module: direct NVRTC with the kernel dir on the include path
        from gpuwm.core.fast_sbm_contract import CUDA_OPTIONS
        kdir = ROOT / "gpuwm/core/kernels"
        return (kdir / "fast_sbm.cu").read_text(), tuple(CUDA_OPTIONS) + (f"-I{kdir}",)
    if name == "nextgen_cfe":
        # gpuwm.hydro.cfe._compile_cfe (strict numerics)
        from gpuwm.hydro import cfe
        return cfe._cfe_source()[1], ("-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true",
                                      "--prec-sqrt=true")
    return K.module_source(name), tuple(K.module_options(name))


def one(args):
    name, mode, out = args
    t0 = time.time()
    try:
        src, opts = assemble(name)
    except Exception as e:
        return dict(unit=name, stage="loader", error=f"{type(e).__name__}: {e}"[:600])
    from gpuwm.metal.transpile import translate
    from gpuwm.metal.msl_compile import compile_msl
    try:
        t = translate(src, opts, f64_mode=mode, filename=f"{name}.cu")
    except Exception as e:
        return dict(unit=name, stage="translate", options=list(opts), error=f"{type(e).__name__}: {e}"[-1500:])
    t1 = time.time()
    import re as _re
    groups = len(_re.findall(r"int __arwen_tgt_\d+ = 0;", t.msl))      # merged backward-goto regions
    try:
        lib = Path(out) / f"{name}.metallib"
        compile_msl(t.msl, t.metal_flags, out=lib)
    except Exception as e:
        (Path(out) / f"{name}.metal").write_text(t.msl)
        (Path(out) / f"{name}.err").write_text(str(e))
        first = [l for l in str(e).splitlines() if " error: " in l][:3]
        return dict(unit=name, stage="msl", options=list(opts), goto_groups=groups,
                    error="\n".join(first) + "\n" + str(e)[-600:])
    return dict(unit=name, stage="ok", options=list(opts), kernels=len(t.kernels), goto_groups=groups,
                translate_s=round(t1 - t0, 1), compile_s=round(time.time() - t1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mode", default="sf64")
    ap.add_argument("--out", default="/tmp/arwen_metal_all_units")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    names = a.only or sorted(p.stem for p in (ROOT / "gpuwm/core/kernels").glob("*.cu"))
    t0 = time.time()
    res = []
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, [(n, a.mode, a.out) for n in names]):
            res.append(r)
            if r["stage"] != "ok":
                print(f"{r['unit']:40s} {r['stage']:9s} {r['error'][:300].replace(chr(10), ' | ')}", flush=True)
    c = collections.Counter(r["stage"] for r in res)
    print(f"units={len(res)} {dict(c)} kernels_compiled={sum(r.get('kernels', 0) for r in res)} "
          f"wall={time.time() - t0:.0f}s mode={a.mode}")
    (Path(a.out) / "report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
