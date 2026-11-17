"""Exhaustive GPU verification of the glibc float libm fast paths (gpuwm.metal.fastlibm).

For every candidate copy (the ones the translator recorded under
<cache>/fastlibm_pending, or --corpus DIR of recorded CUDA sources), build a
standalone module holding the copy, everything it reaches, and a kernel that
evaluates the routed wrapper and the exact original on every binary32 input
(2^32 values, all NaN payloads included) and counts differing words.  A copy
enters gpuwm/metal/fastlibm_receipt.json with its counts; the translator routes
it only when mismatches == 0.

Usage: python tools/metal/verify_fastlibm.py [--corpus probe/cuda_1h] [--limit-bits 32]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("GPUWM_BACKEND", "metal")
import numpy as np  # noqa: E402

from gpuwm.metal import fastlibm as F  # noqa: E402
from gpuwm.metal.frontend import NvrtcOptions, preprocess  # noqa: E402
from gpuwm.metal.native_build import cache_dir  # noqa: E402

KERNEL = r'''
extern "C" __global__ void __arwen_fastlibm_verify(unsigned int base, unsigned int per,
                                                   unsigned int* stats, unsigned int* bad) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int fast = 0;
    for (unsigned int j = 0; j < per; ++j) {
        unsigned int ix = base + t * per + j;
        float x = __uint_as_float(ix);
        float a = EXACT;
        float b = ROUTED;
        if (__float_as_uint(FASTCALL) != 0x7FBADBADu) ++fast;
        if (__float_as_uint(a) != __float_as_uint(b)) {
            unsigned int n = atomicAdd(&stats[0], 1u);
            if (n < 64u) bad[n] = ix;
        }
    }
    atomicAdd(&stats[1], fast);
}
'''


def jobs_from_pending():
    d = cache_dir() / "fastlibm_pending"
    for p in sorted(d.glob("*.json")) if d.exists() else ():
        j = json.loads(p.read_text())
        yield p.stem, j["name"], j["family"], j.get("ybits"), j["source"], j["options"], j.get("cupy_raw_module", False)


def jobs_from_corpus(root: Path):
    idx = json.loads((root / "index.json").read_text())
    seen = set()
    for e in idx:
        if e.get("cupy_generated"):
            continue
        src = (root / "src" / f"{e['sha']}.cu").read_text()
        if "double" not in src:
            continue
        for ftz in ("--ftz=true", "--ftz=false"):
            options = ["-std=c++17", ftz]
            o = NvrtcOptions.parse(options)
            try:
                pp = preprocess(src, o, "kernel.cu")
            except Exception:
                break
            for c in F.candidates(pp, o):
                if c.key not in seen:
                    seen.add(c.key)
                    yield c.key, c.name, c.family, c.ybits, src, options, False


def _hex_literal(bits):
    v = float(np.uint32(bits).view(np.float32))
    return f"{v.hex()}f"


def verify(key, name, family, ybits, source, options, bits, raw=False):
    import cupy as cp
    from gpuwm.metal import runtime
    o = NvrtcOptions.parse(options, cupy_raw_module=raw)
    pp = preprocess(source, o, "kernel.cu")
    cands = [c for c in F.candidates(pp, o) if c.key == key]
    if not cands:
        return {"error": "fingerprint not reproduced"}
    unit = F.closure_source(pp, name)
    fast = F.FAMILIES[family][0]
    if ybits is None:
        k = KERNEL.replace("EXACT", f"{name}__arwen_exact(x)").replace("ROUTED", f"{name}(x)")
        k = k.replace("FASTCALL", f"{fast}(x)")
    else:
        yexpr = f"__uint_as_float(0x{ybits:08X}u)"
        k = KERNEL.replace("EXACT", f"{name}(x, {yexpr})").replace("ROUTED", f"{name}(x, {_hex_literal(ybits)})")
        k = k.replace("FASTCALL", f"{fast}(x, {yexpr})")
    unit += k
    os.environ["ARWEN_METAL_FASTLIBM_VERIFYING"] = key
    try:
        mod = runtime.Module(unit, tuple(options), cupy_raw_module=raw)
    finally:
        del os.environ["ARWEN_METAL_FASTLIBM_VERIFYING"]
    k = mod.get_function("__arwen_fastlibm_verify")
    stats = cp.zeros(2, cp.uint32)
    bad = cp.zeros(64, cp.uint32)
    total = 1 << bits
    chunk = min(total, 1 << 26)
    per, block = 64, 256
    threads = chunk // per
    t0 = time.time()
    for base in range(0, total, chunk):
        k((threads // block,), (block,), (np.uint32(base), np.uint32(per), stats, bad))
        runtime.synchronize()
    s = cp.asnumpy(stats)
    nb = min(int(s[0]), 64)
    return {"name": name, "family": family, "options": list(options), "cupy_raw_module": raw, "inputs_checked": total,
            "exponent": None if ybits is None else f"0x{ybits:08X}",
            "mismatches": int(s[0]), "fast_path_fraction": float(s[1]) / total,
            "first_mismatch_inputs": [f"0x{v:08X}" for v in cp.asnumpy(bad)[:nb]],
            "gpu_seconds": round(time.time() - t0, 1),
            "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--limit-bits", type=int, default=32, help="< 32 only for a quick smoke check (not recorded)")
    a = ap.parse_args()
    jobs = list(jobs_from_corpus(a.corpus) if a.corpus else jobs_from_pending())
    rec = json.loads(F.RECEIPT.read_text()) if F.RECEIPT.exists() else {}
    rec.setdefault("about", "gpuwm.metal.fastlibm: copies routed through arwen_fastlibm.h, each compared with the "
                            "exact original on all 2^32 binary32 inputs on the GPU (tools/metal/verify_fastlibm.py)")
    ver = rec.setdefault("verified", {})
    for key, name, family, ybits, src, options, raw in jobs:
        if key in ver and a.limit_bits == 32:
            continue
        r = verify(key, name, family, ybits, src, options, a.limit_bits, raw)
        ytxt = "" if ybits is None else f"y=0x{ybits:08X}"
        print(f"{name:24s} {family} {ytxt:12s} {' '.join(options)[:40]:40s} -> {json.dumps({k: r.get(k) for k in ('mismatches', 'fast_path_fraction', 'gpu_seconds', 'first_mismatch_inputs', 'error')})}",
              flush=True)
        if a.limit_bits == 32 and "error" not in r:
            ver[key] = r
            F.RECEIPT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
            p = cache_dir() / "fastlibm_pending" / f"{key}.json"
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    sys.exit(main())
