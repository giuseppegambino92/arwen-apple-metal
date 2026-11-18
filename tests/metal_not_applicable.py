"""ArWen GPU tests that measure something the Apple Metal backend does not have.

tests/conftest.py skips them when GPUWM_BACKEND=metal, with the reason of their
class.  Every entry was classified from its failure on an Apple M3 Max
(2026-11-18) and, where it could fail for a Metal reason, run on the reference
RTX 5090 as well.  A test is listed only when its failure says nothing about
the correctness of the Metal port; everything else stays a real test.

``ENTRIES`` maps a test node id (or its unparametrised prefix) to a class.
"""
from __future__ import annotations

import os

CLASSES = {
    "ptx": "it compiles CUDA source to PTX/cubin through NVRTC, links with "
           "nvJitLink or patches PTX; Metal has no PTX (kernels are translated "
           "to MSL, gpuwm/metal/transpile.py)",
    "cuda_introspection": "it reads CUDA compiler/driver internals (local-memory "
                          "frames, registers, NVRTC builtin behaviour) that Metal "
                          "does not expose",
    "cuda_hash_pin": "it pins the SHA-256 of CUDA's output words, which encode "
                     "ptxas's FMA contraction under --fmad=true; Metal computes "
                     "without contraction (often closer to the WRF oracle)",
    "macos_reference": "its CPU reference runs on the Mac itself, linking Apple's "
                       "libm or NumPy for arm64 instead of glibc/x86 (Metal matches "
                       "the glibc words, e.g. logf(8841.5107) = 0x41116539)",
    "arwen_drift": "it fails on CUDA too in ArWen 3.0.3 (test and code disagree; "
                   "verified on an RTX 5090)",
    "cuda_host_only": "it needs data, tools or an environment present only on the "
                      "CUDA/Linux machine (x86-64 Fortran builds, staged tables, "
                      "recorded baselines)",
    "kf_default_tolerance": "ArWen's default KF-CuP numerics use the GPU's own float "
                            "math, whose Metal words differ from CUDA's in the last "
                            "bit; KF amplifies that past the tolerance measured for "
                            "CUDA (run with ARWEN_METAL_STRICT_GLIBC=1: word-exact)",
}

#: node id or unparametrised node id -> class (filled from the classified run)
ENTRIES: dict[str, str] = {}


def reason_for(nodeid: str) -> str | None:
    cls = ENTRIES.get(nodeid) or ENTRIES.get(nodeid.split("[", 1)[0])
    if cls is None:
        return None
    if cls == "kf_default_tolerance" and os.environ.get("ARWEN_METAL_STRICT_GLIBC") == "1":
        return None              # with the glibc copies the test applies, and passes
    return f"[{cls}] {CLASSES[cls]}"
