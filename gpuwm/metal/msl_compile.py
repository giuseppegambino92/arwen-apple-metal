"""Compile translated MSL to a .metallib with the Xcode Metal toolchain (xcrun metal)."""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

INCLUDE = Path(__file__).resolve().parent / "include"
MSL_STD = os.environ.get("ARWEN_METAL_STD", "metal4.0")
# AIR optimisation level.  Apple's GPU back end (MTLCompilerService, run when a
# pipeline is created) can spend tens of minutes on a large kernel built at -O3
# (measured: HRRR Thompson fork column > 15 min, 7.7 s at -Os).
MSL_OPT = os.environ.get("ARWEN_METAL_OPT", "-O3")


class MetalCompileError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def toolchain_identity() -> str:
    r = subprocess.run(["xcrun", "metal", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise MetalCompileError("Metal toolchain unavailable (xcodebuild -downloadComponent MetalToolchain):\n" + r.stderr)
    return r.stdout.strip() + " " + MSL_STD + " " + MSL_OPT


def compile_msl(msl: str, flags=(), *, out: Path, opt=None) -> Path:
    """MSL text -> metallib at ``out`` (written atomically)."""
    opt = opt or MSL_OPT
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arwen-msl-") as td:
        src = Path(td) / "module.metal"
        air = Path(td) / "module.air"
        src.write_text(msl)
        cmd = ["xcrun", "metal", f"-std={MSL_STD}", opt, *flags, "-I", str(INCLUDE), "-c", str(src), "-o", str(air)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            keep = os.environ.get("ARWEN_METAL_KEEP_FAILED")      # diagnostics: keep the failing MSL
            if keep:
                import hashlib
                Path(keep).mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256(msl.encode()).hexdigest()[:12]
                (Path(keep) / f"{h}.metal").write_text(msl)
                (Path(keep) / f"{h}.err").write_text(r.stderr)
            raise MetalCompileError(r.stderr)
        tmp = out.with_suffix(".tmp")
        r = subprocess.run(["xcrun", "metallib", str(air), "-o", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalCompileError(r.stderr)
        os.replace(tmp, out)
    return out
