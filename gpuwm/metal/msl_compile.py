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


class MetalCompileError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def toolchain_identity() -> str:
    r = subprocess.run(["xcrun", "metal", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise MetalCompileError("Metal toolchain unavailable (xcodebuild -downloadComponent MetalToolchain):\n" + r.stderr)
    return r.stdout.strip()


def compile_msl(msl: str, flags=(), *, out: Path, opt="-O3") -> Path:
    """MSL text -> metallib at ``out`` (written atomically)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arwen-msl-") as td:
        src = Path(td) / "module.metal"
        air = Path(td) / "module.air"
        src.write_text(msl)
        cmd = ["xcrun", "metal", f"-std={MSL_STD}", opt, *flags, "-I", str(INCLUDE), "-c", str(src), "-o", str(air)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalCompileError(r.stderr)
        tmp = out.with_suffix(".tmp")
        r = subprocess.run(["xcrun", "metallib", str(air), "-o", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalCompileError(r.stderr)
        os.replace(tmp, out)
    return out
