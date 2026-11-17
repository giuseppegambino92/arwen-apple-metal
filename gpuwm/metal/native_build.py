"""Build (once, cached by source hash) the native Metal runtime extension."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "native" / "arwen_metal_rt.mm"


def cache_dir() -> Path:
    d = Path(os.environ.get("ARWEN_METAL_CACHE", Path.home() / ".cache" / "arwen-metal"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _flags():
    import nanobind
    nb = Path(nanobind.source_dir()).parent
    py_inc = sysconfig.get_paths()["include"]
    ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    defs = ["-DNB_FREE_THREADED"] if ft else []
    inc = [f"-I{nanobind.include_dir()}", f"-I{nb / 'ext' / 'robin_map' / 'include'}", f"-I{py_inc}"]
    return nb, defs, inc


def build(force: bool = False) -> Path:
    nb, defs, inc = _flags()
    h = hashlib.sha256(SRC.read_bytes())
    h.update(sys.version.encode())
    h.update(str(nb).encode())
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    out = cache_dir() / f"_arwen_metal_rt-{h.hexdigest()[:16]}{suffix}"
    if out.exists() and not force:
        return out
    objdir = cache_dir() / "build"
    objdir.mkdir(exist_ok=True)
    common = ["-O2", "-fvisibility=hidden", "-fPIC", "-std=c++17", *defs, *inc]
    nbobj = objdir / f"nb_combined-{h.hexdigest()[:8]}.o"
    subprocess.run(["xcrun", "clang++", *common, "-c", str(nb / "src" / "nb_combined.cpp"), "-o", str(nbobj)], check=True)
    rtobj = objdir / f"rt-{h.hexdigest()[:8]}.o"
    subprocess.run(["xcrun", "clang++", *common, "-fobjc-arc", "-x", "objective-c++", "-c", str(SRC), "-o", str(rtobj)], check=True)
    tmp = out.with_suffix(".tmp.so")
    subprocess.run(["xcrun", "clang++", "-shared", "-undefined", "dynamic_lookup", str(nbobj), str(rtobj),
                    "-framework", "Metal", "-framework", "Foundation", "-o", str(tmp)], check=True)
    os.replace(tmp, out)
    return out


def load():
    path = build()
    spec = importlib.util.spec_from_file_location("_arwen_metal_rt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
