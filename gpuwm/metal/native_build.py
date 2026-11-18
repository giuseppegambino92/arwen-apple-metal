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


_GIL_SRC = r"""
#include <Python.h>
/* Single-phase init without a Py_mod_gil slot: importing this module makes a
   free-threaded interpreter re-enable the GIL for the process. */
static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "_arwen_metal_gil", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__arwen_metal_gil(void) { return PyModule_Create(&def); }
"""


def require_gil():
    """The Metal backend runs with the GIL.  On free-threaded Python, concurrent
    ensemble members without it corrupted Python objects (a race in the
    shim/runtime not yet isolated), so the backend re-enables the GIL, exactly as
    importing any extension without free-threading support does."""
    if not bool(sysconfig.get_config_var("Py_GIL_DISABLED")) or sys._is_gil_enabled():
        return
    _, _, inc = _flags()
    h = hashlib.sha256(_GIL_SRC.encode() + sys.version.encode()).hexdigest()[:16]
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    out = cache_dir() / f"_arwen_metal_gil-{h}{suffix}"
    if not out.exists():
        src = cache_dir() / f"gil-{h}.c"
        src.write_text(_GIL_SRC)
        tmp = out.with_suffix(".tmp.so")
        subprocess.run(["xcrun", "clang", "-O2", "-fPIC", *inc, "-shared", "-undefined", "dynamic_lookup",
                        str(src), "-o", str(tmp)], check=True)
        os.replace(tmp, out)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # "The global interpreter lock (GIL) has been enabled ..."
        spec = importlib.util.spec_from_file_location("_arwen_metal_gil", out)
        importlib.util.module_from_spec(spec)


def load():
    require_gil()
    path = build()
    spec = importlib.util.spec_from_file_location("_arwen_metal_rt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
