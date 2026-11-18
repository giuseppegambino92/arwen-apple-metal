"""CUDA front end of the CUDA->MSL translator: preprocess, then parse with libclang.

The CUDA source handed to CuPy/NVRTC stays canonical.  This module only reads
it: the source is fully preprocessed (with the -D/-U/-I options the caller gave
NVRTC, and NVRTC's predefined macros), so every AST node has exactly one
spelling location in the text the rewriter edits.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSE_HEADER = HERE / "cuda_parse.h"
#: NVRTC's predefined macros for the CUDA target whose output is the reference
#: (RTX 5090, sm_120).  Device-code branches select on these, so the Metal
#: translation takes the same branches as the reference run.
CUDA_ARCH = 1200
NVRTC_DEFINES = ("__CUDACC__", "__CUDACC_RTC__", "__NVCC__", f"__CUDA_ARCH__={CUDA_ARCH}",
                 "__CUDA_ARCH_LIST__=1200")


@dataclasses.dataclass(frozen=True)
class NvrtcOptions:
    defines: tuple[str, ...] = ()
    undefines: tuple[str, ...] = ()
    includes: tuple[str, ...] = ()
    ftz: bool = True          # CuPy appends -ftz=true unless the direct-NVRTC route says otherwise
    fmad: bool = True
    fast_math: bool = False
    prec_div: bool = True
    prec_sqrt: bool = True
    std: str = "c++17"
    unknown: tuple[str, ...] = ()

    @classmethod
    def parse(cls, options, *, cupy_raw_module=False):
        d, u, inc, unk = [], [], [], []
        flags = dict(ftz=None, fmad=True, fast_math=False, prec_div=True, prec_sqrt=True, std="c++17")
        it = iter(options or ())
        for o in it:
            o = str(o)
            if o in ("-D", "-U", "-I"):
                o = o + next(it)
            if o.startswith("-D"):
                d.append(o[2:])
            elif o.startswith("-U"):
                u.append(o[2:])
            elif o.startswith("-I"):
                inc.append(o[2:])
            elif o.startswith(("-std=", "--std=")):
                flags["std"] = o.split("=", 1)[1]
            elif o.startswith(("--ftz=", "-ftz=")):
                flags["ftz"] = o.split("=", 1)[1] == "true"
            elif o.startswith(("--fmad=", "-fmad=")):
                flags["fmad"] = o.split("=", 1)[1] == "true"
            elif o.startswith(("--prec-div=", "-prec-div=")):
                flags["prec_div"] = o.split("=", 1)[1] == "true"
            elif o.startswith(("--prec-sqrt=", "-prec-sqrt=")):
                flags["prec_sqrt"] = o.split("=", 1)[1] == "true"
            elif o in ("--use_fast_math", "-use_fast_math"):
                flags["fast_math"] = True
            elif o.startswith(("-arch", "--gpu-architecture", "--maxrregcount", "-maxrregcount", "-lineinfo", "--device-as-default-execution-space",
                               "--extra-device-vectorization", "-default-device", "--restrict")):
                continue    # code-generation tuning with no Metal meaning
            else:
                unk.append(o)
        if flags["ftz"] is None:
            # CuPy RawModule/ElementwiseKernel append -ftz=true; the direct NVRTC route does not.
            flags["ftz"] = bool(cupy_raw_module)
        if flags["fast_math"]:
            flags.update(ftz=True, fmad=True, prec_div=False, prec_sqrt=False)
        return cls(tuple(d), tuple(u), tuple(inc), unknown=tuple(unk), **flags)


@functools.lru_cache(maxsize=1)
def _clang():
    exe = os.environ.get("ARWEN_METAL_CLANG") or shutil.which("clang")
    if exe is None:
        exe = subprocess.run(["xcrun", "-f", "clang"], capture_output=True, text=True, check=True).stdout.strip()
    return exe


def preprocess(source: str, opts: NvrtcOptions, filename="kernel.cu") -> str:
    args = [_clang(), "-x", "c++", f"-std={opts.std if opts.std.startswith('c++') else 'c++17'}",
            "-E", "-P", "-nostdinc", "-nostdinc++", "-undef", "-fno-blocks"]
    args += ["-D" + m for m in NVRTC_DEFINES]
    args += ["-D" + m for m in opts.defines] + ["-U" + m for m in opts.undefines]
    args += ["-I" + p for p in opts.includes]
    args += ["-"]
    r = subprocess.run(args, input=source, capture_output=True, text=True)
    if r.returncode != 0:
        raise TranslationError(f"preprocessing {filename} failed:\n{r.stderr[-4000:]}")
    # Builtin names (__global__, threadIdx, FLT_MIN, ...) stay unexpanded tokens:
    # the parser gets their declarations from PARSE_HEADER via -include, and
    # the rewriter maps them to MSL.
    return r.stdout


class TranslationError(RuntimeError):
    """A construct the translator cannot express faithfully in MSL."""


@functools.lru_cache(maxsize=1)
def _index():
    import clang.cindex as ci
    return ci.Index.create()


def _install_cursor_identity():
    """libclang's ``Cursor.hash`` (clang_hashCursor) is 32 bits: a translation unit
    of 60 000-460 000 nodes has colliding hashes (measured: thompson 1, goddard 10,
    cam3_emissivity 44), and the translator keys parents, enclosing functions and
    "already edited" sets by it.  The identity used here adds the node kind and its
    location, which tells colliding nodes apart."""
    import clang.cindex as ci
    if getattr(ci.Cursor, "_arwen_identity", False):
        return

    def key(self):
        k = getattr(self, "_arwen_key", None)
        if k is None:
            # the location, not the extent: a declaration reached through
            # ``referenced`` can report another extent (measured: 101 of 9 273
            # references in thompson), never another location
            loc = ci.conf.lib.clang_getCursorLocation(self)
            k = (ci.conf.lib.clang_hashCursor(self), self._kind_id, loc.offset)
            self._arwen_key = k
        return k
    ci.Cursor.hash = property(key)
    ci.Cursor.__hash__ = lambda self: hash(key(self))
    ci.Cursor._arwen_identity = True


_install_cursor_identity()


def parse(pp_text: str, filename="kernel.cu"):
    """Parse preprocessed CUDA device code; return (translation unit, error list)."""
    import clang.cindex as ci
    args = ["-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib", "--cuda-gpu-arch=sm_90",
            "-std=c++17", "-fno-delayed-template-parsing", "-Wno-everything",
            "-include", str(PARSE_HEADER)]
    tu = _index().parse(filename, args=args, unsaved_files=[(filename, pp_text)],
                        options=0)
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    return tu, errors
