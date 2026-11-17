"""Python side of the Metal runtime: device memory, module cache, kernel launch.

Memory is unified: a device array's bytes are an MTLBuffer in shared storage,
visible to the CPU at ``cpu_address`` and to kernels at ``gpu_address``.
Host access to device data only needs the GPU work queued so far to finish
(``synchronize``); nothing is copied.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import struct
import threading
from pathlib import Path

import numpy as np

from . import transpile
from .msl_compile import compile_msl, toolchain_identity
from .native_build import cache_dir, load as _load_native

_lock = threading.RLock()


@functools.lru_cache(maxsize=1)
def native():
    return _load_native()


_RT = None


def rt():
    global _RT
    if _RT is None:
        with _lock:
            if _RT is None:
                _RT = native().Runtime()
                _RT.batch = int(os.environ.get("ARWEN_METAL_BATCH", "256"))
    return _RT


# --------------------------------------------------------------------- memory
class DeviceMemory:
    """An MTLBuffer exporting its bytes through the buffer protocol."""
    __slots__ = ("buf", "cpu", "gpu", "nbytes", "_view", "__weakref__")

    def __init__(self, nbytes: int):
        self.buf = rt().alloc(int(nbytes))
        self.cpu = self.buf.cpu_address
        self.gpu = self.buf.gpu_address
        self.nbytes = int(nbytes)
        self._view = (ctypes.c_char * max(self.nbytes, 1)).from_address(self.cpu)

    def __buffer__(self, flags):
        return memoryview(self._view).cast("B")

    def __release_buffer__(self, view):
        pass


def synchronize():
    rt().synchronize()


# --------------------------------------------------------------------- modules
def _key(source, options, f64_mode, raw):
    return transpile.cache_key(source, options, f64_mode, raw) + "-" + hashlib.sha256(toolchain_identity().encode()).hexdigest()[:8]


class KernelError(RuntimeError):
    pass


class Module:
    """A translated, compiled and loaded CUDA module."""

    def __init__(self, source: str, options=(), *, f64_mode: str | None = None, cupy_raw_module=False,
                 filename="kernel.cu", name_expressions=None):
        f64_mode = f64_mode or os.environ.get("ARWEN_METAL_F64", "sf64")
        key = _key(source, options, f64_mode, cupy_raw_module)
        d = cache_dir() / "modules"
        d.mkdir(parents=True, exist_ok=True)
        lib, meta = d / f"{key}.metallib", d / f"{key}.json"
        if not (lib.exists() and meta.exists()):
            t = transpile.translate(source, options, f64_mode=f64_mode, filename=filename, cupy_raw_module=cupy_raw_module)
            (d / f"{key}.metal").write_text(t.msl)
            compile_msl(t.msl, t.metal_flags, out=lib)
            tmp = meta.with_suffix(".tmp")
            tmp.write_text(json.dumps(t.metadata()))
            os.replace(tmp, meta)
        self.key = key
        self.meta = json.loads(meta.read_text())
        self.library = rt().load_library(str(lib))
        self.f64_mode = f64_mode
        self._pipelines = {}
        self.globals = None
        gsize = self.meta.get("globals_size", 0)
        if gsize:
            self.globals = DeviceMemory(gsize)
            ctypes.memset(self.globals.cpu, 0, gsize)          # CUDA zero-initialises __device__ data
            if "__arwen_init_globals" in self.library.function_names():
                p = self.library.pipeline("__arwen_init_globals")
                rt().dispatch(p, b"", _launch(1, 1, 1, 1, 1, 1, 1), self.globals.buf, 1, 1, 1, 1, 1, 1, 0)

    def pipeline(self, name):
        p = self._pipelines.get(name)
        if p is None:
            p = self._pipelines[name] = self.library.pipeline(name)
        return p

    def get_function(self, name):
        if name not in self.meta["kernels"]:
            raise KernelError(f"no kernel {name!r} in module (has {sorted(self.meta['kernels'])[:20]})")
        return Kernel(self, name)

    def get_global(self, name):
        g = self.meta["globals"].get(name)
        if g is None:
            raise KernelError(f"{name!r} is not a mutable device global of this module "
                              "(constant-initialised tables are compiled into MSL constant data)")
        return self.globals, g


def _launch(gx, gy, gz, bx, by, bz, split):
    return struct.pack("<8I", gx, gy, gz, bx, by, bz, split, 0)


_FMT = {"f": "<f", "d": "<d", "i": "<i", "I": "<I", "q": "<q", "Q": "<Q", "h": "<h", "H": "<H",
        "b": "<b", "B": "<B", "?": "<?"}


def device_address(a) -> int:
    """GPU address of the first element of a device array (or 0 for None)."""
    if a is None:
        return 0
    mem = getattr(a, "_gpuwm_mem", None)
    if mem is None:
        if isinstance(a, int):
            return a
        raise KernelError(f"kernel argument of type {type(a).__name__} is not a Metal device array")
    return mem.gpu + (a.__array_interface__["data"][0] - mem.cpu)


class Kernel:
    def __init__(self, module: Module, name: str):
        self.module = module
        self.name = name
        self.info = module.meta["kernels"][name]
        self.pso = module.pipeline(name)
        self.max_threads = self.pso.max_threads
        self.splittable = not (self.info["uses_barrier"] or self.info["uses_shared"] or self.info["uses_warp"]
                               or self.info["dyn_shared"])
        self._layout = self.info["params"]
        self._size = self.info["args_size"]

    def pack(self, args) -> bytes:
        params = self._layout
        if len(args) != len(params):
            raise KernelError(f"{self.name}: {len(args)} arguments for {len(params)} parameters")
        buf = bytearray(self._size or 16)
        for p, a in zip(params, args):
            k, off = p["kind"], p["offset"]
            if k == "ptr":
                struct.pack_into("<Q", buf, off, device_address(a))
            elif k == "scalar":
                code = p["code"]
                if isinstance(a, np.generic):
                    a = a.item()
                if code in "fd":
                    struct.pack_into(_FMT[code], buf, off, float(a))
                elif code == "?":
                    struct.pack_into("<?", buf, off, bool(a))
                else:
                    struct.pack_into(_FMT[code], buf, off, int(a))
            else:
                raw = bytes(np.asarray(a).tobytes()) if not isinstance(a, (bytes, bytearray)) else bytes(a)
                if len(raw) != p["size"]:
                    raise KernelError(f"{self.name}: struct argument {p['name']} has {len(raw)} bytes, expected {p['size']}")
                buf[off:off + len(raw)] = raw
        return bytes(buf)

    def launch(self, grid, block, args, shared_mem=0):
        grid = tuple(int(x) for x in (tuple(grid) + (1, 1, 1))[:3])
        block = tuple(int(x) for x in (tuple(block) + (1, 1, 1))[:3])
        nthreads = block[0] * block[1] * block[2]
        split = 1
        gx, gy, gz = grid
        bx, by, bz = block
        if nthreads > self.max_threads:
            if not self.splittable or by * bz > self.max_threads:
                raise KernelError(f"{self.name}: block {block} exceeds this pipeline's {self.max_threads} threads "
                                  "and the kernel synchronises its block; it cannot be split")
            split = -(-nthreads // self.max_threads)
            while bx % split:
                split += 1
            bx //= split
            gx *= split
        g = self.module.globals.buf if self.module.globals is not None else None
        rt().dispatch(self.pso, self.pack(args), _launch(grid[0], grid[1], grid[2], block[0], block[1], block[2], split),
                      g, gx, gy, gz, bx, by, bz, int(shared_mem))

    __call__ = launch
