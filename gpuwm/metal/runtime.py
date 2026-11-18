"""Python side of the Metal runtime: device memory, module cache, kernel launch.

Memory is unified: a device array's bytes are an MTLBuffer in shared storage,
visible to the CPU at ``cpu_address`` and to kernels at ``gpu_address``.
Host access to device data only needs the GPU work queued so far to finish
(``synchronize``); nothing is copied.
"""
from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import json
import os
import re
import struct
import sys
import contextlib
import threading
import time
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
                _RT.eager = int(os.environ.get("ARWEN_METAL_EAGER", "16"))
                _RT.pool_cap = int(os.environ.get("ARWEN_METAL_POOL_MB", "4096")) << 20
                if os.environ.get("ARWEN_METAL_ORDER"):           # diagnostic: 0 MTLEvent, 1 MTLSharedEvent, 2 none
                    _RT.set_order_mode(int(os.environ["ARWEN_METAL_ORDER"]))
                if os.environ.get("ARWEN_METAL_TIMELINE"):
                    _RT.timeline = True
                    atexit.register(_write_timeline)
                if os.environ.get("ARWEN_METAL_STATS"):
                    atexit.register(lambda: print("[arwen-metal] runtime stats:", dict(_RT.stats()), file=sys.stderr))
                _RT.flush_on_alloc = os.environ.get("ARWEN_METAL_FLUSH_ON_ALLOC", "") == "1"
                _RT.concurrent = os.environ.get("ARWEN_METAL_CONCURRENT", "") == "1"
                if os.environ.get("ARWEN_METAL_PROFILE"):
                    _RT.profile = True
                    atexit.register(_write_profile)
    return _RT


# --------------------------------------------------------------------- memory
import weakref

_LIVE = weakref.WeakSet()


def live_memories():
    return list(_LIVE)


class DeviceMemory:
    """An MTLBuffer exporting its bytes through the buffer protocol."""
    __slots__ = ("buf", "cpu", "gpu", "nbytes", "_view", "__weakref__")

    @property
    def ptr(self) -> int:
        """CuPy's ``Memory.ptr``: the allocation's base GPU address."""
        return self.gpu

    @property
    def size(self) -> int:
        """CuPy's ``Memory.size``: the allocation's byte count."""
        return self.nbytes

    def __init__(self, nbytes: int):
        self.buf = rt().alloc(int(nbytes))
        self.cpu = self.buf.cpu_address
        self.gpu = self.buf.gpu_address
        self.nbytes = int(nbytes)
        self._view = (ctypes.c_char * max(self.nbytes, 1)).from_address(self.cpu)
        _LIVE.add(self)

    def __buffer__(self, flags):
        return memoryview(self._view).cast("B")

    def __release_buffer__(self, view):
        pass


_PROFILE = {}
_KTAGS = {}        # kernel name -> numeric flags of its module(s), for the profile


def _collect_profile():
    for name, sec in _RT.profile_drain():
        t = _PROFILE.setdefault(name, [0, 0.0])
        t[0] += 1
        t[1] += sec


def _write_profile():
    """ARWEN_METAL_PROFILE=<file>: GPU seconds per kernel name (one dispatch per command buffer)."""
    try:
        _RT.synchronize()
    except Exception:
        pass
    _collect_profile()
    rows = sorted(_PROFILE.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for v in _PROFILE.values()) or 1.0
    path = os.environ["ARWEN_METAL_PROFILE"].replace("{pid}", str(os.getpid()))
    with open(path, "w") as f:
        f.write(f"# total GPU seconds {total:.3f} over {sum(v[0] for v in _PROFILE.values())} dispatches\n")
        for name, (n, sec) in rows:
            tags = ",".join(sorted(_KTAGS.get(name, ())))
            f.write(f"{sec:10.4f} s {100*sec/total:6.2f}% {n:8d} calls {1e6*sec/max(n,1):10.1f} us/call  {name}  [{tags}]\n")


def _write_timeline():
    """ARWEN_METAL_TIMELINE=<file.npz>: command-buffer GPU intervals and host synchronize() intervals."""
    try:
        _RT.synchronize()
    except Exception:
        pass
    g, sy, now = _RT.timeline_drain()
    path = os.environ["ARWEN_METAL_TIMELINE"].replace("{pid}", str(os.getpid()))
    np.savez(path, gpu=np.asarray(g).reshape(-1, 2), sync=np.asarray(sy).reshape(-1, 2), end=now)


_pending = False


_SYNC_TRACE = None


def _trace_sync():
    """ARWEN_METAL_SYNC_TRACE=1: count host synchronizations by gpuwm call site (printed at exit)."""
    global _SYNC_TRACE
    import collections
    import traceback
    if _SYNC_TRACE is None:
        _SYNC_TRACE = collections.Counter()

        def report():
            print("[arwen-metal] host synchronizations by call site:", file=sys.stderr)
            for site, n in _SYNC_TRACE.most_common(25):
                print(f"  {n:6d}  {site}", file=sys.stderr)
        atexit.register(report)
    frames = [f for f in traceback.extract_stack()[:-2] if "/gpuwm/" in f.filename and "/gpuwm/metal/" not in f.filename]
    site = " <- ".join(f"{f.filename.split('/gpuwm/')[-1]}:{f.lineno}" for f in frames[-3:][::-1]) or "?"
    _SYNC_TRACE[site] += 1


def synchronize():
    """Wait for all queued GPU work."""
    global _pending
    if _pending and os.environ.get("ARWEN_METAL_SYNC_TRACE") == "1":
        _trace_sync()
    rt().synchronize()
    _pending = False
    if rt().status_take():
        raise RuntimeError("metal: a kernel executed a device-side trap (CUDA __trap() / asm(\"trap;\")) "
                           "since the last synchronization: its results are invalid")
    if _PROFILE is not None and getattr(_RT, "profile", False):
        _collect_profile()


def host_sync():
    """Before the CPU touches device memory: wait only if GPU work is queued."""
    if _pending:
        synchronize()


# --------------------------------------------------------------------- modules
def tg_memory_limit() -> int:
    """MTLDevice.maxThreadgroupMemoryLength (32 KiB on Apple GPU families 7-9)."""
    global _TG_LIMIT
    if _TG_LIMIT is None:
        _TG_LIMIT = int(os.environ.get("ARWEN_METAL_TG_LIMIT") or rt().max_threadgroup_memory())
    return _TG_LIMIT


_TG_LIMIT = None

#: the work being captured into a CUDA graph (gpuwm.metal.xp.cuda.Stream.begin_capture)
CAPTURE = None


def replay(ops):
    """Run captured work in capture order."""
    global _pending
    for op in ops:
        if op[0] == "dispatch":
            _pending = True
            rt().dispatch(*op[1])
        else:
            synchronize()            # a host step reads what the GPU work before it wrote
            op[1]()


_SLOW = None


def _slow_pipelines() -> set:
    global _SLOW
    if _SLOW is None:
        try:
            _SLOW = set(json.loads((cache_dir() / "slow_pipelines.json").read_text()))
        except (OSError, ValueError):
            _SLOW = set()
    return _SLOW


# kernels whose -O3 back-end compile exceeded the budget on the development machine
# (M3 Max, macOS 27.2): built from -Os AIR straight away, every variant of the name
_SEED_SLOW_KERNELS = frozenset({"thompson_fork_column", "cam_driver_stage", "nssl2_nucond_default",
                                "flg_stream_driver"})


#: names every module shares: remembered per module (ident), never by name
_PER_MODULE_ONLY = frozenset({"__arwen_init_globals"})


def _slow_kernel_names() -> set:
    return ({i.rsplit(":", 1)[1] for i in _slow_pipelines()} - _PER_MODULE_ONLY) | _SEED_SLOW_KERNELS


def _remember_slow(ident: str) -> None:
    s = _slow_pipelines()
    s.add(ident)
    path = cache_dir() / "slow_pipelines.json"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(sorted(s)))
        os.replace(tmp, path)
    except OSError:
        pass


def _key(source, options, f64_mode, raw):
    return transpile.cache_key(source, options, f64_mode, raw) + "-" + hashlib.sha256(toolchain_identity().encode()).hexdigest()[:8]


class KernelError(RuntimeError):
    pass


#: libraries above this size create their pipelines one process at a time: Apple's
#: compiler service drops its XPC connection when several processes back-end
#: compile very large libraries at once (Goddard, under a parallel test run)
HEAVY_LIBRARY_BYTES = int(os.environ.get("ARWEN_METAL_HEAVY_LIBRARY", str(1 << 20)))


@contextlib.contextmanager
def _heavy_compile_lock(nbytes):
    if nbytes < HEAVY_LIBRARY_BYTES:
        yield
        return
    import fcntl
    path = cache_dir() / "pipeline.lock"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _xpc_retry(make, attempts=3):
    """Apple's compiler service (MTLCompilerService) can drop its XPC connection
    under memory pressure; the request itself is fine and is sent again."""
    for i in range(attempts):
        try:
            return make()
        except RuntimeError as e:
            if "XPC_ERROR" not in str(e) or i == attempts - 1:
                raise
            print(f"[arwen-metal] Metal compiler service interrupted; retrying ({i + 1})", file=sys.stderr)
            time.sleep(5 * (i + 1))


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
            msl_tmp = d / f"{key}.{os.getpid()}.{threading.get_ident()}.metal.tmp"
            msl_tmp.write_text(t.msl)
            os.replace(msl_tmp, d / f"{key}.metal")
            compile_msl(t.msl, t.metal_flags, out=lib)
            tmp = meta.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")   # concurrent builders
            tmp.write_text(json.dumps(t.metadata()))
            os.replace(tmp, meta)
        self.key = key
        self.lib_bytes = lib.stat().st_size
        self.meta = json.loads(meta.read_text())
        if _PROFILE is not None and os.environ.get("ARWEN_METAL_PROFILE"):
            from .frontend import NvrtcOptions
            o = NvrtcOptions.parse(options, cupy_raw_module=cupy_raw_module)
            tag = ("ftz" if o.ftz else "denorm") + ("+f64" if re.search(r"\bdouble\b", source) else "")
            for k in self.meta["kernels"]:
                _KTAGS.setdefault(k, set()).add(tag)
        self.library = rt().load_library(str(lib))
        self.has_heap = "__arwen_heap_reset" in self.meta["kernels"]
        self.f64_mode = f64_mode
        self._pipelines = {}
        self.globals = None
        gsize = self.meta.get("globals_size", 0)
        if gsize:
            self.globals = DeviceMemory(gsize)
            ctypes.memset(self.globals.cpu, 0, gsize)          # CUDA zero-initialises __device__ data
            if "__arwen_init_globals" in self.library.function_names():
                p = self.pipeline("__arwen_init_globals")       # same -O3 / -Os policy as kernels
                global _pending
                _pending = True
                rt().dispatch(p, b"", _launch(1, 1, 1, 1, 1, 1, 1), self.globals.buf, 1, 1, 1, 1, 1, 1, 0)

    def pipeline(self, name):
        p = self._pipelines.get(name)
        if p is None:
            with _heavy_compile_lock(self.lib_bytes):
                p = self._pipelines[name] = self._make_pipeline(name)
        return p

    def _make_pipeline(self, name):
        """Pipeline for kernel ``name``.  Apple's GPU back end compiles the -O3 AIR
        when the pipeline is created, and for a few very large kernels that takes
        tens of minutes or fails (XPC interruption); the -Os AIR of the same MSL
        compiles in seconds and computes the same words (checked on a 1 h
        forecast).  So: -O3 within a time budget, else the -Os build, and the
        kernel is remembered so later processes go straight to it."""
        from .msl_compile import MSL_OPT
        ident = f"{self.key}:{name}"
        if MSL_OPT == "-Os":
            return _xpc_retry(lambda: self.library.pipeline(name))
        if ident in _slow_pipelines() or name in _slow_kernel_names():
            return _xpc_retry(lambda: self._os_library().pipeline(name))
        budget = float(os.environ.get("ARWEN_METAL_PIPELINE_BUDGET", "90"))
        result = {}

        def build():
            try:
                result["p"] = self.library.pipeline(name)
            except Exception as e:            # noqa: BLE001 - reported through the fallback
                result["e"] = e
        th = threading.Thread(target=build, name=f"pso-{name}", daemon=True)
        th.start()
        th.join(budget)
        if "p" in result:
            return result["p"]
        why = f"failed ({result['e']})" if "e" in result else f"exceeded {budget:.0f} s"
        print(f"[arwen-metal] pipeline {name}: -O3 back-end compile {why}; using the -Os build", file=sys.stderr)
        _remember_slow(ident)
        return _xpc_retry(lambda: self._os_library().pipeline(name))

    def _os_library(self):
        lib = getattr(self, "_lib_os", None)
        if lib is None:
            d = cache_dir() / "modules"
            path = d / f"{self.key}-Os.metallib"
            if not path.exists():
                compile_msl((d / f"{self.key}.metal").read_text(), tuple(self.meta["metal_flags"]), out=path, opt="-Os")
            lib = self._lib_os = rt().load_library(str(path))
        return lib

    @property
    def heap_reset(self):
        k = getattr(self, "_heap_reset", None)
        if k is None:
            k = self._heap_reset = Kernel(self, "__arwen_heap_reset")
        return k

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
    try:
        return a._ga                   # cached: a device array's data pointer never changes
    except AttributeError:
        pass
    mem = getattr(a, "_gpuwm_mem", None)
    if mem is None:
        if isinstance(a, (int, np.integer)):
            return int(a)              # a GPU address carried as an integer (pointer tables)
        raise KernelError(f"kernel argument of type {type(a).__name__} is not a Metal device array")
    ga = mem.gpu + (a._v.__array_interface__["data"][0] - mem.cpu)
    try:
        a._ga = ga
    except AttributeError:
        pass
    return ga


class Kernel:
    def __init__(self, module: Module, name: str):
        self.module = module
        self.name = name
        self.info = module.meta["kernels"][name]
        self.pso = module.pipeline(name)
        self.max_threads = self.pso.max_threads
        self.static_tg = int(self.pso.static_threadgroup_memory)
        self.splittable = not (self.info["uses_barrier"] or self.info["uses_shared"] or self.info["uses_warp"]
                               or self.info["dyn_shared"])
        self._layout = self.info["params"]
        self._size = self.info["args_size"]

    def _packer(self):
        """One struct.Struct for the whole argument block (None if a parameter kind needs the slow path)."""
        fmt, pos, kinds = "<", 0, []
        for p in sorted(self._layout, key=lambda p: p["offset"]):
            off = p["offset"]
            if off < pos:
                return None
            fmt += "x" * (off - pos)
            if p["kind"] == "ptr":
                fmt += "Q"; pos = off + 8; kinds.append(0)
            elif p["kind"] == "scalar" and p["code"] in _FMT:
                c = _FMT[p["code"]][1]
                fmt += c; pos = off + struct.calcsize("<" + c)
                kinds.append(1 if p["code"] in "fd" else (2 if p["code"] == "?" else 3))
            elif p["kind"] == "struct":
                fmt += f"{p['size']}s"; pos = off + p["size"]; kinds.append(4)
            else:
                return None
        if [p["offset"] for p in self._layout] != sorted(p["offset"] for p in self._layout):
            return None
        fmt += "x" * ((self._size or 16) - pos)
        return struct.Struct(fmt), kinds

    def pack(self, args) -> bytes:
        pk = self.__dict__.get("_pk", False)
        if pk is False:
            pk = self._pk = self._packer()
        if pk is not None and len(args) == len(pk[1]):
            vals = []
            try:
                for k, a in zip(pk[1], args):
                    if k == 0:
                        vals.append(device_address(a) if a is not None else 0)
                    elif k == 1:
                        vals.append(float(a))
                    elif k == 2:
                        vals.append(bool(a))
                    elif k == 3:
                        vals.append(int(a))
                    else:
                        raw = a if isinstance(a, (bytes, bytearray)) else bytes(np.asarray(a).tobytes())
                        vals.append(bytes(raw))
                return pk[0].pack(*vals)
            except struct.error:
                pass                         # wrong struct size etc.: the slow path reports it
        return self._pack_slow(args)

    def _pack_slow(self, args) -> bytes:
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
        shared_mem = int(shared_mem)
        if self.static_tg + shared_mem > tg_memory_limit():
            # CUDA refuses such a launch ("too many resources requested"); Metal would
            # run it with threadgroup memory cut short and compute garbage silently
            raise KernelError(f"{self.name}: {self.static_tg} B static + {shared_mem} B dynamic threadgroup memory "
                              f"exceeds this GPU's {tg_memory_limit()} B per threadgroup")
        global _pending
        check = bool(_NANCHECK)
        if check:
            internal = self.name in ("ew", "ek", "rk", "red_thread", "red_block", "scan_rows", "gather_axis",
                                     "flag_count", "scan_counts", "scatter_idx", "scatter_rows")
            before = _nan_counts(args, nan_only=internal)
        g = self.module.globals.buf if self.module.globals is not None else None
        if self.module.has_heap and self.name != "__arwen_heap_reset":
            # CUDA device heap: every launch of the module starts from an empty heap
            self.module.heap_reset.launch((1, 1, 1), (1, 1, 1), ())
        call = (self.pso, self.pack(args), _launch(grid[0], grid[1], grid[2], block[0], block[1], block[2], split),
                g, gx, gy, gz, bx, by, bz, int(shared_mem))
        if CAPTURE is not None:
            # CUDA-graph capture: recorded with its arguments, run by Graph.launch
            CAPTURE.append(("dispatch", call, self, args))   # args keep the operands' memory alive
            return
        _pending = True
        rt().dispatch(*call)
        if _SYNC_EACH:
            synchronize()
        if check:
            after = _nan_counts(args, nan_only=internal)
            for i, (b0, a0) in enumerate(zip(before, after)):
                if a0 > b0:
                    import sys
                    p = self._layout[i]["name"] if i < len(self._layout) else str(i)
                    msg = (f"[arwen-metal NANCHECK] kernel {self.name} (module {self.module.key[:12]}) "
                           f"raised the non-finite count of argument {i} ({p}) from {b0} to {a0}; grid {grid} block {block}\n"
                           + "\n".join(f"    arg {j}: {_summary(x)}" for j, x in enumerate(args)))
                    print(msg, file=sys.stderr, flush=True)
                    dump = os.environ.get("ARWEN_METAL_NANCHECK_DUMP")
                    if dump:
                        np.savez(dump, kernel=self.name, module=self.module.key, grid=np.array(grid), block=np.array(block),
                                 **{f"arg{j}": (x._v.copy() if getattr(x, "_gpuwm_mem", None) is not None
                                                else np.asarray(x)) for j, x in enumerate(args)
                                    if not isinstance(x, (bytes, bytearray))})
                    if _NANCHECK == "raise":
                        raise FloatingPointError(msg)

    __call__ = launch


_NANCHECK = os.environ.get("ARWEN_METAL_NANCHECK", "")
_SYNC_EACH = os.environ.get("ARWEN_METAL_SYNC_EACH", "") == "1"   # debugging: wait for every dispatch


def _summary(a):
    if getattr(a, "_gpuwm_mem", None) is None:
        return repr(a)[:60] if not isinstance(a, (bytes, bytearray)) else f"<{len(a)} bytes>"
    v = a._v
    if v.dtype.kind != "f" or v.size == 0:
        return f"{v.dtype}{v.shape}"
    fin = np.isfinite(v)
    return (f"{v.dtype}{v.shape} nan={int(np.isnan(v).sum())} inf={int(np.isinf(v).sum())} "
            f"min={float(v[fin].min()) if fin.any() else None} max={float(v[fin].max()) if fin.any() else None}")


def _nan_counts(args, nan_only=False):
    out = []
    synchronize()
    for a in args:
        mem = getattr(a, "_gpuwm_mem", None)
        if mem is not None and getattr(a, "dtype", None) is not None and a.dtype.kind == "f":
            # whole underlying buffer: kernels address memory beyond the view's extent
            whole = np.frombuffer(mem, dtype=a.dtype, count=mem.nbytes // a.dtype.itemsize)
            out.append(int(np.count_nonzero(np.isnan(whole) if nan_only else ~np.isfinite(whole))))
        else:
            out.append(0)
    return out
