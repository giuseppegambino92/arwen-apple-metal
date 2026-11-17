"""``cupy.cuda`` on Metal: one device, one in-order queue.

Streams map to the single command queue (an in-order queue satisfies every
stream ordering); events record host-side completion points.  NVRTC compile
calls return a token carrying the CUDA source, which ``function.Module.load``
translates and compiles for Metal.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
import types

import numpy as np

from .. import runtime as _rt_module


class CUDARuntimeError(RuntimeError):
    def __init__(self, status=0, msg="Metal backend runtime error"):
        super().__init__(msg)
        self.status = status


class _Limits:
    cudaLimitStackSize = 0
    cudaLimitMallocHeapSize = 2


# ------------------------------------------------------------------ runtime
class _RuntimeNS(types.SimpleNamespace):
    pass


def _props():
    r = _rt_module.rt()
    total = int(r.stats()["recommended_max_working_set"])
    return {"name": r.name.encode(), "totalGlobalMem": total, "multiProcessorCount": 30,
            "maxThreadsPerBlock": 1024, "sharedMemPerBlock": 32768, "sharedMemPerBlockOptin": 32768,
            "sharedMemPerMultiprocessor": 65536, "regsPerBlock": 65536, "warpSize": 32,
            "major": 0, "minor": 0, "clockRate": 1398000, "memoryClockRate": 0, "memoryBusWidth": 512,
            "l2CacheSize": 0, "maxThreadsDim": (1024, 1024, 1024), "maxGridSize": (2**31 - 1, 65535, 65535),
            "integrated": 1, "canMapHostMemory": 1, "pciBusID": 0, "pciDeviceID": 0, "pciDomainID": 0,
            "uuid": b"apple-metal-gpu0", "gcnArchName": b"apple-metal"}


def _mem_info():
    s = _rt_module.rt().stats()
    total = int(s["recommended_max_working_set"])
    used = int(s["current_allocated_size"])
    return max(total - used, 0), total


runtime_ns = _RuntimeNS(
    is_hip=False,
    CUDARuntimeError=CUDARuntimeError,
    getDevice=lambda: 0,
    setDevice=lambda d: None if int(d) == 0 else (_ for _ in ()).throw(CUDARuntimeError(101, "invalid device ordinal")),
    getDeviceCount=lambda: 1,
    getDeviceProperties=lambda d=0: _props(),
    memGetInfo=_mem_info,
    deviceSynchronize=lambda: _rt_module.synchronize(),
    runtimeGetVersion=lambda: 0,
    driverGetVersion=lambda: 0,
    deviceGetPCIBusId=lambda d=0: "0000:00:00.0",
    deviceGetLimit=lambda limit: 0,
    deviceSetLimit=lambda limit, value: None,
    memcpyHostToDevice=1, memcpyDeviceToHost=2, memcpyDeviceToDevice=3, memcpyDefault=4,
    cudaLimitStackSize=0, cudaLimitMallocHeapSize=2,
)


def _memcpy(dst, src, size, kind):
    _rt_module.synchronize()
    import ctypes
    ctypes.memmove(_host_ptr(dst), _host_ptr(src), int(size))


def _host_ptr(p):
    """CPU address for a GPU address produced by this backend (unified memory)."""
    p = int(p)
    from ._core import ndarray  # noqa
    for mem in _rt_module.live_memories():
        if mem.gpu <= p < mem.gpu + mem.nbytes:
            return mem.cpu + (p - mem.gpu)
    return p


runtime_ns.memcpy = _memcpy
runtime_ns.memcpyAsync = lambda dst, src, size, kind, stream=0: _memcpy(dst, src, size, kind)
runtime_ns.memsetAsync = lambda ptr, value, size, stream=0: (_rt_module.synchronize(),
                                                            __import__("ctypes").memset(_host_ptr(ptr), int(value), int(size)))
runtime = runtime_ns      # cupy.cuda.runtime


# ------------------------------------------------------------------ device
class Device:
    def __init__(self, device=None):
        self.id = 0 if device is None else int(device)
        if self.id != 0:
            raise CUDARuntimeError(101, "invalid device ordinal (Metal backend exposes one GPU)")

    compute_capability = "metal"
    pci_bus_id = "0000:00:00.0"

    @property
    def attributes(self):
        return {"MultiProcessorCount": 30, "MaxThreadsPerBlock": 1024, "WarpSize": 32}

    @property
    def mem_info(self):
        return _mem_info()

    def use(self):
        return None

    def synchronize(self):
        _rt_module.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __int__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Device) and other.id == self.id

    def __hash__(self):
        return hash(self.id)


# ------------------------------------------------------------------ streams and events
_tls = threading.local()


class Stream:
    null = None

    def __init__(self, null=False, non_blocking=False, ptds=False, priority=None):
        self.ptr = 0 if null else id(self)
        self.device_id = 0

    def synchronize(self):
        _rt_module.synchronize()

    def query(self):
        return not _rt_module.rt().busy()

    def record(self, event=None):
        event = event or Event()
        event.record(self)
        return event

    def wait_event(self, event):
        return None          # a single in-order queue already orders everything

    def is_capturing(self):
        return False         # no CUDA-graph capture on Metal

    def use(self):
        _tls.stream = self
        return self

    def __enter__(self):
        self._prev = getattr(_tls, "stream", None)
        _tls.stream = self
        return self

    def __exit__(self, *exc):
        _tls.stream = self._prev
        return False

    def launch_host_func(self, fn, arg):
        _rt_module.synchronize()
        fn(arg)


Stream.null = Stream(null=True)
ExternalStream = Stream


def get_current_stream(device_id=None):
    return getattr(_tls, "stream", None) or Stream.null


class Event:
    """Completion points.  Recording flushes queued work; the time of an event is
    the host time at which its work was observed complete (measured by
    ``synchronize``), so elapsed times include queue latency."""

    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self._t = None
        self.done = True

    def record(self, stream=None):
        _rt_module.rt().flush()
        self._t = None
        self.done = False

    def synchronize(self):
        _rt_module.synchronize()
        if self._t is None:
            self._t = time.perf_counter()
        self.done = True

    def query(self):
        if not _rt_module.rt().busy():
            self.synchronize()
        return self.done


def get_elapsed_time(start, end):
    for e in (start, end):
        if e._t is None:
            e.synchronize()
    return (end._t - start._t) * 1000.0


# ------------------------------------------------------------------ compiler / modules
_TOKEN = b"ARWEN-METAL-SOURCE\x00"


def _token(source, options, name_expressions=None):
    return _TOKEN + json.dumps({"source": source, "options": [str(o) for o in options or ()],
                                "name_expressions": list(name_expressions or ())}).encode()


def compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", name_expressions=None, log_stream=None,
                        cache_in_memory=False, jitify=False, method=None):
    """Instead of PTX/cubin: a token carrying the CUDA source for function.Module.load."""
    return _token(source, options, name_expressions), None


def _compile_with_cache_cuda(source, options, arch=None, cache_dir=None, extra_source=None, backend="nvrtc",
                             enable_cooperative_groups=False, name_expressions=None, log_stream=None,
                             cache_in_memory=False, jitify=False):
    m = function.Module()
    m.load(_token(source, options, name_expressions))
    return m


class _NVRTCProgram:
    """Stand-in for CuPy's NVRTC program object.  ``compile`` returns the final
    option list; gpuwm's option hooks (gpuwm.wrf_exact, gpuwm.cuda_subnormals)
    wrap it exactly as they wrap CuPy's, and every Metal compile asks it."""

    def __init__(self, src, name="kern.cu", headers=(), include_names=(), name_expressions=None, method="ptx"):
        self.src, self.name = src, name

    def compile(self, options=(), log_stream=None):
        return tuple(options)


def effective_options(source, options):
    return tuple(compiler._NVRTCProgram(source).compile(tuple(str(o) for o in options or ())))


compiler = types.SimpleNamespace(
    _NVRTCProgram=_NVRTCProgram,
    compile_using_nvrtc=compile_using_nvrtc,
    _compile_with_cache_cuda=_compile_with_cache_cuda,
    _compile_using_nvrtc_no_warning=compile_using_nvrtc,
    _get_arch=lambda: "metal",
    _get_arch_for_options_for_nvrtc=lambda arch=None: ("-arch=metal", "metal"),
    get_cache_dir=lambda: str(__import__("gpuwm.metal.native_build", fromlist=["cache_dir"]).cache_dir()),
    CompileException=RuntimeError,
)


class _Module:
    def __init__(self):
        self._m = None
        self.mapping = {}

    def load(self, image):
        if isinstance(image, str):
            image = image.encode()
        image = bytes(image)
        if not image.startswith(_TOKEN):
            raise RuntimeError("function.Module.load on Metal needs the source token produced by "
                               "compiler.compile_using_nvrtc (a PTX/cubin image cannot run on Metal)")
        d = json.loads(image[len(_TOKEN):].decode())
        self._m = _rt_module.Module(d["source"], effective_options(d["source"], d["options"]), cupy_raw_module=False)
        return self

    def load_file(self, path):
        with open(path, "rb") as f:
            return self.load(f.read())

    def get_function(self, name):
        from ._custom import Function
        return Function(self._m.get_function(self.mapping.get(name, name)))

    def get_global_var(self, name):
        from ._custom import RawModule  # noqa
        mem, g = self._m.get_global(name)
        from ._core import MemoryPointer
        return MemoryPointer(mem, mem.gpu + g["offset"])

    def _set_mapping(self, mapping):
        self.mapping = dict(mapping or {})


function = types.SimpleNamespace(Module=_Module, Function=None)


# ------------------------------------------------------------------ memory
class OutOfMemoryError(MemoryError):
    pass


class _Pool:
    def used_bytes(self):
        return int(_rt_module.rt().stats()["live_bytes"])

    def total_bytes(self):
        return self.used_bytes()

    def free_all_blocks(self, stream=None):
        return None

    def n_free_blocks(self):
        return 0

    def set_limit(self, size=None, fraction=None):
        return None

    def get_limit(self):
        return _mem_info()[1]


_POOL = _Pool()


class PinnedMemoryPointer:
    """Host staging memory.  On unified memory every host buffer is directly
    usable; this keeps CuPy's interface (buffer protocol, .ptr, .size)."""

    def __init__(self, size):
        self._a = np.empty(max(int(size), 1), dtype=np.uint8)
        self.size = int(size)
        self.ptr = self._a.ctypes.data
        self.mem = self

    def __int__(self):
        return self.ptr

    def __len__(self):
        return self.size

    def __buffer__(self, flags):
        return memoryview(self._a)[: self.size]

    def __release_buffer__(self, view):
        pass


def alloc_pinned_memory(size):
    return PinnedMemoryPointer(size)


class UnownedMemory:
    def __init__(self, ptr, size, owner, device_id=None):
        self.ptr, self.size, self.owner = ptr, size, owner


class MemoryPointer:
    def __init__(self, mem, offset):
        self.mem, self.ptr = mem, getattr(mem, "ptr", 0) + offset


memory = types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError, MemoryPool=_Pool, UnownedMemory=UnownedMemory,
                               MemoryPointer=MemoryPointer)
MemoryPool = _Pool
get_allocator = lambda: None  # noqa: E731


@contextlib.contextmanager
def using_allocator(allocator=None):
    yield


nvrtc = types.SimpleNamespace(getVersion=lambda: (0, 0))
driver = types.SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda f, b, s: 1)
