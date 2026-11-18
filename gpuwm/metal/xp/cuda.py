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
    memcpyHostToHost=0, memcpyHostToDevice=1, memcpyDeviceToHost=2, memcpyDeviceToDevice=3, memcpyDefault=4,
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
def _captured(fn):
    def f(*args, **kw):
        if _rt_module.CAPTURE is not None:
            _rt_module.CAPTURE.append(("host", lambda: fn(*args, **kw)))
            return None
        return fn(*args, **kw)
    return f


runtime_ns.memcpyAsync = _captured(lambda dst, src, size, kind, stream=0: _memcpy(dst, src, size, kind))
runtime_ns.memsetAsync = lambda ptr, value, size, stream=0: (_rt_module.synchronize(),
                                                            __import__("ctypes").memset(_host_ptr(ptr), int(value), int(size)))
_PINNED = __import__("weakref").WeakSet()


def _pointer_attributes(ptr):
    """cudaPointerGetAttributes: memory from alloc_pinned_memory is page-locked host
    memory (type 1), a Metal buffer is device memory (type 2, its CPU mapping the
    host pointer: unified memory), anything else is unregistered host memory (0)."""
    p = int(ptr)
    for m in list(_PINNED):
        if m.ptr <= p < m.ptr + max(m.size, 1):
            return types.SimpleNamespace(type=1, device=0, devicePointer=p, hostPointer=p)
    for mem in _rt_module.live_memories():
        if mem.gpu <= p < mem.gpu + mem.nbytes:
            return types.SimpleNamespace(type=2, device=0, devicePointer=p, hostPointer=mem.cpu + (p - mem.gpu))
    return types.SimpleNamespace(type=0, device=-2, devicePointer=0, hostPointer=p)


runtime_ns.pointerGetAttributes = _pointer_attributes


def _memcpy3d(params, stream=0):
    """cudaMemcpy3DAsync on linear pitched memory, reading cudaMemcpy3DParms by its
    ABI (srcArray, srcPos, srcPtr, dstArray, dstPos, dstPtr, extent, kind; size_t
    fields): row by row over unified memory, after the GPU is done."""
    import ctypes
    q = ctypes.cast(int(params), ctypes.POINTER(ctypes.c_size_t * 19)).contents
    src_array, sx, sy, sz, sptr, spitch, _sxs, sysz = q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]
    dst_array, dx, dy, dz, dptr, dpitch, _dxs, dysz = q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15]
    width, height, depth = q[16], q[17], q[18]
    if src_array or dst_array:
        raise NotImplementedError("cudaMemcpy3DAsync with CUDA arrays on the Metal backend")
    _rt_module.synchronize()
    s_base, d_base = _host_ptr(sptr), _host_ptr(dptr)
    for z in range(depth):
        for y in range(height):
            so = s_base + ((sz + z) * sysz + (sy + y)) * spitch + sx
            do = d_base + ((dz + z) * dysz + (dy + y)) * dpitch + dx
            ctypes.memmove(do, so, width)


runtime_ns.memcpy3DAsync = _memcpy3d
runtime_ns.memcpy3D = lambda params: _memcpy3d(params)


class _HostBlock:
    """cudaHostAlloc's page-locked host memory: on unified memory any host memory
    is usable by the GPU path; the block is registered so pointerGetAttributes
    reports it as host (pinned) memory, as CUDA does."""

    def __init__(self, size):
        self._a = np.empty(max(int(size), 1), dtype=np.uint8)
        self.ptr, self.size = self._a.ctypes.data, int(size)


_HOST_BLOCKS = {}


def _host_alloc(size, flags=0):
    b = _HostBlock(size)
    _HOST_BLOCKS[b.ptr] = b
    _PINNED.add(b)
    return b.ptr


def _free_host(ptr):
    _HOST_BLOCKS.pop(int(ptr), None)


runtime_ns.hostAlloc = _host_alloc
runtime_ns.mallocHost = lambda size: _host_alloc(size)
runtime_ns.freeHost = _free_host
runtime_ns.hostAllocDefault, runtime_ns.hostAllocPortable, runtime_ns.hostAllocMapped = 0, 1, 2
runtime_ns.memoryTypeUnregistered, runtime_ns.memoryTypeHost, runtime_ns.memoryTypeDevice = 0, 1, 2
runtime = runtime_ns      # cupy.cuda.runtime


# ------------------------------------------------------------------ device
class Device:
    def __init__(self, device=None):
        self.id = 0 if device is None else int(device)
        if self.id != 0:
            raise CUDARuntimeError(101, "invalid device ordinal (Metal backend exposes one GPU)")

    # CuPy's contract is a digit string; the translator targets __CUDA_ARCH__=1200
    compute_capability = "120"
    pci_bus_id = "0000:00:00.0"

    @property
    def attributes(self):
        """CuPy's device attribute names, with the Apple GPU's values (cores as
        multiprocessors; threadgroup memory as shared memory; one 32-lane SIMD-group
        as a warp; the reported compute capability matches the sm_120 target)."""
        tg = 32768
        try:
            from ..runtime import tg_memory_limit
            tg = tg_memory_limit()
        except Exception:  # noqa: BLE001
            pass
        return {
            "MultiProcessorCount": 30, "MaxThreadsPerBlock": 1024, "MaxThreadsPerMultiProcessor": 1024,
            "MaxBlockDimX": 1024, "MaxBlockDimY": 1024, "MaxBlockDimZ": 64,
            "MaxGridDimX": 2**31 - 1, "MaxGridDimY": 65535, "MaxGridDimZ": 65535, "WarpSize": 32,
            "MaxSharedMemoryPerBlock": tg, "MaxSharedMemoryPerBlockOptin": tg,
            "MaxSharedMemoryPerMultiprocessor": 2 * tg, "MaxRegistersPerBlock": 65536,
            "MaxRegistersPerMultiprocessor": 65536, "TotalConstantMemory": 65536,
            "ComputeCapabilityMajor": 12, "ComputeCapabilityMinor": 0, "Integrated": 1,
            "ConcurrentKernels": 1, "UnifiedAddressing": 1, "ManagedMemory": 1, "PageableMemoryAccess": 1,
            "CanMapHostMemory": 1, "CooperativeLaunch": 0, "L2CacheSize": 0, "ClockRate": 1398000,
            "MemoryClockRate": 0, "GlobalMemoryBusWidth": 512, "AsyncEngineCount": 1, "ECCEnabled": 0,
            "PciBusId": 0, "PciDeviceId": 0, "PciDomainId": 0, "MaxPitch": 2**31 - 1,
            "TextureAlignment": 256, "MultiGpuBoardGroupID": 0, "ComputeMode": 0,
        }

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


class Graph:
    """An instantiated CUDA graph: the captured launches and copies, replayed in order."""

    def __init__(self, ops):
        self._ops = list(ops)

    def upload(self, stream=None):
        return None

    def launch(self, stream=None):
        _rt_module.replay(self._ops)


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
        return _rt_module.CAPTURE is not None and getattr(self, "_capturing", False)

    def begin_capture(self, mode=None):
        """CUDA-graph capture: kernel launches and copies issued from now on are
        recorded with their arguments instead of run (as on CUDA)."""
        if _rt_module.CAPTURE is not None:
            raise CUDARuntimeError(900, "a stream is already capturing")
        _rt_module.CAPTURE = []
        self._capturing = True

    def end_capture(self):
        ops, _rt_module.CAPTURE = _rt_module.CAPTURE or [], None
        self._capturing = False
        return Graph(ops)

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
    # the translator preprocesses with __CUDA_ARCH__=1200 (the reference RTX 5090's
    # sm_120); answer the same so ArWen's architecture audits see a real target
    _get_arch=lambda: "120",
    _get_arch_for_options_for_nvrtc=lambda arch=None: ("-arch=sm_120", "cubin"),
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

    def free_bytes(self):
        """Bytes the pool holds for reuse (retired buffers waiting in size classes)."""
        st = _rt_module.rt().stats()
        return int(st.get("pooled_bytes", 0))

    def total_bytes(self):
        return self.used_bytes()

    def free_all_blocks(self, stream=None):
        _rt_module.rt().pool_release()

    def n_free_blocks(self):
        return int(_rt_module.rt().pool_blocks())

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
        _PINNED.add(self)

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
    """CuPy's view of memory owned elsewhere.  On Metal ``ptr`` may be the
    MemoryPointer a module's get_global_var returned: the buffer it lives in is
    kept, so arrays built on it are real views of the module's globals."""
    def __init__(self, ptr, size, owner, device_id=None):
        from ._core import MemoryPointer as _MP
        self.buffer = ptr.mem if isinstance(ptr, _MP) else getattr(owner, "_gpuwm_mem", None)
        self.ptr = int(ptr.ptr) if isinstance(ptr, _MP) else ptr
        if self.buffer is not None and not (self.buffer.gpu <= int(self.ptr) <= self.buffer.gpu + self.buffer.nbytes):
            raise ValueError("UnownedMemory: the address is not inside its owner's Metal buffer")
        self.size, self.owner = size, owner


def MemoryPointer(mem, offset):
    """``cupy.cuda.MemoryPointer(mem, offset)``: over a Metal buffer (or an
    UnownedMemory wrapping one) it is the array-capable pointer of _core."""
    from ._core import MemoryPointer as _MP
    buf = getattr(mem, "buffer", None) if isinstance(mem, UnownedMemory) else mem
    if buf is None or not hasattr(buf, "gpu"):
        raise TypeError("cupy.cuda.MemoryPointer on Metal needs memory backed by a Metal buffer")
    base = mem.ptr if isinstance(mem, UnownedMemory) else buf.gpu
    return _MP(buf, int(base) + int(offset))


memory = types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError, MemoryPool=_Pool, UnownedMemory=UnownedMemory,
                               MemoryPointer=MemoryPointer)
MemoryPool = _Pool
get_allocator = lambda: None  # noqa: E731


@contextlib.contextmanager
def using_allocator(allocator=None):
    yield


nvrtc = types.SimpleNamespace(getVersion=lambda: (0, 0))
driver = types.SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda f, b, s: 1)
