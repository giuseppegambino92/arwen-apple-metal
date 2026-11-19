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
import os
import types
import weakref

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
            "maxThreadsPerBlock": 1024, "maxThreadsPerMultiProcessor": 1024, "maxBlocksPerMultiProcessor": 32,
            "concurrentKernels": 1, "asyncEngineCount": 0, "unifiedAddressing": 1, "managedMemory": 1,
            "ECCEnabled": 0, "computeMode": 0, "concurrentManagedAccess": 1, "pageableMemoryAccess": 1, "totalConstMem": 65536, "sharedMemPerBlock": 32768, "sharedMemPerBlockOptin": 32768,
            "sharedMemPerMultiprocessor": 65536, "regsPerBlock": 65536, "warpSize": 32,
            "major": 12, "minor": 0, "clockRate": 1398000, "memoryClockRate": 0, "memoryBusWidth": 512,
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
            return types.SimpleNamespace(type=1, device=0, devicePointer=m._dm.gpu + (p - m.ptr), hostPointer=p)
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


def _memcpy2d(dst, dpitch, src, spitch, width, height, kind, stream=0):
    """cudaMemcpy2DAsync over unified memory, after the GPU's queued work."""
    import ctypes
    _rt_module.synchronize()
    d, sp = _host_ptr(int(dst)), _host_ptr(int(src))
    for row in range(int(height)):
        ctypes.memmove(d + row * int(dpitch), sp + row * int(spitch), int(width))


runtime_ns.memcpy2DAsync = _captured(_memcpy2d)
runtime_ns.memcpy2D = lambda dst, dpitch, src, spitch, width, height, kind: _memcpy2d(
    dst, dpitch, src, spitch, width, height, kind)
def _launch_host_func(stream, callback, arg):
    """cudaLaunchHostFunc: ``callback(arg)`` after the work queued before it (in
    capture order when a CUDA graph is being captured)."""
    if _rt_module.CAPTURE is not None:
        _rt_module.CAPTURE.append(("host", lambda: callback(arg)))
        return
    _rt_module.synchronize()
    callback(arg)


runtime_ns.launchHostFunc = _launch_host_func
runtime_ns.memcpy3DAsync = _memcpy3d
runtime_ns.memcpy3D = lambda params: _memcpy3d(params)


class _HostBlock:
    """cudaHostAlloc's page-locked host memory: on unified memory any host memory
    is usable by the GPU path; the block is registered so pointerGetAttributes
    reports it as host (pinned) memory, as CUDA does."""

    def __init__(self, size):
        self._dm = _rt_module.DeviceMemory(max(int(size), 1))   # shared storage: GPU-mapped as well
        self._a = np.frombuffer(self._dm, dtype=np.uint8)
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


_GRAPHS = weakref.WeakValueDictionary()
_EVENTS = weakref.WeakValueDictionary()


class Graph:
    """An instantiated CUDA graph: the captured launches and copies, replayed in order."""

    def __init__(self, ops):
        self._ops = list(ops)
        _GRAPHS[id(self)] = self

    @property
    def graphExec(self):
        """The handle native code passes to cuGraphLaunch."""
        return id(self)

    def upload(self, stream=None):
        return None

    def launch(self, stream=None):
        _rt_module.replay(self._ops)


class Stream:
    null = None

    def __init__(self, null=False, non_blocking=False, ptds=False, priority=None):
        self.ptr = 0 if null else id(self)
        self.device_id = 0
        self.priority = int(priority or 0)     # one Metal queue: priorities are recorded, not enforced

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
        _rt_module.set_capture([])
        self._capturing = True

    def end_capture(self):
        ops = _rt_module.CAPTURE or []
        _rt_module.set_capture(None)
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
        self._done = True

    @property
    def ptr(self):
        _EVENTS[id(self)] = self
        return id(self)

    def _mark(self):
        self._t = time.perf_counter()
        self._done = True
        self._ticket = 0

    def record(self, stream=None):
        if _rt_module.CAPTURE is not None:
            # an event node of the graph: marked when the replay reaches it
            self._done = False
            _rt_module.CAPTURE.append(("host", self._mark))
            return
        r = _rt_module.rt()
        r.flush()
        self._t = None
        self._ticket = r.committed()          # the work recorded so far ends with this command buffer
        self._done = False

    def synchronize(self):
        _rt_module.synchronize()
        if self._t is None:
            self._t = time.perf_counter()
        self._done = True

    def query(self):
        if not self._done:
            r = _rt_module.rt()
            if r.completed() >= getattr(self, "_ticket", 0):
                self._done = True
                if self._t is None:
                    self._t = time.perf_counter()
        return self._done

    @property
    def done(self):
        """CuPy's ``Event.done`` queries the event (cudaEventQuery)."""
        return self.query()


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
    def __init__(self, allocator=None):
        self._allocator = allocator

    def used_bytes(self):
        """Bytes in use, counted as CuPy's pool counts them: every live allocation
        rounded up to its 512-byte unit (ArWen's memory plans are written in it)."""
        return int(sum((int(m.nbytes) + 511) // 512 * 512 for m in _rt_module.live_memories()
                       if m not in _rt_module.MODULE_GLOBALS))

    def free_bytes(self):
        """Bytes the pool holds for reuse (retired buffers waiting in size classes)."""
        st = _rt_module.rt().stats()
        return int(st.get("pooled_bytes", 0))

    def total_bytes(self):
        return self.used_bytes()

    def malloc(self, size):
        """MemoryPool.malloc: a MemoryPointer to a new allocation (from the pool's
        own allocator when it was built with one, as MemoryPool(allocator))."""
        if self._allocator is not None:
            return self._allocator(int(size))
        from ._core import MemoryPointer as _MPtr
        mem = _rt_module.DeviceMemory(max(int(size), 1))
        return _MPtr(mem, mem.gpu)

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
        # a shared-storage Metal buffer: on Apple silicon "pinned host memory" is
        # device-accessible as is (CUDA's mapped pinned memory), at its GPU address
        self._dm = _rt_module.DeviceMemory(max(int(size), 1))
        self._a = np.frombuffer(self._dm, dtype=np.uint8)
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
        if self.buffer is None:
            # a GPU address inside a live Metal buffer (mapped pinned memory, a pool block, ...)
            p = int(self.ptr)
            self.buffer = next((m for m in _rt_module.live_memories() if m.gpu <= p < m.gpu + max(m.nbytes, 1)), None)
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
                               MemoryPointer=MemoryPointer, alloc=lambda size: _POOL.malloc(size))
alloc = memory.alloc
MemoryPool = _Pool
class MemoryHook:
    """cupy.cuda.MemoryHook: ``with hook:`` calls its alloc/malloc pre- and
    post-processing methods around every device-array allocation of this thread
    (free hooks are not called: Metal buffers are released by the runtime)."""
    name = "MemoryHook"

    def __enter__(self):
        import importlib
        core = importlib.import_module("gpuwm.metal.xp._core")
        active = getattr(core.HOOKS, "active", None)
        if active is None:
            active = core.HOOKS.active = []
        active.append(self)
        return self

    def __exit__(self, *exc):
        import importlib
        core = importlib.import_module("gpuwm.metal.xp._core")
        core.HOOKS.active.remove(self)
        return False

    def alloc_preprocess(self, **kw): pass
    def alloc_postprocess(self, **kw): pass
    def malloc_preprocess(self, **kw): pass
    def malloc_postprocess(self, **kw): pass
    def free_preprocess(self, **kw): pass
    def free_postprocess(self, **kw): pass


def get_allocator():
    """The allocator new arrays use: using_allocator's, or the default pool's malloc."""
    import importlib
    core = importlib.import_module("gpuwm.metal.xp._core")
    return getattr(core.ALLOCATOR, "fn", None) or _POOL.malloc


def set_allocator(allocator=None):
    import importlib
    core = importlib.import_module("gpuwm.metal.xp._core")
    core.ALLOCATOR.fn = None if allocator is _POOL.malloc else allocator


@contextlib.contextmanager
def using_allocator(allocator=None):
    """New arrays of this thread take their memory from ``allocator(nbytes)``
    (a MemoryPointer), as in CuPy; None restores the default pool."""
    import importlib
    core = importlib.import_module("gpuwm.metal.xp._core")   # the module (xp._core is CuPy's namespace alias)
    prev = getattr(core.ALLOCATOR, "fn", None)
    core.ALLOCATOR.fn = allocator
    try:
        yield
    finally:
        core.ALLOCATOR.fn = prev


def malloc_managed(size):
    """cudaMallocManaged: on Apple silicon all device memory is unified."""
    from ._core import MemoryPointer as _MPtr
    mem = _rt_module.DeviceMemory(max(int(size), 1))
    return _MPtr(mem, mem.gpu)


def _nvrtc_version():
    """No NVRTC runs on Metal.  Code that picks a CUDA-toolchain-measured table
    by (compute capability, NVRTC version) gets sm_120 / NVRTC 13.4: the
    translator reproduces the CUDA source semantics those rows were recorded
    under, so Metal is gated by CUDA's measured budgets (it does not reproduce
    ptxas' FMA contraction under --fmad=true, so hash pins of CUDA's output
    words can differ).  ARWEN_METAL_NVRTC_VERSION overrides it."""
    v = os.environ.get("ARWEN_METAL_NVRTC_VERSION", "13.4")
    return tuple(int(x) for x in v.split(".")[:2])


nvrtc = types.SimpleNamespace(getVersion=_nvrtc_version)
_FUNC_ATTRS = ("max_threads_per_block", "shared_size_bytes", "const_size_bytes", "local_size_bytes",
               "num_regs", "ptx_version", "binary_version", "cache_mode_ca", "max_dynamic_shared_size_bytes",
               "preferred_shared_memory_carveout")


def _func_get_attribute(attribute, ptr):
    """cuFuncGetAttribute over a Metal pipeline.  Metal reserves no per-thread
    local frame the way CUDA does, so LOCAL_SIZE_BYTES (like num_regs) reads 0:
    CUDA frame-size budgets are vacuous on this backend (README.md)."""
    from ._custom import Function, _BY_PTR
    k = _BY_PTR.get(int(ptr))
    if k is None:
        raise CUDARuntimeError(400, "funcGetAttribute: not a Metal kernel handle")
    return Function(k).attributes[_FUNC_ATTRS[int(attribute)]]


def _cu_launch_kernel_impl(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra):
    """cuLaunchKernel for native callers (a C function pointer): ``params`` is
    CUDA's void** kernelParams, each entry pointing at one argument's value; the
    values are read by the translated kernel's parameter layout and launched."""
    import ctypes
    import numpy as _np
    from ._custom import _BY_PTR
    k = _BY_PTR.get(int(f or 0))
    if k is None:
        return 400                       # CUDA_ERROR_INVALID_HANDLE
    args = []
    try:
        for i, p in enumerate(k._layout):
            at = params[i]
            if p["kind"] == "ptr":
                args.append(ctypes.c_uint64.from_address(at).value)
            elif p["kind"] == "scalar":
                code = p["code"]
                dt = _np.dtype(_rt_module._FMT[code])
                args.append(_np.frombuffer(ctypes.string_at(at, dt.itemsize), dt)[0])
            else:
                args.append(ctypes.string_at(at, p["size"]))
        k.launch((gx, gy, gz), (bx, by, bz), tuple(args), shmem)
    except Exception as e:  # noqa: BLE001 - a C caller gets a CUDA status, the reason goes to stderr
        import sys
        print(f"[arwen-metal] cuLaunchKernel({k.name}): {e}", file=sys.stderr)
        return 1                         # CUDA_ERROR_INVALID_VALUE
    return 0


def _cu_graph_launch_impl(graph_exec, stream):
    g = _GRAPHS.get(int(graph_exec or 0))
    if g is None:
        return 400
    try:
        g.launch()
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[arwen-metal] cuGraphLaunch: {e}", file=sys.stderr)
        return 1
    return 0


def _cu_event_record_with_flags(event, stream, flags=0):
    e = _EVENTS.get(int(event or 0))
    if e is None:
        return 400
    e.record()
    return 0


def _make_cu_graph_launch():
    import ctypes
    return ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(_cu_graph_launch_impl)


def _make_cu_launch_kernel():
    import ctypes
    proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
    return proto(_cu_launch_kernel_impl)


class CUDADriverError(RuntimeError):
    """cupy.cuda.driver.CUDADriverError (raised with a CUresult status)."""
    def __init__(self, status=1, msg=None):
        self.status = int(status)
        super().__init__(msg or f"CUDA driver error {self.status} (Metal backend)")


driver = types.SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda f, b, s: 1,
                               CUDADriverError=CUDADriverError,
                               ctxGetCurrent=lambda: 1,          # one context, always current
                               ctxGetDevice=lambda: 0,
                               cuLaunchKernel=_make_cu_launch_kernel(),
                               cuGraphLaunch=_make_cu_graph_launch(),
                               cuEventRecordWithFlags=_cu_event_record_with_flags,
                               funcGetAttribute=_func_get_attribute,
                               **{"CU_FUNC_ATTRIBUTE_" + n.upper(): i for i, n in enumerate(_FUNC_ATTRS)})


# ------------------------------------------------------------------ cuFFT
def _cufft_module():
    """cupy.cuda.cufft on Metal: Apple GPUs have no cuFFT; plans run numpy.fft on
    the CPU over unified memory (declared on first use), in queue order (and
    recorded, not run, while a CUDA graph is captured).  Work areas are empty."""
    import sys as _sys
    import numpy as _np
    m = types.ModuleType("cupy.cuda.cufft")
    for k, v in dict(CUFFT_R2C=0x2a, CUFFT_C2R=0x2c, CUFFT_C2C=0x29, CUFFT_D2Z=0x6a, CUFFT_Z2D=0x6c,
                     CUFFT_Z2Z=0x69, CUFFT_FORWARD=-1, CUFFT_INVERSE=1).items():
        setattr(m, k, v)
    told = []

    class _Plan:
        def __init__(self, shape, fft_type, batch=1, axes=None):
            self.shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
            self.fft_type, self.batch = int(fft_type), int(batch)
            self.nx = self.shape[-1]
            self.work_area = types.SimpleNamespace(mem=types.SimpleNamespace(size=0), ptr=0)

        def _run(self, a, out, direction):
            if not told:
                told.append(1)
                print("[arwen-metal] cuFFT plan: numpy.fft on the CPU over unified memory (Metal has no cuFFT)",
                      file=_sys.stderr)
            _rt_module.synchronize()
            n = len(self.shape)
            x = a._v.reshape((-1,) + (a._v.shape[-n:] if a._v.ndim >= n else a._v.shape))
            axes = tuple(range(-n, 0))
            t = self.fft_type
            if t in (m.CUFFT_R2C, m.CUFFT_D2Z):
                r = _np.fft.rfftn(x, s=self.shape, axes=axes)
            elif t in (m.CUFFT_C2R, m.CUFFT_Z2D):
                r = _np.fft.irfftn(x, s=self.shape, axes=axes) * float(_np.prod(self.shape))   # cuFFT: unnormalised
            elif direction == m.CUFFT_FORWARD:
                r = _np.fft.fftn(x, axes=axes)
            else:
                r = _np.fft.ifftn(x, axes=axes) * float(_np.prod(self.shape))
            out._v[...] = r.reshape(out._v.shape).astype(out._v.dtype, copy=False)

        def fft(self, a, out, direction):
            if _rt_module.CAPTURE is not None:
                _rt_module.CAPTURE.append(("host", lambda: self._run(a, out, direction)))
                return
            self._run(a, out, direction)

    class Plan1d(_Plan):
        def __init__(self, nx, fft_type, batch, devices=None, out=None):
            super().__init__((nx,), fft_type, batch)

    class PlanNd(_Plan):
        def __init__(self, shape, inembed=None, istride=1, idist=0, onembed=None, ostride=1, odist=0,
                     fft_type=None, batch=1, order="C", last_axis=-1, last_size=None):
            super().__init__(shape, fft_type, batch)

    m.Plan1d, m.PlanNd = Plan1d, PlanNd
    return m


cufft = _cufft_module()
