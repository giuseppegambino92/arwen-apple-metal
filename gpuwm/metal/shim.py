"""Serve ``import cupy`` / ``import cupyx`` from the Metal array module.

ArWen's ~600 CuPy-importing modules stay unchanged: when the metal backend is
selected (``--backend metal`` or GPUWM_BACKEND=metal), this installs
gpuwm.metal.xp as ``cupy`` before any of them is imported.  Prebuilt CUDA
kernel packs cannot run on Metal, so GPUWM_AOT is forced off.
"""
from __future__ import annotations

import os
import sys
import types


def requested(argv=None) -> bool:
    argv = sys.argv if argv is None else argv
    if os.environ.get("GPUWM_BACKEND", "").strip().lower() == "metal":
        return True
    for i, a in enumerate(argv):
        if a == "--backend" and i + 1 < len(argv) and argv[i + 1].strip().lower() == "metal":
            return True
        if a.strip().lower() == "--backend=metal":
            return True
    return False


def _stub(name):
    m = types.ModuleType(name)
    m.__path__ = []

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise NotImplementedError(f"{name}.{attr} is not available on the Metal backend")
    m.__getattr__ = __getattr__
    return m


def install():
    if getattr(sys.modules.get("cupy"), "_gpuwm_metal", False):
        return sys.modules["cupy"]
    if "cupy" in sys.modules:
        raise RuntimeError("the metal backend must be selected before cupy is imported")
    os.environ["GPUWM_BACKEND"] = "metal"
    os.environ["GPUWM_AOT"] = "off"
    from gpuwm.metal import xp
    # gpuwm recognises device arrays by their type's module, as CuPy's is "cupy"
    xp.ndarray.__module__ = "cupy"
    xp._gpuwm_metal = True
    xp.__path__ = []
    sys.modules["cupy"] = xp
    cuda = types.ModuleType("cupy.cuda")
    cuda.__dict__.update({k: v for k, v in vars(xp.cuda).items() if not k.startswith("__")})
    cuda.__path__ = []
    xp.cuda = cuda
    sys.modules["cupy.cuda"] = cuda
    for sub in ("runtime", "compiler", "function", "memory", "nvrtc", "driver"):
        mod = types.ModuleType(f"cupy.cuda.{sub}")
        mod.__dict__.update(vars(getattr(cuda, sub)) if not isinstance(getattr(cuda, sub), types.ModuleType)
                            else vars(getattr(cuda, sub)))
        setattr(cuda, sub, mod)
        sys.modules[f"cupy.cuda.{sub}"] = mod
    for name in ("cupy._core", "cupy._core.core", "cupy.fft", "cupy.linalg", "cupy.random", "cupy.testing",
                 "cupy_backends", "cupy_backends.cuda", "cupy_backends.cuda.api", "cupy_backends.cuda.api.runtime",
                 "cupyx", "cupyx.scipy", "cupyx.scipy.fft", "cupyx.scipy.ndimage", "cupyx.scipy.special",
                 "cupyx.jit", "cupyx.profiler"):
        if name == "cupy._core":
            m = types.ModuleType(name)
            m.core = types.SimpleNamespace(ndarray=xp.ndarray)
            m.ndarray = xp.ndarray
            m.__path__ = []
        elif name == "cupy._core.core":
            m = types.ModuleType(name)
            m.ndarray = xp.ndarray
        else:
            m = _stub(name)
        sys.modules[name] = m
    return xp


_HOST_CONVERSIONS = ("asarray", "array", "asanyarray", "ascontiguousarray", "asfortranarray", "frombuffer")


def _guard_numpy_host_conversions(xp):
    """NumPy hands out a view of an ndarray subclass without asking it, so
    ``np.asarray(device_array)`` would read GPU memory with work still queued
    (CuPy raises instead).  On unified memory the view is valid once the GPU
    is done: these entry points wait for it.  ARWEN_METAL_TRACE_HOST=1 prints
    where gpuwm converts device arrays to host arrays."""
    import numpy as np
    from gpuwm.metal import runtime
    trace = os.environ.get("ARWEN_METAL_TRACE_HOST") == "1"

    def wrap(name, fn):
        def guarded(obj, *args, **kwargs):
            mem = getattr(obj, "_gpuwm_mem", None)
            passthrough = (name in ("array", "asanyarray") and kwargs.get("subok", name == "asanyarray")
                           and kwargs.get("copy", True if name == "array" else None) in (False, None)
                           and not args and kwargs.get("dtype") in (None, getattr(obj, "dtype", None)))
            if mem is not None and not passthrough:
                if trace and runtime._pending:
                    import traceback
                    sys.stderr.write(f"[arwen-metal host read] np.{name} of a device array with GPU work queued\n"
                                     + "".join(traceback.format_stack(limit=6)[:-1]))
                runtime.host_sync()
            return fn(obj, *args, **kwargs)
        guarded.__wrapped__ = fn
        guarded.__name__ = name
        return guarded
    for name in _HOST_CONVERSIONS:
        fn = getattr(np, name)
        if not getattr(fn, "_gpuwm_metal_guard", False):
            g = wrap(name, fn)
            g._gpuwm_metal_guard = True
            setattr(np, name, g)


def backend():
    """The gpuwm.backend.Backend for Metal."""
    from gpuwm.backend import Backend
    xp = install()

    class MetalBackend(Backend):
        def compilation_target(self, device=None):
            return "metal"

        def current_device_id(self):
            return 0

        def memory_info(self):
            return xp.cuda.runtime.memGetInfo()

        def active_blocks_per_multiprocessor(self, function, block_size, dynamic_shared=0):
            return max(1, function.max_threads_per_block // max(block_size, 1))

        def synchronize(self):
            from gpuwm.metal import runtime
            runtime.synchronize()

    return MetalBackend("metal", xp, False)
