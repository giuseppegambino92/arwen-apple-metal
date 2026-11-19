"""The Metal backend's array module: the CuPy API surface gpuwm uses.

Installed as ``cupy`` (see gpuwm.metal.shim) when ``--backend metal`` is
selected.  Arrays live in unified GPU memory; computation is GPU kernels; a
NumPy function without a Metal kernel runs NumPy on the CPU over the same
memory, declared on first use (_functions.host_call).
"""
from __future__ import annotations

import sys as _sys
import types as _types

import numpy as _np

from . import cuda  # noqa: F401
from ._core import (ndarray, MemoryPointer, empty, zeros, ones, full, empty_like, zeros_like, ones_like,  # noqa: F401
                    full_like, array, asarray, ascontiguousarray, asfortranarray, asnumpy, arange, linspace, is_device)
from ._custom import RawModule, RawKernel, ElementwiseKernel, ReductionKernel  # noqa: F401
from . import _functions as _F
from . import _ufuncs as _U

__version__ = "14.2.0+metal"
_gpuwm_host_backend = False

# dtypes and constants
for _n in ("bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64",
           "intp", "uintp", "dtype", "nan", "inf", "pi", "e", "newaxis", "finfo", "iinfo", "integer", "floating",
           "number", "signedinteger", "unsignedinteger", "generic", "float16", "complex64", "complex128",
           "can_cast", "promote_types", "result_type", "isscalar", "ndim", "shape", "broadcast_shapes"):
    globals()[_n] = getattr(_np, _n)
bool = _np.bool_  # noqa: A001
float_ = _np.float64
int_ = _np.int64

# ufuncs: numpy's own objects, dispatched to the GPU through __array_ufunc__
_UFUNCS = ("add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "power", "remainder", "mod", "fmod",
           "negative", "positive", "absolute", "abs", "fabs", "sign", "sqrt", "cbrt", "square", "reciprocal", "exp",
           "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
           "arctan2", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "hypot", "deg2rad", "rad2deg",
           "radians", "degrees", "floor", "ceil", "trunc", "rint", "maximum", "minimum", "fmax", "fmin", "isnan",
           "isinf", "isfinite", "signbit", "copysign", "logical_and", "logical_or", "logical_xor", "logical_not",
           "greater", "greater_equal", "less", "less_equal", "equal", "not_equal", "bitwise_and", "bitwise_or",
           "bitwise_xor", "invert", "bitwise_not", "left_shift", "right_shift", "divmod")


import builtins as _bi


def _ufunc_wrapper(uf):
    def f(*args, **kwargs):
        out = kwargs.get("out")
        outs = out if isinstance(out, tuple) else (out,)
        if not _bi.any(is_device(x) for x in args) and not _bi.any(is_device(o) for o in outs):
            # CuPy computes on the device even for scalar/list inputs
            args = tuple(asarray(x) if isinstance(x, (list, tuple, _np.ndarray)) else x for x in args)
            if not _bi.any(is_device(x) for x in args):
                args = (asarray(_np.asarray(args[0])),) + args[1:]
        return uf(*args, **kwargs)
    f.__name__ = uf.__name__
    f.reduce = uf.reduce
    f.accumulate = uf.accumulate
    f.reduceat = lambda *a, **k: _F.host_call(uf.reduceat, a, k)
    f.outer = lambda *a, **k: _F.host_call(uf.outer, a, k)
    f.at = lambda *a, **k: _F.host_call(uf.at, a, k)
    f.nin, f.nout, f.ufunc = uf.nin, uf.nout, uf
    return f


for _n in _UFUNCS:
    globals()[_n] = _ufunc_wrapper(getattr(_np, _n))


def _fn(npfunc):
    impl = _F._IMPL[npfunc]

    def f(*args, **kwargs):
        return impl(*args, **kwargs)
    f.__name__ = npfunc.__name__
    return f


for _func in list(_F._IMPL):
    if getattr(_func, "__name__", None) and _func.__name__ not in globals():
        globals()[_func.__name__] = _fn(_func)
where = _fn(_np.where)
clip = _fn(_np.clip)
amax = _fn(_np.amax)
amin = _fn(_np.amin)
sum = _fn(_np.sum)  # noqa: A001
max = _fn(_np.max)  # noqa: A001
min = _fn(_np.min)  # noqa: A001
any = _fn(_np.any)  # noqa: A001
all = _fn(_np.all)  # noqa: A001
round = None  # set below  # noqa: A001


def around(a, decimals=0, out=None):
    if decimals:
        raise NotImplementedError("round(decimals != 0) on the Metal backend")
    return rint(a, out=out)  # noqa: F821  (numpy rounds half to even)


round = around  # noqa: A001
round_ = around


def get_array_module(*args):
    for a in args:
        if is_device(a):
            return _sys.modules[__name__]
    return _np


def get_default_memory_pool():
    return cuda._POOL


def get_default_pinned_memory_pool():
    return cuda._POOL


def fuse(*a, **k):
    raise NotImplementedError("cupy.fuse on the Metal backend")


class _Unavailable(AttributeError, NotImplementedError):
    """A CuPy name this backend does not provide: an AttributeError, so hasattr()
    and getattr(module, name, default) behave as on a module without it."""


def _unimplemented_module(name):
    m = _types.ModuleType(name)

    def __getattr__(attr):
        raise _Unavailable(f"{name}.{attr} on the Metal backend")
    m.__getattr__ = __getattr__
    return m


def _fft_module():
    """cupy.fft on Metal: numpy.fft on the CPU over unified memory (declared on
    first use); Apple GPUs have no cuFFT and ArWen uses FFTs only to build
    random perturbation patterns."""
    m = _types.ModuleType("cupy.fft")

    def __getattr__(attr):
        fn = getattr(_np.fft, attr, None)
        if fn is None or attr.startswith("_"):
            raise _Unavailable(f"cupy.fft.{attr} on the Metal backend")
        if not callable(fn):
            return fn

        def f(*args, **kw):
            return _F.host_call(fn, args, kw)
        f.__name__ = attr
        setattr(m, attr, f)
        return f
    m.__getattr__ = __getattr__
    return m


fft = _fft_module()
fft.config = _types.SimpleNamespace(enable_nd_planning=False, get_plan_cache=lambda: _types.SimpleNamespace(
    set_size=lambda n: None, set_memsize=lambda n: None, clear=lambda: None, get_size=lambda: 0))
fft._fft = _types.SimpleNamespace(_get_cufft_plan_nd=lambda shape, fft_type, axes=None, order="C", out_size=None,
                                  to_cache=True: cuda.cufft.PlanNd(tuple(shape[a] for a in (axes or range(len(shape)))),
                                                                   fft_type=fft_type))
def _linalg_module():
    """cupy.linalg on Metal: Apple GPUs have no LAPACK (no cuSOLVER equivalent), so
    these run numpy.linalg (LAPACK on the CPU) on the same unified memory, after the
    GPU is done with the operands, and return device arrays.  Declared, not silent:
    the first call prints which routine ran on the CPU."""
    import sys as _sys
    import numpy as _np
    m = _types.ModuleType("cupy.linalg")
    told = set()

    def wrap(name):
        fn = getattr(_np.linalg, name)

        def f(*args, **kw):
            if name not in told:
                told.add(name)
                print(f"[arwen-metal] cupy.linalg.{name}: LAPACK on the CPU (unified memory; "
                      "Metal has no GPU LAPACK)", file=_sys.stderr)
            host = [asnumpy(a) if is_device(a) else a for a in args]
            out = fn(*host, **kw)
            if isinstance(out, tuple):
                return type(out)(*(asarray(x) if isinstance(x, _np.ndarray) else x for x in out)) \
                    if hasattr(out, "_fields") else tuple(asarray(x) if isinstance(x, _np.ndarray) else x for x in out)
            return asarray(out) if isinstance(out, _np.ndarray) else out
        f.__name__ = name
        return f
    for name in ("eigh", "eigvalsh", "solve", "inv", "norm", "det", "slogdet", "cholesky", "qr", "svd",
                 "lstsq", "pinv", "matrix_rank", "matrix_power"):
        setattr(m, name, wrap(name))
    m.LinAlgError = _np.linalg.LinAlgError

    def __getattr__(attr):
        raise _Unavailable(f"cupy.linalg.{attr} on the Metal backend")
    m.__getattr__ = __getattr__
    return m


linalg = _linalg_module()
def _random_module():
    """cupy.random on Metal: NumPy's generator on the CPU, results as device
    arrays (declared on first use).  CuPy's own streams differ from NumPy's in
    any case, so no caller can depend on CuPy's exact random words."""
    m = _types.ModuleType("cupy.random")

    def __getattr__(attr):
        fn = getattr(_np.random, attr, None)
        if fn is None or attr.startswith("_"):
            raise _Unavailable(f"cupy.random.{attr} on the Metal backend")
        if attr in ("RandomState", "Generator", "default_rng"):
            return _device_generator(fn)
        if isinstance(fn, type):           # other host classes
            return fn

        def f(*args, **kw):
            dtype = kw.pop("dtype", None)        # CuPy's generators take dtype=; NumPy's module functions do not
            r = _F.host_call(fn, args, kw)
            return r.astype(dtype) if dtype is not None and hasattr(r, "astype") else r
        f.__name__ = attr
        setattr(m, attr, f)
        return f
    m.__getattr__ = __getattr__
    return m


def _device_generator(factory):
    """cupy.random.RandomState / default_rng: NumPy's generator on the CPU whose
    draws come back as device arrays and accept CuPy's ``dtype=``."""
    class _Gen:
        def __init__(self, *a, **k):
            self._g = factory(*a, **k)

        def __getattr__(self, name):
            meth = getattr(self._g, name)
            if not callable(meth):
                return meth

            def f(*args, **kw):
                dtype = kw.pop("dtype", None)
                r = meth(*args, **kw)
                if isinstance(r, _np.ndarray) or isinstance(r, _np.generic):
                    r = asarray(_np.asarray(r) if dtype is None else _np.asarray(r).astype(dtype))
                return r
            return f
    _Gen.__name__ = getattr(factory, "__name__", "Generator")
    return _Gen


random = _random_module()
def _testing_module():
    """cupy.testing's array assertions: device operands are read back (after the GPU
    is done) and compared by numpy.testing, exactly as CuPy's own do."""
    import numpy.testing as npt
    m = _types.ModuleType("cupy.testing")

    def _h(x):
        return asnumpy(x) if is_device(x) else x

    def wrap(fn):
        def f(actual, desired, *a, **k):
            return fn(_h(actual), _h(desired), *a, **k)
        f.__name__ = fn.__name__
        return f
    for name in ("assert_array_equal", "assert_allclose", "assert_array_almost_equal",
                 "assert_array_less", "assert_array_max_ulp", "assert_array_almost_equal_nulp"):
        setattr(m, name, wrap(getattr(npt, name)))

    def __getattr__(attr):
        raise _Unavailable(f"cupy.testing.{attr} on the Metal backend")
    m.__getattr__ = __getattr__
    return m


testing = _testing_module()
_core = _types.SimpleNamespace(core=_types.SimpleNamespace(ndarray=ndarray), ndarray=ndarray)
_environment = _types.SimpleNamespace(get_cuda_path=lambda: None, get_nvcc_path=lambda: None)



def __getattr__(name):
    """``cupy.<numpy function>`` without a Metal kernel: NumPy's function, run on the
    CPU over unified memory by _functions.host_call (declared on first use)."""
    fn = getattr(_np, name, None)
    if callable(fn) and not isinstance(fn, type) and not name.startswith("_"):
        def f(*args, **kw):
            return _F.host_call(fn, args, kw)
        f.__name__ = name
        globals()[name] = f
        return f
    raise AttributeError(f"module 'cupy' (Metal backend) has no attribute {name!r}")
