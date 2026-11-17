"""The Metal backend's array module: the CuPy API surface gpuwm uses.

Installed as ``cupy`` (see gpuwm.metal.shim) when ``--backend metal`` is
selected.  Arrays live in unified GPU memory; computation is GPU kernels; a
function without a GPU implementation raises NotImplementedError naming it.
"""
from __future__ import annotations

import sys as _sys
import types as _types

import numpy as _np

from . import cuda  # noqa: F401
from ._core import (ndarray, MemoryPointer, empty, zeros, ones, full, empty_like, zeros_like, ones_like,  # noqa: F401
                    full_like, array, asarray, ascontiguousarray, asfortranarray, asnumpy, arange, is_device)
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


def _unimplemented_module(name):
    m = _types.ModuleType(name)

    def __getattr__(attr):
        raise NotImplementedError(f"{name}.{attr} on the Metal backend")
    m.__getattr__ = __getattr__
    return m


fft = _unimplemented_module("cupy.fft")
linalg = _unimplemented_module("cupy.linalg")
random = _unimplemented_module("cupy.random")
testing = _unimplemented_module("cupy.testing")
_core = _types.SimpleNamespace(core=_types.SimpleNamespace(ndarray=ndarray), ndarray=ndarray)
_environment = _types.SimpleNamespace(get_cuda_path=lambda: None, get_nvcc_path=lambda: None)
