"""Device arrays for the Metal backend: numpy ndarrays living in GPU memory.

``ndarray`` is a numpy.ndarray subclass whose bytes are an MTLBuffer in shared
storage.  Shape/stride/view logic is numpy's own (views never copy); every
computation is a GPU kernel (ufuncs, reductions, the numpy functions in
``_functions``); every host access first waits for queued GPU work.  A numpy
operation with no GPU implementation raises NotImplementedError naming it --
there is no silent host fallback.
"""
from __future__ import annotations

import builtins
import numbers
import operator

import numpy as np

from .. import runtime
from . import _kernels as K

_np_ndarray = np.ndarray


class MemoryPointer:
    """``a.data`` of CuPy: ``.ptr`` is the GPU address of the first element."""
    __slots__ = ("mem", "ptr")

    def __init__(self, mem, ptr):
        self.mem, self.ptr = mem, ptr


class ndarray(np.ndarray):
    _gpuwm_backend_array = True
    __array_priority__ = 100.0

    def __new__(cls, shape, dtype=float, memptr=None, strides=None, order="C"):
        return _empty(shape, dtype, order=order)

    def __array_finalize__(self, obj):
        mem = getattr(obj, "_gpuwm_mem", None)
        if mem is not None:
            # only a view into the same MTLBuffer is a device array; numpy-made
            # copies of a device array live in host memory and are not
            p = self.__array_interface__["data"][0]
            if not (mem.cpu <= p < mem.cpu + max(mem.nbytes, 1)) and self.size:
                mem = None
        self._gpuwm_mem = mem

    # ----------------------------------------------------------- CuPy surface
    @property
    def data(self):
        return MemoryPointer(self._gpuwm_mem, runtime.device_address(self))

    @property
    def __cuda_array_interface__(self):
        """CuPy's device-array description; ``data`` is the GPU address (unified memory)."""
        if self._gpuwm_mem is None:
            raise AttributeError("__cuda_array_interface__")
        return {"shape": self.shape, "typestr": self.dtype.str, "descr": self.dtype.descr,
                "data": (runtime.device_address(self), False), "version": 3,
                "strides": None if self.flags.c_contiguous else self.strides}

    @property
    def device(self):
        from . import cuda
        return cuda.Device(0)

    def get(self, stream=None, order="C", out=None, blocking=True):
        runtime.host_sync()
        h = np.array(self.view(_np_ndarray), order=order, copy=True)
        if out is not None:
            out[...] = h
            return out
        return h

    def set(self, arr, stream=None):
        arr = np.asarray(arr)
        if arr.shape != self.shape:
            raise ValueError(f"shape mismatch in set: {arr.shape} vs {self.shape}")
        runtime.host_sync()
        self.view(_np_ndarray)[...] = arr

    def toDlpack(self):
        raise NotImplementedError("DLPack export is not available on the Metal backend")

    # ----------------------------------------------------------- host access
    def _host(self):
        runtime.host_sync()
        return self.view(_np_ndarray)

    def tobytes(self, order="C"):
        return self._host().tobytes(order)

    @property
    def ctypes(self):
        runtime.host_sync()
        return _np_ndarray.ctypes.__get__(self)

    def item(self, *args):
        return self._host().item(*args)

    def tolist(self):
        return self._host().tolist()

    def __float__(self):
        return float(self._host())

    def __int__(self):
        return int(self._host())

    def __index__(self):
        return operator.index(self._host())

    def __bool__(self):
        return bool(self._host())

    def __complex__(self):
        return complex(self._host())

    def __repr__(self):
        return "metal." + repr(self._host())

    def __str__(self):
        return str(self._host())

    def __array__(self, dtype=None, copy=None):
        # CuPy refuses implicit conversion to host arrays; so do we.
        raise TypeError("Implicit conversion to a NumPy array is not allowed on the Metal backend; "
                        "use .get() or asnumpy()")

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        for i in range(self.shape[0]):
            yield self[i]

    def __reduce__(self):
        return (asarray, (self.get(),))

    def __deepcopy__(self, memo):
        return self.copy()

    def __copy__(self):
        return self.copy()

    # ----------------------------------------------------------- numpy protocols
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from . import _ufuncs
        return _ufuncs.dispatch(ufunc, method, inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        from . import _functions
        return _functions.dispatch(func, args, kwargs)

    # ----------------------------------------------------------- indexing
    def __getitem__(self, key):
        if _is_basic(key):
            res = _np_ndarray.__getitem__(self, key)
            if isinstance(res, _np_ndarray):
                return res
            # a scalar read: the element must be up to date
            runtime.host_sync()
            return _np_ndarray.__getitem__(self, key)
        from . import _functions
        return _functions.advanced_getitem(self, key)

    def __setitem__(self, key, value):
        if _is_basic(key):
            dst = _np_ndarray.__getitem__(self, key)
            if not isinstance(dst, _np_ndarray):
                dst = _np_ndarray.__getitem__(self, (key,) if not isinstance(key, tuple) else key + (Ellipsis,))
            assign(dst, value)
            return
        from . import _functions
        _functions.advanced_setitem(self, key, value)

    # ----------------------------------------------------------- methods with GPU semantics
    def copy(self, order="C"):
        out = _empty(self.shape, self.dtype, order=order if order in "CF" else "C")
        assign(out, self)
        return out

    def astype(self, dtype, order="K", casting=None, subok=None, copy=True):
        dtype = np.dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        out = _empty(self.shape, dtype, order="F" if (order == "F" or (order in "AK" and self.flags.f_contiguous
                                                                       and not self.flags.c_contiguous)) else "C")
        assign(out, self)
        return out

    def fill(self, value):
        assign(self, value)

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if order == "C":
            v = _np_ndarray.view(self)
            try:
                v.shape = shape          # succeeds only when no copy is needed
                return v
            except AttributeError:
                pass
            return _np_ndarray.reshape(self.copy("C"), shape)
        src = self if self.flags.f_contiguous else self.copy("F")
        return _np_ndarray.reshape(src, shape, order="F")

    def ravel(self, order="C"):
        if (order == "C" and self.flags.c_contiguous) or (order == "F" and self.flags.f_contiguous):
            return _np_ndarray.reshape(self, -1, order=order)
        return _np_ndarray.reshape(self.copy(order if order in "CF" else "C"), -1, order=order if order in "CF" else "C")

    def flatten(self, order="C"):
        return self.copy(order if order in "CF" else "C").ravel(order)

    def view(self, dtype=None, type=None):
        if dtype is None and type is None:
            return _np_ndarray.view(self)
        if type is not None and type is not ndarray and dtype is None and not isinstance(dtype, np.dtype):
            raise TypeError("viewing a device array as another array type is not supported")
        return _np_ndarray.view(self, dtype) if dtype is not None else _np_ndarray.view(self)

    def __len__(self):
        return _np_ndarray.__len__(self)

    def __hash__(self):
        raise TypeError("unhashable type: 'ndarray'")

    def __dlpack__(self, *a, **k):
        raise NotImplementedError("DLPack is not available on the Metal backend")

    def __matmul__(self, other):
        raise NotImplementedError("matmul on the Metal backend")


def _is_basic(key) -> bool:
    if not isinstance(key, tuple):
        key = (key,)
    for k in key:
        if k is None or k is Ellipsis or isinstance(k, slice):
            continue
        if isinstance(k, (bool, np.bool_)):
            return False
        if isinstance(k, (numbers.Integral, np.integer)):
            continue
        return False
    return True


# ------------------------------------------------------------------ allocation
def _empty(shape, dtype=float, order="C"):
    if isinstance(shape, (numbers.Integral, np.integer)):
        shape = (int(shape),)
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    if dtype.kind not in "biuf":
        raise TypeError(f"dtype {dtype} is not supported on the Metal backend")
    n = 1
    for s in shape:
        if s < 0:
            raise ValueError("negative dimensions are not allowed")
        n *= s
    mem = runtime.DeviceMemory(max(n * dtype.itemsize, 1))
    a = _np_ndarray.__new__(ndarray, shape, dtype, buffer=mem, order="F" if order == "F" else "C")
    a._gpuwm_mem = mem
    return a


def is_device(a) -> bool:
    return isinstance(a, ndarray) and a._gpuwm_mem is not None


# ------------------------------------------------------------------ assignment (GPU copy / fill / cast)
def assign(dst, src):
    """dst[...] = src on the GPU (broadcasting, casting like numpy's same_kind/unsafe assignment)."""
    if not is_device(dst):
        raise TypeError("assignment target is not a device array")
    if isinstance(src, (numbers.Number, np.generic, bool)) or (isinstance(src, _np_ndarray) and not is_device(src) and src.ndim == 0):
        if dst.size == 0:
            return
        val = np.asarray(src).item() if not isinstance(src, (numbers.Number, bool)) else src
        from ._ufuncs import elementwise
        elementwise("o0 = s0;", [dst], [], [(np.dtype(dst.dtype), _cast_scalar(val, dst.dtype))])
        return
    if isinstance(src, _np_ndarray) and not is_device(src):
        # host -> device: unified memory, so a CPU write once the GPU is done with dst
        src = np.asarray(src)
        runtime.host_sync()
        dst.view(_np_ndarray)[...] = src
        return
    if not is_device(src):
        src = asarray(src)
    if dst.size == 0:
        return
    src_b = np.broadcast_to(src.view(_np_ndarray), dst.shape) if src.shape != dst.shape else src
    if src_b is not src:
        src_b = src_b.view(ndarray)
        src_b._gpuwm_mem = src._gpuwm_mem
    from ._ufuncs import elementwise
    elementwise("o0 = (%s)a0;" % K.ctype(dst.dtype), [dst], [src_b], [])


def _cast_scalar(v, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return float(v)
    if dtype.kind == "b":
        return bool(v)
    return int(v)


# ------------------------------------------------------------------ creation
def empty(shape, dtype=float, order="C"):
    return _empty(shape, dtype, order)


def zeros(shape, dtype=float, order="C"):
    a = _empty(shape, dtype, order)
    a.fill(0)
    return a


def ones(shape, dtype=float, order="C"):
    a = _empty(shape, dtype, order)
    a.fill(1)
    return a


def full(shape, fill_value, dtype=None, order="C"):
    if dtype is None:
        dtype = np.asarray(fill_value).dtype if not is_device(fill_value) else fill_value.dtype
    a = _empty(shape, dtype, order)
    assign(a, fill_value)
    return a


def _like_order(a, order):
    if order == "K" or order == "A":
        return "F" if a.flags.f_contiguous and not a.flags.c_contiguous else "C"
    return order


def empty_like(a, dtype=None, order="K", subok=None, shape=None):
    return _empty(a.shape if shape is None else shape, a.dtype if dtype is None else dtype, _like_order(a, order))


def zeros_like(a, dtype=None, order="K", subok=None, shape=None):
    r = empty_like(a, dtype, order, shape=shape)
    r.fill(0)
    return r


def ones_like(a, dtype=None, order="K", subok=None, shape=None):
    r = empty_like(a, dtype, order, shape=shape)
    r.fill(1)
    return r


def full_like(a, fill_value, dtype=None, order="K", subok=None, shape=None):
    r = empty_like(a, dtype, order, shape=shape)
    assign(r, fill_value)
    return r


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0, blocking=False):
    if is_device(obj):
        a = obj if dtype is None else obj.astype(dtype, copy=False)
        if copy and a is obj:
            a = obj.copy(order="F" if order == "F" else "C")
        while a.ndim < ndmin:
            a = a[None]
        return a
    if isinstance(obj, (list, tuple)) and any(is_device(x) for x in obj):
        from ._functions import stack
        return stack([array(x, dtype=dtype) for x in obj])
    h = np.array(obj, dtype=dtype, order=order if order in "CFAK" else "K", ndmin=ndmin)
    if h.dtype.kind not in "biuf":
        raise TypeError(f"cannot create a device array of dtype {h.dtype}")
    out = _empty(h.shape, h.dtype, "F" if (h.flags.f_contiguous and not h.flags.c_contiguous) else "C")
    if h.size:
        out.view(_np_ndarray)[...] = h       # fresh buffer: no GPU work can touch it yet
    return out


def asarray(obj, dtype=None, order=None, blocking=False):
    if is_device(obj):
        if dtype is not None and np.dtype(dtype) != obj.dtype:
            return obj.astype(dtype)
        if order == "C" and not obj.flags.c_contiguous:
            return obj.copy("C")
        if order == "F" and not obj.flags.f_contiguous:
            return obj.copy("F")
        return obj
    return array(obj, dtype=dtype, order=order or "K")


def ascontiguousarray(a, dtype=None):
    a = asarray(a, dtype=dtype)
    return a if a.flags.c_contiguous else a.copy("C")


def asfortranarray(a, dtype=None):
    a = asarray(a, dtype=dtype)
    return a if a.flags.f_contiguous else a.copy("F")


def asnumpy(a, stream=None, order="C", out=None, blocking=True):
    if is_device(a):
        return a.get(order=order, out=out)
    r = np.asarray(a, order=order)
    if out is not None:
        out[...] = r
        return out
    return r


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = 0, start
    dt = np.dtype(dtype) if dtype is not None else np.result_type(start, stop, step)
    n = int(np.ceil((stop - start) / step)) if step else 0
    out = _empty(max(n, 0), dt)
    if n > 0:
        from ._ufuncs import elementwise
        t = K.ctype(dt)
        elementwise(f"o0 = ({t})(s0 + (double)i * s1);" if dt.kind == "f" else f"o0 = ({t})(s0 + i * s1);",
                    [out], [], [(np.dtype(np.float64 if dt.kind == "f" else np.int64), start),
                                (np.dtype(np.float64 if dt.kind == "f" else np.int64), step)])
    return out
