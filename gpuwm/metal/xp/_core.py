"""Device arrays for the Metal backend.

``ndarray`` is NOT a numpy.ndarray, exactly as CuPy's is not: gpuwm (like any
CuPy code) tells host from device arrays with ``isinstance(x, np.ndarray)``,
and NumPy must refuse to read device memory implicitly (``np.asarray`` of a
device array raises TypeError, as in CuPy).

Each array holds ``_v``, a numpy view over its MTLBuffer (shared storage,
unified memory), used only for shape/stride/view bookkeeping -- slicing,
transposes and reshapes never copy.  Every computation is a GPU kernel
(ufuncs, reductions, the numpy functions in ``_functions``); host reads wait
for queued GPU work.  A NumPy operation without a GPU implementation raises
NotImplementedError naming it: there is no silent host fallback.
"""
from __future__ import annotations

import numbers
import operator

import numpy as np

from .. import runtime
from . import _kernels as K


class MemoryPointer:
    """``a.data`` of CuPy: ``.ptr`` is the GPU address of the first element."""
    __slots__ = ("mem", "ptr")

    def __init__(self, mem, ptr):
        self.mem, self.ptr = mem, ptr

    def __int__(self):
        return int(self.ptr)

    def __add__(self, offset):
        """CuPy's ``MemoryPointer + n``: the same allocation, n bytes further."""
        return MemoryPointer(self.mem, int(self.ptr) + int(offset))

    def __radd__(self, offset):
        return self.__add__(offset)

    def __sub__(self, offset):
        return MemoryPointer(self.mem, int(self.ptr) - int(offset))

    @property
    def device_id(self):
        return 0

    # byte-level operations on unified memory, ordered after the GPU's queued work
    # (recorded instead when a CUDA graph is being captured)
    def _bytes(self, size):
        off = int(self.ptr) - self.mem.gpu
        if off < 0 or off + int(size) > self.mem.nbytes:
            raise ValueError("MemoryPointer range outside its allocation")
        return np.frombuffer(self.mem, dtype=np.uint8)[off:off + int(size)]

    def _host_op(self, fn):
        if runtime.CAPTURE is not None:
            runtime.CAPTURE.append(("host", fn))
            return
        runtime.host_sync()
        fn()

    def memset(self, value, size):
        self._host_op(lambda: self._bytes(size).fill(int(value) & 0xFF))

    def memset_async(self, value, size, stream=None):
        self.memset(value, size)

    def copy_from_device(self, src, size):
        self._host_op(lambda: self._bytes(size).__setitem__(slice(None), src._bytes(size)))

    def copy_from_device_async(self, src, size, stream=None):
        self.copy_from_device(src, size)

    def copy_from_host(self, mem, size):
        import ctypes
        self._host_op(lambda: ctypes.memmove(self._bytes(size).ctypes.data, int(getattr(mem, "value", mem) or 0)
                                             if not hasattr(mem, "ptr") else int(mem.ptr), int(size)))

    def copy_from_host_async(self, mem, size, stream=None):
        self.copy_from_host(mem, size)

    def copy_to_host(self, mem, size):
        import ctypes
        self._host_op(lambda: ctypes.memmove(int(getattr(mem, "value", mem) or 0) if not hasattr(mem, "ptr")
                                             else int(mem.ptr), self._bytes(size).ctypes.data, int(size)))

    def copy_to_host_async(self, mem, size, stream=None):
        self.copy_to_host(mem, size)

    def copy_from(self, mem, size):
        if isinstance(mem, MemoryPointer):
            self.copy_from_device(mem, size)
        else:
            self.copy_from_host(mem, size)

    def copy_from_async(self, mem, size, stream=None):
        self.copy_from(mem, size)


def _wrap(view: np.ndarray, mem) -> "ndarray":
    a = object.__new__(ndarray)
    a._v = view
    a._gpuwm_mem = mem
    return a


class ndarray:
    __slots__ = ("_v", "_gpuwm_mem", "_ga", "__weakref__")   # _ga: GPU address of element 0, cached
    _gpuwm_backend_array = True
    __array_priority__ = 100.0

    def __init__(self, shape, dtype=float, memptr=None, strides=None, order="C"):
        if memptr is None:
            src = _empty(shape, dtype, order=order)
            self._v, self._gpuwm_mem = src._v, src._gpuwm_mem
            return
        mem = memptr.mem
        dtype = np.dtype(dtype)
        if isinstance(shape, (numbers.Integral, np.integer)):
            shape = (int(shape),)
        offset = int(memptr.ptr) - mem.gpu
        if not 0 <= offset <= mem.nbytes:
            raise ValueError("memptr does not point into its memory")
        base = np.frombuffer(mem, dtype=np.uint8)
        self._v = np.ndarray(tuple(shape), dtype, buffer=base, offset=offset, strides=strides,
                             order="F" if order == "F" else "C")
        self._gpuwm_mem = mem

    # ----------------------------------------------------------- structure (numpy's own bookkeeping)
    shape = property(lambda s: s._v.shape)
    dtype = property(lambda s: s._v.dtype)
    strides = property(lambda s: s._v.strides)
    ndim = property(lambda s: s._v.ndim)
    size = property(lambda s: s._v.size)
    itemsize = property(lambda s: s._v.itemsize)
    nbytes = property(lambda s: s._v.nbytes)
    flags = property(lambda s: s._v.flags)
    base = property(lambda s: None)

    @shape.setter
    def shape(self, value):
        self._v.shape = value          # numpy raises when it would need a copy

    @property
    def T(self):
        return _wrap(self._v.T, self._gpuwm_mem)

    @property
    def real(self):
        # a view, as NumPy's / CuPy's (complex: the real parts in place)
        return _wrap(self._v.real, self._gpuwm_mem) if self.dtype.kind == "c" else self

    @real.setter
    def real(self, value):
        assign(self.real, value)

    @property
    def imag(self):
        if self.dtype.kind == "c":
            return _wrap(self._v.imag, self._gpuwm_mem)
        z = _empty(self.shape, self.dtype)
        z.fill(0)
        return z

    @imag.setter
    def imag(self, value):
        if self.dtype.kind != "c":
            raise TypeError("array does not have imaginary part to set")
        assign(self.imag, value)

    @property
    def flat(self):
        """A flat view (contiguous arrays) or NumPy's flat iterator over the same
        unified memory, after the GPU is done with the array."""
        if self.flags.c_contiguous:
            return self.ravel()
        runtime.host_sync()
        return self._v.flat

    @property
    def data(self):
        return MemoryPointer(self._gpuwm_mem, runtime.device_address(self))

    @property
    def device(self):
        from . import cuda
        return cuda.Device(0)

    @property
    def __cuda_array_interface__(self):
        """CuPy's device-array description; ``data`` is the GPU address."""
        return {"shape": self.shape, "typestr": self.dtype.str, "descr": self.dtype.descr,
                "data": (runtime.device_address(self), False), "version": 3,
                "strides": None if self.flags.c_contiguous else self.strides}

    def __len__(self):
        return len(self._v)

    def __hash__(self):
        raise TypeError("unhashable type: 'cupy.ndarray'")

    # ----------------------------------------------------------- host access (waits for the GPU)
    def _host(self):
        runtime.host_sync()
        return self._v

    def get(self, stream=None, order="C", out=None, blocking=True):
        h = np.array(self._host(), order=order, copy=True)
        if out is not None:
            out[...] = h
            return out
        return h

    def set(self, arr, stream=None):
        arr = np.asarray(arr)
        if arr.shape != self.shape:
            raise ValueError(f"shape mismatch in set: {arr.shape} vs {self.shape}")
        runtime.host_sync()
        self._v[...] = arr

    def item(self, *args):
        return self._host().item(*args)

    def tolist(self):
        return self._host().tolist()

    def tobytes(self, order="C"):
        return self._host().tobytes(order)

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
        return "array(" + np.array2string(self._host(), separator=", ") + f", dtype={self.dtype})"

    def __str__(self):
        return str(self._host())

    def __format__(self, spec):
        return format(self._host() if self.ndim else self._host().item(), spec)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Please use `.get()` to construct a NumPy array explicitly.")

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

    def __dlpack__(self, *a, **k):
        raise NotImplementedError("DLPack is not available on the Metal backend")

    def toDlpack(self):
        raise NotImplementedError("DLPack is not available on the Metal backend")

    # ----------------------------------------------------------- numpy protocols
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from . import _ufuncs
        return _ufuncs.dispatch(ufunc, method, inputs, kwargs)

    def __getattr__(self, name):
        """A NumPy ndarray method with no Metal implementation: run on the CPU over the
        same unified memory (declared on first use; see _functions.host_call)."""
        meth = getattr(np.ndarray, name, None)
        if name.startswith("_") or not callable(meth):
            raise AttributeError(f"'cupy.ndarray' (Metal backend) has no attribute {name!r}")
        from . import _functions

        def call(*args, **kw):
            return _functions.host_call(meth, (self,) + args, kw)
        return call

    def __array_function__(self, func, types, args, kwargs):
        from . import _functions
        return _functions.dispatch(func, args, kwargs)

    # ----------------------------------------------------------- indexing
    def __getitem__(self, key):
        if _is_basic(key):
            res = self._v[_keyv(key)]
            if isinstance(res, np.ndarray):
                return _wrap(res, self._gpuwm_mem)
            # an element, as CuPy: a 0-d device array (a view; float()/int()/bool()/.get()
            # wait for the GPU)
            k = _keyv(key)
            k = k if isinstance(k, tuple) else (k,)
            return _wrap(self._v[k + (Ellipsis,)], self._gpuwm_mem)
        from . import _functions
        if self.dtype.kind == "c":
            return _functions.host_getitem(self, key)
        try:
            return _functions.advanced_getitem(self, key)
        except NotImplementedError:
            return _functions.host_getitem(self, key)

    def __setitem__(self, key, value):
        if _is_basic(key):
            k = _keyv(key)
            k = k if isinstance(k, tuple) else (k,)
            if not any(x is Ellipsis for x in k):
                k = k + (Ellipsis,)                # an Ellipsis makes numpy return a (0-d) view
            assign(_wrap(self._v[k], self._gpuwm_mem), value)
            return
        from . import _functions
        if self.dtype.kind == "c" or getattr(getattr(value, "dtype", None), "kind", None) == "c":
            _functions.host_setitem(self, key, value)
            return
        try:
            _functions.advanced_setitem(self, key, value)
        except NotImplementedError:
            _functions.host_setitem(self, key, value)

    # ----------------------------------------------------------- views
    def view(self, dtype=None, type=None):
        if type is not None and type is not ndarray:
            raise TypeError("a device array cannot be viewed as another array type")
        return _wrap(self._v.view() if dtype is None else self._v.view(dtype), self._gpuwm_mem)

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if order == "C":
            v = self._v.view()
            try:
                v.shape = shape                   # succeeds only when no copy is needed
                return _wrap(v, self._gpuwm_mem)
            except AttributeError:
                c = self.copy("C")
                return _wrap(c._v.reshape(shape), c._gpuwm_mem)
        src = self if self.flags.f_contiguous else self.copy("F")
        return _wrap(src._v.reshape(shape, order="F"), src._gpuwm_mem)

    def ravel(self, order="C"):
        if (order == "C" and self.flags.c_contiguous) or (order == "F" and self.flags.f_contiguous):
            return _wrap(self._v.reshape(-1, order=order), self._gpuwm_mem)
        o = order if order in "CF" else "C"
        c = self.copy(o)
        return _wrap(c._v.reshape(-1, order=o), c._gpuwm_mem)

    def flatten(self, order="C"):
        o = order if order in "CF" else "C"
        c = self.copy(o)
        return _wrap(c._v.reshape(-1, order=o), c._gpuwm_mem)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return _wrap(self._v.transpose(*axes) if axes else self._v.transpose(), self._gpuwm_mem)

    def swapaxes(self, a, b):
        return _wrap(self._v.swapaxes(a, b), self._gpuwm_mem)

    def squeeze(self, axis=None):
        return _wrap(self._v.squeeze(axis), self._gpuwm_mem)

    # ----------------------------------------------------------- data-moving methods (GPU)
    def copy(self, order="C"):
        out = _empty(self.shape, self.dtype, order="F" if order == "F" else "C")
        assign(out, self)
        return out

    def astype(self, dtype, order="K", casting=None, subok=None, copy=True):
        dtype = np.dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        f = order == "F" or (order in "AK" and self.flags.f_contiguous and not self.flags.c_contiguous)
        out = _empty(self.shape, dtype, "F" if f else "C")
        assign(out, self)
        return out

    def fill(self, value):
        assign(self, value)

    # reductions and friends: NumPy's functions dispatch back to the GPU implementations
    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        return np.max(self, axis=axis, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return np.min(self, axis=axis, out=out, keepdims=keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return np.any(self, axis=axis, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        return np.all(self, axis=axis, out=out, keepdims=keepdims)

    def argmax(self, axis=None, out=None, keepdims=False):
        return np.argmax(self, axis=axis, out=out, keepdims=keepdims)

    def argmin(self, axis=None, out=None, keepdims=False):
        return np.argmin(self, axis=axis, out=out, keepdims=keepdims)

    def cumsum(self, axis=None, dtype=None, out=None):
        return np.cumsum(self, axis=axis, dtype=dtype, out=out)

    def clip(self, min=None, max=None, out=None):
        return np.clip(self, min, max, out=out)

    def nonzero(self):
        return np.nonzero(self)

    def round(self, decimals=0, out=None):
        if decimals:
            raise NotImplementedError("round(decimals != 0) on the Metal backend")
        return np.rint(self, out=out)

    def conj(self):
        return np.conjugate(self) if self.dtype.kind == "c" else self

    conjugate = conj

    def __matmul__(self, other):
        raise NotImplementedError("matmul on the Metal backend")


# operators: NumPy's ufuncs, which dispatch to the GPU through __array_ufunc__
def _binop(uf):
    def f(self, other):
        return uf(self, other)

    def r(self, other):
        return uf(other, self)

    def i(self, other):
        return uf(self, other, out=(self,))
    return f, r, i


for _name, _uf in (("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("truediv", np.true_divide),
                   ("floordiv", np.floor_divide), ("mod", np.remainder), ("pow", np.power), ("and", np.bitwise_and),
                   ("or", np.bitwise_or), ("xor", np.bitwise_xor), ("lshift", np.left_shift),
                   ("rshift", np.right_shift)):
    _f, _r, _i = _binop(_uf)
    setattr(ndarray, f"__{_name}__", _f)
    setattr(ndarray, f"__r{_name}__", _r)
    setattr(ndarray, f"__i{_name}__", _i)
for _name, _uf in (("lt", np.less), ("le", np.less_equal), ("gt", np.greater), ("ge", np.greater_equal),
                   ("eq", np.equal), ("ne", np.not_equal)):
    setattr(ndarray, f"__{_name}__", (lambda uf: lambda self, other: uf(self, other))(_uf))
ndarray.__neg__ = lambda self: np.negative(self)
ndarray.__pos__ = lambda self: np.positive(self)
ndarray.__abs__ = lambda self: np.absolute(self)
ndarray.__invert__ = lambda self: np.invert(self)
ndarray.__divmod__ = lambda self, other: np.divmod(self, other)


def _keyv(key):
    """Index for the internal numpy view: device-array scalars become ints."""
    if isinstance(key, tuple):
        return tuple(_keyv(k) for k in key)
    if isinstance(key, ndarray) and key.ndim == 0:
        return key.item()
    return key


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
        if isinstance(k, ndarray) and k.ndim == 0 and k.dtype.kind in "iu":
            continue
        return False
    return True


# ------------------------------------------------------------------ allocation
def _empty(shape, dtype=float, order="C"):
    if isinstance(shape, (numbers.Integral, np.integer)):
        shape = (int(shape),)
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    if dtype.kind not in "biufc":
        raise TypeError(f"dtype {dtype} is not supported on the Metal backend")
    n = 1
    for s in shape:
        if s < 0:
            raise ValueError("negative dimensions are not allowed")
        n *= s
    alloc = getattr(ALLOCATOR, "fn", None)
    if alloc is not None and n * dtype.itemsize:
        # cupy.cuda.using_allocator: the array's memory comes from that allocator
        mp = alloc(n * dtype.itemsize)
        mem = mp.mem
        v = np.ndarray(shape, dtype, buffer=mem, offset=int(mp.ptr) - mem.gpu, order="F" if order == "F" else "C")
        a = _wrap(v, mem)
        _ALLOCATED_BY[id(a)] = mp      # keeps the allocator's owner object alive with the array
        _weakref.finalize(a, _ALLOCATED_BY.pop, id(a), None)
        return a
    mem = runtime.DeviceMemory(max(n * dtype.itemsize, 1))
    v = np.ndarray(shape, dtype, buffer=mem, order="F" if order == "F" else "C")
    return _wrap(v, mem)


import threading as _threading
import weakref as _weakref

#: the allocator cupy.cuda.using_allocator installs for this thread (None: the default pool)
ALLOCATOR = _threading.local()
_ALLOCATED_BY = {}


def is_device(a) -> bool:
    return isinstance(a, ndarray)


def broadcast(a, shape):
    """A zero-copy broadcast view of a device array."""
    return _wrap(np.broadcast_to(a._v, shape), a._gpuwm_mem)


# ------------------------------------------------------------------ assignment (GPU copy / fill / cast)
def assign(dst, src):
    """dst[...] = src on the GPU (broadcasting, casting like numpy's unsafe assignment)."""
    if not is_device(dst):
        raise TypeError("assignment target is not a device array")
    if dst.dtype.kind == "c" or getattr(getattr(src, "dtype", None), "kind", None) == "c" or isinstance(src, complex):
        # complex numbers: stored in unified memory, moved by NumPy on the CPU (no Metal kernel)
        runtime.host_sync()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            np.copyto(dst._v, src._v if is_device(src) else np.asarray(src), casting="unsafe")
        return
    if isinstance(src, (numbers.Number, np.generic, bool)) or (isinstance(src, np.ndarray) and src.ndim == 0):
        if dst.size == 0:
            return
        val = np.asarray(src).item() if not isinstance(src, (numbers.Number, bool)) else src
        from ._ufuncs import elementwise
        elementwise("o0 = s0;", [dst], [], [(np.dtype(dst.dtype), _cast_scalar(val, dst.dtype))])
        return
    if isinstance(src, np.ndarray):
        # host -> device: unified memory, so a CPU write once the GPU is done with dst
        runtime.host_sync()
        dst._v[...] = src
        return
    if not is_device(src):
        src = asarray(src)
    if dst.size == 0:
        return
    src_b = broadcast(src, dst.shape) if src.shape != dst.shape else src
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
        dtype = fill_value.dtype if is_device(fill_value) else np.asarray(fill_value).dtype
    a = _empty(shape, dtype, order)
    assign(a, fill_value)
    return a


def _like_order(a, order):
    if order in ("K", "A"):
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
    if h.dtype.kind not in "biufc":
        raise TypeError(f"cannot create a device array of dtype {h.dtype}")
    out = _empty(h.shape, h.dtype, "F" if (h.flags.f_contiguous and not h.flags.c_contiguous) else "C")
    if h.size:
        out._v[...] = h              # fresh buffer: no GPU work can touch it yet
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


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """numpy.linspace's values (computed on the host in binary64, as NumPy), on the device."""
    vals = np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
    if retstep:
        return asarray(vals[0]), vals[1]
    return asarray(vals)


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
