"""NumPy functions on device arrays (``__array_function__``).

View-only functions use numpy's own implementation (no data movement).
Everything that reads or writes elements is a GPU kernel.  Anything not listed
here runs in NumPy on the CPU over the same unified memory, declared on first
use (host_call): never silently.
"""
from __future__ import annotations

import numbers

import numpy as np

from .. import runtime
from . import _kernels as K
from ._core import _empty, _wrap, asarray, assign, broadcast, is_device, ndarray
from . import _ufuncs as U

_IMPL = {}


def implements(*funcs):
    def deco(f):
        for fn in funcs:
            _IMPL[fn] = f
        return f
    return deco


def dispatch(func, args, kwargs):
    impl = _IMPL.get(func)
    if impl is None:
        return host_call(func, args, kwargs)
    try:
        return impl(*args, **kwargs)
    except NotImplementedError:
        # a case the Metal kernel does not cover (e.g. pad mode, diff prepend)
        return host_call(func, args, kwargs)


_TOLD = set()


def host_call(func, args, kwargs):
    """A NumPy function with no Metal kernel: NumPy runs it on the CPU over the
    SAME unified memory (device arrays are passed as views, no copy in; an
    ``out=`` device array is written in place), after the GPU has finished with
    the operands.  Results that are views of a device array stay views; others
    become device arrays (scalars as 0-d arrays, as CuPy returns).  Declared:
    the first use of each function in a process is printed."""
    import sys
    from .. import runtime
    name = getattr(func, "__name__", str(func))
    if name not in _TOLD:
        _TOLD.add(name)
        print(f"[arwen-metal] numpy.{name}: computed on the CPU over unified memory (no Metal kernel)",
              file=sys.stderr)
    runtime.host_sync()
    mems = []

    def down(x):
        if isinstance(x, ndarray):
            mems.append((x._v, x._gpuwm_mem))
            return x._v
        if isinstance(x, (list, tuple)) and any(isinstance(y, ndarray) for y in x):
            return type(x)(down(y) for y in x)
        return x
    hargs = [down(x) for x in args]
    hkw = {k: down(v) for k, v in kwargs.items()}
    res = func(*hargs, **hkw)
    out = kwargs.get("out")
    if runtime.CAPTURE is not None:
        return _captured_host_call(func, hargs, hkw, res, out, mems)
    if isinstance(out, ndarray):
        return out

    def up(r):
        if isinstance(r, np.ndarray):
            for v, mem in mems:
                if r.size and np.shares_memory(r, v):
                    return _wrap(r, mem)          # a view of device memory stays one
            return asarray(r)
        if isinstance(r, np.generic):
            return asarray(r)
        if isinstance(r, tuple):
            return type(r)(*(up(y) for y in r)) if hasattr(r, "_fields") else tuple(up(y) for y in r)
        if isinstance(r, list):
            return [up(y) for y in r]
        return r
    return up(res)


def _captured_host_call(func, hargs, hkw, res, out, mems):
    """host_call while a CUDA graph is being captured: the operands are not
    computed yet (the captured kernels run at Graph.launch), so the value just
    computed only gives the result's shape and dtype.  The call is recorded and
    runs again, in capture order, at every launch, writing into the same
    device arrays it returns now."""
    from .. import runtime
    if isinstance(out, ndarray):
        runtime.CAPTURE.append(("host", lambda: func(*hargs, **hkw)))
        return out
    leaves = []

    def alloc(r):
        if isinstance(r, np.ndarray):
            if any(r.size and np.shares_memory(r, v) for v, _ in mems):
                raise NotImplementedError(f"numpy.{getattr(func, '__name__', func)} returning a view "
                                          "while a CUDA graph is captured, on the Metal backend")
            d = _empty(r.shape, r.dtype)
            leaves.append(d)
            return d
        if isinstance(r, np.generic):
            d = _empty((), r.dtype)
            leaves.append(d)
            return d
        if isinstance(r, tuple):
            return type(r)(*(alloc(y) for y in r)) if hasattr(r, "_fields") else tuple(alloc(y) for y in r)
        if isinstance(r, list):
            return [alloc(y) for y in r]
        raise NotImplementedError(f"numpy.{getattr(func, '__name__', func)} returning "
                                  f"{type(r).__name__} while a CUDA graph is captured, on the Metal backend")
    result = alloc(res)

    def replay():
        r = func(*hargs, **hkw)
        flat = []

        def walk(x):
            if isinstance(x, (tuple, list)):
                for y in x:
                    walk(y)
            else:
                flat.append(np.asarray(x))
        walk(r)
        for d, v in zip(leaves, flat):
            d._v[...] = v
    runtime.CAPTURE.append(("host", replay))
    return result


def _dev(a):
    return a if is_device(a) else asarray(a)


# ------------------------------------------------------------------ views (numpy metadata, no copy)
def _item(a, key):
    return a[key]


def _views(fn):
    """A numpy view function applied to the device arrays' internal views."""
    def f(*args, **kwargs):
        mem = next(x._gpuwm_mem for x in args if isinstance(x, ndarray))
        res = fn(*[x._v if isinstance(x, ndarray) else x for x in args], **kwargs)
        if isinstance(res, (list, tuple)):
            return type(res)(_wrap(r, mem) if isinstance(r, np.ndarray) else r for r in res)
        return _wrap(res, mem) if isinstance(res, np.ndarray) else res
    return f


for _fn in (np.expand_dims, np.squeeze, np.transpose, np.moveaxis, np.swapaxes, np.atleast_1d, np.atleast_2d,
            np.atleast_3d, np.flip):
    implements(_fn)(_views(_fn))
implements(np.shape)(lambda a: a.shape)
implements(np.ndim)(lambda a: a.ndim)
implements(np.size)(lambda a, axis=None: a.size if axis is None else a.shape[axis])
implements(np.result_type)(lambda *args: np.result_type(*[x.dtype if isinstance(x, ndarray) else x for x in args]))
implements(np.may_share_memory, np.shares_memory)(
    lambda a, b, max_work=None: isinstance(a, ndarray) and isinstance(b, ndarray) and a._gpuwm_mem is b._gpuwm_mem
    and np.may_share_memory(a._v, b._v))


@implements(np.broadcast_to)
def broadcast_to(a, shape, subok=True):
    return broadcast(_dev(a), shape)


@implements(np.broadcast_arrays)
def broadcast_arrays(*args, subok=True):
    arrs = [_dev(a) for a in args]
    shape = np.broadcast_shapes(*[a.shape for a in arrs])
    return [broadcast(a, shape) for a in arrs]


@implements(np.reshape)
def reshape(a, shape=None, order="C", *, newshape=None, copy=None):
    return _dev(a).reshape(shape if shape is not None else newshape, order=order)


@implements(np.ravel)
def ravel(a, order="C"):
    return _dev(a).ravel(order)


# ------------------------------------------------------------------ creation-like
from . import _core as C  # noqa: E402

implements(np.empty_like)(C.empty_like)
implements(np.zeros_like)(C.zeros_like)
implements(np.ones_like)(C.ones_like)
implements(np.full_like)(C.full_like)


@implements(np.copy)
def copy(a, order="K", subok=False):
    return _dev(a).copy("F" if order == "F" else "C")


@implements(np.copyto)
def copyto(dst, src, casting="same_kind", where=True):
    if where is True:
        assign(dst, src)
        return
    w = _dev(where)
    s = _dev(src) if not isinstance(src, (numbers.Number, np.generic)) else src
    where_(w, s, dst, out=dst)


@implements(np.ascontiguousarray)
def _asc(a, dtype=None, *, like=None):
    return C.ascontiguousarray(a, dtype)


# ------------------------------------------------------------------ elementwise-shaped functions
@implements(np.where)
def where_(cond, x=None, y=None, out=None):
    if x is None and y is None:
        return nonzero(cond)
    cond = _dev(cond)
    xs = [x, y]
    dts = []
    for v in xs:
        dts.append(v.dtype if is_device(v) else (v.dtype if isinstance(v, np.generic) else type(v)))
    rt = np.result_type(*[np.empty(0, d) if isinstance(d, np.dtype) else d(0) for d in dts]) \
        if all(isinstance(d, np.dtype) for d in dts) else np.result_type(*[d if isinstance(d, np.dtype) else d(0) for d in dts])
    shapes = [cond.shape] + [v.shape for v in xs if is_device(v)]
    shape = np.broadcast_shapes(*shapes)
    res = out if out is not None else _empty(shape, rt)
    T = K.ctype(res.dtype)
    ins, scal, ex = [broadcast(cond, shape)], [], []
    for j, v in enumerate(xs):
        if is_device(v):
            ex.append(f"({T})a{len(ins)}")
            ins.append(broadcast(v, shape))
        else:
            ex.append(f"s{len(scal)}")
            scal.append((np.dtype(res.dtype), U._cast(np.asarray(v).item(), res.dtype)))
    U.elementwise(f"o0 = a0 ? {ex[0]} : {ex[1]};", [res], ins, scal)
    return res


@implements(np.clip)
def clip(a, a_min=None, a_max=None, out=None, **kw):
    a = _dev(a)
    lo = -np.inf if a_min is None else a_min
    hi = np.inf if a_max is None else a_max
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        lo = info.min if a_min is None else a_min
        hi = info.max if a_max is None else a_max
    return U.call(np._core.umath.clip, (a, lo, hi), out=out)


@implements(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = _dev(a), (_dev(b) if not isinstance(b, (numbers.Number, np.generic)) else b)
    shape = np.broadcast_shapes(a.shape, b.shape if is_device(b) else ())
    res = _empty(shape, np.bool_)
    dt = np.result_type(a.dtype, b.dtype if is_device(b) else np.asarray(b).dtype, np.float64 if a.dtype.kind != "f" else a.dtype)
    T = K.ctype(dt)
    ins = [broadcast(a, shape)]
    scal = [(np.dtype(dt), float(rtol)), (np.dtype(dt), float(atol))]
    if is_device(b):
        ins.append(broadcast(b, shape))
        bx = f"({T})a1"
    else:
        scal.append((np.dtype(dt), float(b)))
        bx = "s2"
    en = "true" if equal_nan else "false"
    U.elementwise(f"{T} x = ({T})a0, y = {bx}; "
                  f"o0 = (x == y) || (isfinite(x) && isfinite(y) && fabs(x - y) <= s1 + s0 * fabs(y)) || ({en} && x != x && y != y);",
                  [res], ins, scal)
    return res


@implements(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return bool(U.reduce("logical_and", isclose(a, b, rtol, atol, equal_nan)))


@implements(np.array_equal)
def array_equal(a1, a2, equal_nan=False):
    a1, a2 = _dev(a1), _dev(a2)
    if a1.shape != a2.shape:
        return False
    return bool(U.reduce("logical_and", U.call(np.equal, (a1, a2))))


@implements(np.diff)
def diff(a, n=1, axis=-1, prepend=None, append=None):
    if prepend is not None or append is not None:
        raise NotImplementedError("numpy.diff(prepend/append) on the Metal backend")
    a = _dev(a)
    axis = axis % a.ndim
    for _ in range(n):
        s1 = [slice(None)] * a.ndim
        s0 = [slice(None)] * a.ndim
        s1[axis] = slice(1, None)
        s0[axis] = slice(None, -1)
        a = U.call(np.subtract, (a[tuple(s1)], a[tuple(s0)]))
    return a


# ------------------------------------------------------------------ reductions
def _red(name):
    def f(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, **kw):
        return U.reduce(name, _dev(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
    return f


implements(np.sum)(_red("add"))
implements(np.prod)(_red("multiply"))
implements(np.max, np.amax)(_red("maximum"))
implements(np.min, np.amin)(_red("minimum"))
implements(np.nanmax)(_red("fmax"))
implements(np.nanmin)(_red("fmin"))


@implements(np.any)
def any_(a, axis=None, out=None, keepdims=False, where=True):
    return U.reduce("logical_or", _dev(a), axis=axis, out=out, keepdims=keepdims, where=where)


@implements(np.all)
def all_(a, axis=None, out=None, keepdims=False, where=True):
    return U.reduce("logical_and", _dev(a), axis=axis, out=out, keepdims=keepdims, where=where)


@implements(np.mean)
def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    a = _dev(a)
    rdt = np.dtype(dtype) if dtype is not None else (a.dtype if a.dtype.kind == "f" else np.dtype(np.float64))
    s = U.reduce("add", a, axis=axis, dtype=rdt, keepdims=keepdims, where=where)
    axes = U._norm_axes(axis, a.ndim)
    n = 1
    for k in axes:
        n *= a.shape[k]
    return U.call(np.true_divide, (s, rdt.type(n)), out=out)


@implements(np.count_nonzero)
def count_nonzero(a, axis=None, keepdims=False):
    a = _dev(a)
    nz = U.call(np.not_equal, (a, a.dtype.type(0)))
    r = U.reduce("add", nz, axis=axis, dtype=np.int64, keepdims=keepdims)
    return int(r) if axis is None and not keepdims else r


@implements(np.argmax)
def argmax(a, axis=None, out=None, keepdims=False):
    a = _dev(a)
    src = a.ravel() if axis is None else a
    return U.reduce("argmax", src, axis=0 if axis is None else axis, out=out, keepdims=keepdims, arg=True)


@implements(np.argmin)
def argmin(a, axis=None, out=None, keepdims=False):
    a = _dev(a)
    src = a.ravel() if axis is None else a
    return U.reduce("argmin", src, axis=0 if axis is None else axis, out=out, keepdims=keepdims, arg=True)


@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    a = _dev(a)
    if axis is None:
        a, axis = a.ravel(), 0
    return U.accumulate("add", a, axis=axis, dtype=dtype, out=out)


# ------------------------------------------------------------------ joining / rearranging (GPU copies)
@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    arrays = [_dev(a) for a in arrays]
    if axis is None:
        arrays, axis = [a.ravel() for a in arrays], 0
    nd = arrays[0].ndim
    axis = axis % nd
    rt = np.dtype(dtype) if dtype is not None else np.result_type(*[a.dtype for a in arrays])
    shape = list(arrays[0].shape)
    shape[axis] = sum(a.shape[axis] for a in arrays)
    res = out if out is not None else _empty(tuple(shape), rt)
    pos = 0
    for a in arrays:
        sl = [slice(None)] * nd
        sl[axis] = slice(pos, pos + a.shape[axis])
        assign(_item(res, tuple(sl)), a)
        pos += a.shape[axis]
    return res


@implements(np.stack)
def stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    arrays = [_dev(a) for a in arrays]
    nd = arrays[0].ndim + 1
    axis = axis % nd
    return concatenate([_wrap(np.expand_dims(a._v, axis), a._gpuwm_mem) for a in arrays], axis=axis, out=out, dtype=dtype)


@implements(np.vstack)
def vstack(tup, *, dtype=None, casting="same_kind"):
    return concatenate([_views(np.atleast_2d)(_dev(a)) for a in tup], axis=0, dtype=dtype)


@implements(np.hstack)
def hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = [_views(np.atleast_1d)(_dev(a)) for a in tup]
    return concatenate(arrs, axis=0 if arrs[0].ndim == 1 else 1, dtype=dtype)


@implements(np.roll)
def roll(a, shift, axis=None):
    a = _dev(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)
    if not isinstance(axis, (tuple, list)):
        axis, shift = (axis,), (shift,)
    res = a
    for ax, sh in zip(axis, shift):
        ax %= res.ndim
        n = res.shape[ax]
        if n == 0:
            continue
        sh %= n
        out = _empty(res.shape, res.dtype)
        sl = lambda s: tuple(s if k == ax else slice(None) for k in range(res.ndim))
        assign(_item(out, sl(slice(sh, None))), _item(res, sl(slice(0, n - sh))))
        if sh:
            assign(_item(out, sl(slice(0, sh))), _item(res, sl(slice(n - sh, None))))
        res = out
    return res


@implements(np.pad)
def pad(array, pad_width, mode="constant", **kwargs):
    a = _dev(array)
    pw = np.broadcast_to(np.asarray(pad_width, dtype=np.int64), (a.ndim, 2))
    shape = tuple(int(a.shape[k] + pw[k, 0] + pw[k, 1]) for k in range(a.ndim))
    res = _empty(shape, a.dtype)
    inner = tuple(slice(int(pw[k, 0]), int(pw[k, 0] + a.shape[k])) for k in range(a.ndim))
    if mode == "constant":
        cv = kwargs.get("constant_values", 0)
        if np.ndim(cv) != 0:
            raise NotImplementedError("numpy.pad with per-axis constant_values on the Metal backend")
        res.fill(cv)
        assign(_item(res, inner), a)
        return res
    if mode == "edge":
        assign(_item(res, inner), a)
        cur = [slice(int(pw[k, 0]), int(pw[k, 0] + a.shape[k])) for k in range(a.ndim)]
        for k in range(a.ndim):
            lo, hi = int(pw[k, 0]), int(pw[k, 1])
            full = list(cur)
            if lo:
                full[k] = slice(0, lo)
                src = list(cur)
                src[k] = slice(lo, lo + 1)
                assign(_item(res, tuple(full)), _item(res, tuple(src)))
            if hi:
                full[k] = slice(lo + a.shape[k], lo + a.shape[k] + hi)
                src = list(cur)
                src[k] = slice(lo + a.shape[k] - 1, lo + a.shape[k])
                assign(_item(res, tuple(full)), _item(res, tuple(src)))
            cur[k] = slice(0, shape[k])
        return res
    raise NotImplementedError(f"numpy.pad(mode={mode!r}) on the Metal backend")


# ------------------------------------------------------------------ gathers and compaction
_GATHER = """
extern "C" __global__ void gather_axis(char* out, const char* src, const long long* idx,
                                       long long outer, long long nidx, long long inner, long long axis_len,
                                       long long so_outer, long long so_axis, long long so_inner,
                                       long long ss_outer, long long ss_axis, long long ss_inner, int itemsize,
                                       long long si_outer, long long si_axis, long long si_inner, int along) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = outer * nidx * inner;
    if (t >= total) return;
    long long in_ = t % inner, r = t / inner, j = r % nidx, o = r / nidx;
    long long k = along ? idx[o * si_outer + j * si_axis + in_ * si_inner] : idx[j];
    if (k < 0) k += axis_len;
    const char* s = src + o * ss_outer + k * ss_axis + in_ * ss_inner;
    char* d = out + o * so_outer + j * so_axis + in_ * so_inner;
    for (int b = 0; b < itemsize; ++b) d[b] = s[b];
}
"""


def _gather(a, idx, axis, along=False):
    """out[o, j, i] = a[o, idx[j] (or idx[o, j, i]), i] along ``axis``."""
    idx = _dev(idx).astype(np.int64, copy=False)
    if not idx.flags.c_contiguous and not along:
        idx = idx.copy()
    outer = int(np.prod(a.shape[:axis], dtype=np.int64))
    inner = int(np.prod(a.shape[axis + 1:], dtype=np.int64))
    nidx = idx.size if not along else idx.shape[axis]
    oshape = a.shape[:axis] + ((idx.shape if not along else (nidx,))) + a.shape[axis + 1:]
    src = a if a.flags.c_contiguous else a.copy()
    res = _empty(oshape, a.dtype)
    it = a.dtype.itemsize
    if along:
        ii = idx if idx.flags.c_contiguous else idx.copy()
        si = (nidx * inner, inner, 1)
    else:
        ii, si = idx, (0, 0, 0)
    total = outer * nidx * inner
    if total:
        k = K.kernel(_GATHER, "gather_axis")
        k(((total + 255) // 256,), (256,), (res, src, ii, outer, nidx, inner, a.shape[axis],
                                             nidx * inner * it, inner * it, it,
                                             a.shape[axis] * inner * it, inner * it, it, it,
                                             si[0], si[1], si[2], int(along)))
    return res


@implements(np.take)
def take(a, indices, axis=None, out=None, mode="raise"):
    a = _dev(a)
    if axis is None:
        a, axis = a.ravel(), 0
    return _gather(a, indices, axis % a.ndim)


@implements(np.take_along_axis)
def take_along_axis(arr, indices, axis=-1):
    arr = _dev(arr)
    axis = axis % arr.ndim
    idx = _dev(indices)
    if idx.shape[:axis] != arr.shape[:axis] or idx.shape[axis + 1:] != arr.shape[axis + 1:]:
        idx = broadcast(idx, arr.shape[:axis] + (idx.shape[axis],) + arr.shape[axis + 1:])
    return _gather(arr, idx, axis, along=True)


@implements(np.repeat)
def repeat(a, repeats, axis=None):
    """GPU gather; the source index of every output element follows from the
    shape and ``repeats`` alone (a device ``repeats`` is read like CuPy does)."""
    a = _dev(a)
    if axis is None:
        a, axis = a.ravel(), 0
    axis = axis % a.ndim
    n = a.shape[axis]
    r = np.asarray(repeats.get() if is_device(repeats) else repeats)
    if r.ndim == 0:
        r = np.full(n, int(r), dtype=np.int64)
    if r.shape != (n,) or (r < 0).any():
        raise ValueError(f"repeat: repeats {r.shape} do not match axis length {n} or are negative")
    idx = np.repeat(np.arange(n, dtype=np.int64), r.astype(np.int64))
    return _gather(a, asarray(idx), axis)


_SEARCHSORTED = """
template <class T> __device__ __forceinline__ bool npy_lt(T x, T y) { return x < y || (y != y && x == x); }
extern "C" __global__ void searchsorted_k(const T_* a, long long n, const T_* v, long long m, int right, long long* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    T_ x = v[i];
    long long lo = 0, hi = n;
    while (lo < hi) {
        long long mid = lo + ((hi - lo) >> 1);
        bool go_right = right ? !npy_lt(x, a[mid]) : npy_lt(a[mid], x);
        if (go_right) lo = mid + 1; else hi = mid;
    }
    out[i] = lo;
}
"""


@implements(np.searchsorted)
def searchsorted(a, v, side="left", sorter=None):
    """Binary search on the GPU with numpy's ordering (NaN sorts last)."""
    a = _dev(a)
    if a.ndim != 1:
        raise ValueError("searchsorted: a must be 1-D")
    if sorter is not None:
        a = take(a, _dev(sorter))
    scalar = not is_device(v) and np.ndim(v) == 0
    vv = _dev(v) if not scalar else asarray(np.asarray(v))
    dt = np.result_type(a.dtype, vv.dtype)
    aa = a.astype(dt) if a.dtype != dt else (a if a.flags.c_contiguous else a.copy())
    vf = vv.astype(dt).ravel() if vv.dtype != dt else (vv.ravel() if vv.flags.c_contiguous else vv.copy().ravel())
    out = _empty(vv.shape, np.int64)
    m = vf.size
    if m:
        k = K.kernel(f"#define T_ {K.ctype(dt)}\n" + _SEARCHSORTED, "searchsorted_k")
        k(((m + 255) // 256,), (256,), (aa, aa.size, vf, m, int(side == "right"), out))
    return out[()] if scalar else out


_COMPACT = """
extern "C" __global__ void flag_count(const char* m, int itemsize, long long n, unsigned int* counts, long long chunk) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long lo = b * chunk, hi = lo + chunk < n ? lo + chunk : n;
    if (lo >= n) return;
    unsigned int c = 0;
    for (long long i = lo; i < hi; ++i) { bool nz = false; for (int k = 0; k < itemsize; ++k) nz |= m[i * itemsize + k] != 0; c += nz; }
    counts[b] = c;
}
extern "C" __global__ void scan_counts(unsigned int* counts, long long nb, unsigned long long* total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long s = 0;
    for (long long b = 0; b < nb; ++b) { unsigned int c = counts[b]; counts[b] = (unsigned int)s; s += c; }
    total[0] = s;
}
extern "C" __global__ void scatter_idx(const char* m, int itemsize, long long n, const unsigned int* counts, long long chunk, long long* out) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long lo = b * chunk, hi = lo + chunk < n ? lo + chunk : n;
    if (lo >= n) return;
    unsigned long long p = counts[b];
    for (long long i = lo; i < hi; ++i) { bool nz = false; for (int k = 0; k < itemsize; ++k) nz |= m[i * itemsize + k] != 0; if (nz) out[p++] = i; }
}
"""


@implements(np.flatnonzero)
def flatnonzero(a):
    a = _dev(a)
    flat = a.ravel() if a.flags.c_contiguous else a.copy().ravel()
    if flat.dtype.kind == "f":
        flat = U.call(np.not_equal, (flat, flat.dtype.type(0)))   # -0.0 is zero
    n = flat.size
    chunk = 64
    nb = max(1, (n + chunk - 1) // chunk)
    counts = _empty(nb, np.uint32)
    total = _empty(1, np.uint64)
    it = flat.dtype.itemsize
    kc = K.kernel(_COMPACT, "flag_count")
    kc(((nb + 255) // 256,), (256,), (flat, it, n, counts, chunk))
    K.kernel(_COMPACT, "scan_counts")((1,), (1,), (counts, nb, total))
    cnt = int(total.get()[0])                          # result size is data-dependent (CuPy syncs too)
    res = _empty(cnt, np.int64)
    if cnt:
        K.kernel(_COMPACT, "scatter_idx")(((nb + 255) // 256,), (256,), (flat, it, n, counts, chunk, res))
    return res


@implements(np.nonzero)
def nonzero(a):
    a = _dev(a)
    flat_idx = flatnonzero(a)
    if a.ndim <= 1:
        return (flat_idx,)
    out = []
    rem = flat_idx
    strides = [int(np.prod(a.shape[k + 1:], dtype=np.int64)) for k in range(a.ndim)]
    for k in range(a.ndim):
        q = U.call(np.floor_divide, (rem, np.int64(strides[k])))
        out.append(q)
        rem = U.call(np.remainder, (rem, np.int64(strides[k])))
    return tuple(out)


@implements(np.argwhere)
def argwhere(a):
    return stack(nonzero(a), axis=1)


def advanced_getitem(a, key):
    """One advanced index (boolean mask or integer array), possibly after basic
    indices on leading axes, as a GPU gather."""
    if not isinstance(key, tuple):
        key = (key,)
    basic = lambda x: x is None or x is Ellipsis or isinstance(x, (slice, numbers.Integral, np.integer))
    adv = [p for p, x in enumerate(key) if not basic(x)]
    if len(adv) != 1:
        raise NotImplementedError("indexing with more than one advanced index on the Metal backend")
    pos = adv[0]
    if any(x is None or x is Ellipsis for x in key) or any(isinstance(x, (numbers.Integral, np.integer)) for x in key):
        raise NotImplementedError("advanced indexing mixed with integers/None/Ellipsis on the Metal backend")
    k = key[pos]
    k = _dev(k) if not isinstance(k, (list, tuple)) else asarray(np.asarray(k))
    lead = a[key[:pos]] if pos else a     # slices only: a view
    if k.dtype == np.bool_:
        if k.shape != lead.shape[pos:pos + k.ndim]:
            raise IndexError("boolean index shape mismatch")
        if pos:
            raise NotImplementedError("boolean index after slices on the Metal backend")
        src = lead if lead.flags.c_contiguous else lead.copy()
        flat = src.reshape((-1,) + lead.shape[k.ndim:])
        res = _gather(flat, flatnonzero(k), 0)
    else:
        res = _gather(lead, k, pos)
    rest = key[pos + 1:]
    return res[(slice(None),) * (pos + k.ndim) + rest] if rest else res


_SCATTER = """
extern "C" __global__ void scatter_rows(char* dst, const char* src, const long long* idx, long long nidx,
                                        long long inner_bytes, long long dst_row, long long src_row, int bcast_src) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nidx * inner_bytes) return;
    long long j = t / inner_bytes, b = t % inner_bytes;
    dst[idx[j] * dst_row + b] = src[(bcast_src ? 0 : j * src_row) + b];
}
"""


def advanced_setitem(a, key, value):
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) != 1:
        raise NotImplementedError("advanced assignment with more than one index on the Metal backend")
    k = key[0]
    k = _dev(k) if not isinstance(k, (list, tuple)) else asarray(np.asarray(k))
    if k.dtype == np.bool_:
        if k.shape == a.shape and not is_device(value):
            # a[mask] = scalar  ->  where
            where_(k, value, a, out=a)
            return
        idx = flatnonzero(k)
        flat_shape = (-1,) + a.shape[k.ndim:]
    else:
        idx = k.astype(np.int64, copy=False).ravel()
        flat_shape = a.shape
    if not a.flags.c_contiguous:
        raise NotImplementedError("advanced assignment into a non-contiguous array on the Metal backend")
    dst = a.reshape(flat_shape)
    inner = int(np.prod(dst.shape[1:], dtype=np.int64)) * a.dtype.itemsize
    n = idx.size
    if n == 0:
        return
    if is_device(value):
        v = value.astype(a.dtype, copy=False)
        v = v if v.flags.c_contiguous else v.copy()
        bc = int(v.size * a.dtype.itemsize == inner and n > 1)
    else:
        v = C.full((max(1, inner // a.dtype.itemsize),), value, dtype=a.dtype)
        bc = 1
    K.kernel(_SCATTER, "scatter_rows")(((n * inner + 255) // 256,), (256,), (dst, v, idx, n, inner, inner, inner, bc))



def _host_index(key):
    def down(k):
        if isinstance(k, ndarray):
            return k._v
        if isinstance(k, tuple):
            return tuple(down(x) for x in k)
        if isinstance(k, list):
            return [down(x) for x in k]
        return k
    return down(key)


def host_getitem(a, key):
    """Advanced indexing the Metal gather does not cover: NumPy over unified memory."""
    from .. import runtime
    _notice("advanced indexing (mixed form)")
    runtime.host_sync()
    return asarray(a._v[_host_index(key)])


def host_setitem(a, key, value):
    from .. import runtime
    _notice("advanced assignment (mixed form)")
    runtime.host_sync()
    a._v[_host_index(key)] = value._v if isinstance(value, ndarray) else value


def _notice(what):
    import sys
    if what not in _TOLD:
        _TOLD.add(what)
        print(f"[arwen-metal] {what}: computed on the CPU over unified memory (no Metal kernel)", file=sys.stderr)
