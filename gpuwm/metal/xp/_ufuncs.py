"""NumPy ufuncs on device arrays: elementwise maps, reductions and accumulations
as generated GPU kernels, with NumPy 2 type resolution (NEP 50)."""
from __future__ import annotations

import numbers

import numpy as np

from .. import runtime
from . import _kernels as K
from ._core import _empty, _np_ndarray, is_device, ndarray


def _strides_for(a, shape):
    """Byte strides of ``a`` broadcast to ``shape`` (0 on broadcast axes)."""
    nd = len(shape)
    st = [0] * nd
    off = nd - a.ndim
    for i in range(a.ndim):
        st[off + i] = 0 if (a.shape[i] == 1 and shape[off + i] != 1) else a.strides[i]
    return st


def elementwise(expr, outs, ins, scalars, preamble=""):
    """Run ``expr`` over the broadcast shape of ``outs`` (outputs define the shape)."""
    shape = outs[0].shape
    n = 1
    for s in shape:
        n *= s
    if n == 0:
        return
    for a in list(outs) + list(ins):
        if not is_device(a):
            raise TypeError("elementwise operand is not a device array")
    strides = [_strides_for(a, shape) for a in list(outs) + list(ins)]
    cshape, cstrides = K.collapse(shape, strides)
    if len(outs) + len(ins) > K.MAXOPS:
        raise NotImplementedError(f"more than {K.MAXOPS} array operands in one elementwise kernel")
    src = K.elementwise_source(expr, tuple(K.ctype(o.dtype) for o in outs), tuple(K.ctype(a.dtype) for a in ins),
                               tuple(K.ctype(dt) for dt, _ in scalars), preamble)
    k = K.kernel(src, "ew")
    args = list(outs) + list(ins) + [v for _, v in scalars] + [K.pack_layout(n, cshape, cstrides)]
    K.launch_1d(k, n, args)


# ------------------------------------------------------------------ expressions
_UNARY_FLOAT = {
    "sqrt": "sqrt(x0)", "cbrt": "cbrt(x0)", "exp": "exp(x0)", "exp2": "exp2(x0)", "expm1": "expm1(x0)",
    "log": "log(x0)", "log2": "log2(x0)", "log10": "log10(x0)", "log1p": "log1p(x0)", "sin": "sin(x0)",
    "cos": "cos(x0)", "tan": "tan(x0)", "arcsin": "asin(x0)", "arccos": "acos(x0)", "arctan": "atan(x0)",
    "sinh": "sinh(x0)", "cosh": "cosh(x0)", "tanh": "tanh(x0)", "arcsinh": "asinh(x0)", "arccosh": "acosh(x0)",
    "arctanh": "atanh(x0)", "floor": "floor(x0)", "ceil": "ceil(x0)", "trunc": "trunc(x0)", "rint": "rint(x0)",
    "fabs": "fabs(x0)", "reciprocal": "(T)1 / x0",
    "deg2rad": "x0 * (T)0.017453292519943295", "radians": "x0 * (T)0.017453292519943295",
    "rad2deg": "x0 * (T)57.29577951308232", "degrees": "x0 * (T)57.29577951308232",
}
_BINARY_FLOAT = {
    "arctan2": "atan2(x0, x1)", "hypot": "hypot(x0, x1)", "copysign": "copysign(x0, x1)", "power": "pow(x0, x1)",
    "fmod": "fmod(x0, x1)", "true_divide": "x0 / x1", "divide": "x0 / x1",
}
_CMP = {"greater": ">", "greater_equal": ">=", "less": "<", "less_equal": "<=", "equal": "==", "not_equal": "!="}
_FLOAT_DIVMOD = """
template <class T> __device__ __forceinline__ T npy_divmod(T a, T b, T* modulus) {
    T mod = fmod(a, b);
    if (!(b != (T)0)) { *modulus = mod; return a / b; }
    T div = (a - mod) / b;
    if (mod != (T)0) {
        if ((b < (T)0) != (mod < (T)0)) { mod += b; div -= (T)1; }
    } else {
        mod = copysign((T)0, b);
    }
    T floordiv;
    if (div != (T)0) {
        floordiv = floor(div);
        if (div - floordiv > (T)0.5) floordiv += (T)1;
    } else {
        floordiv = copysign((T)0, a / b);
    }
    *modulus = mod;
    return floordiv;
}
"""


def _expr(name, loop, nin):
    """C statement computing outputs o0.. from loop-typed inputs x0.. (loop dtypes ``loop``)."""
    dt = np.dtype(loop[0])
    T = K.ctype(dt)
    f, i, b = dt.kind == "f", dt.kind in "iu", dt.kind == "b"
    pre = ""
    if name in ("add",):
        e = "x0 || x1" if b else "x0 + x1"
    elif name == "subtract":
        e = "x0 - x1"
    elif name == "multiply":
        e = "x0 && x1" if b else "x0 * x1"
    elif name in _CMP:
        e = f"x0 {_CMP[name]} x1"
    elif name == "maximum":
        e = "(x0 >= x1 || x0 != x0) ? x0 : x1" if f else "(x0 >= x1) ? x0 : x1"
    elif name == "minimum":
        e = "(x0 <= x1 || x0 != x0) ? x0 : x1" if f else "(x0 <= x1) ? x0 : x1"
    elif name == "fmax":
        e = "(x0 >= x1 || x1 != x1) ? x0 : x1" if f else "(x0 >= x1) ? x0 : x1"
    elif name == "fmin":
        e = "(x0 <= x1 || x1 != x1) ? x0 : x1" if f else "(x0 <= x1) ? x0 : x1"
    elif name == "clip":
        mx = "((x0 >= x1 || x0 != x0) ? x0 : x1)" if f else "((x0 >= x1) ? x0 : x1)"
        e = f"(({mx}) <= x2 || ({mx}) != ({mx})) ? ({mx}) : x2" if f else f"(({mx}) <= x2) ? ({mx}) : x2"
    elif name == "negative":
        e = "-x0"
    elif name == "positive":
        e = "x0"
    elif name == "absolute":
        e = "fabs(x0)" if f else ("x0" if b or dt.kind == "u" else "(x0 < 0 ? -x0 : x0)")
    elif name == "sign":
        e = "(x0 != x0) ? x0 : (x0 > (T)0 ? (T)1 : (x0 < (T)0 ? (T)-1 : (T)0))"
    elif name == "square":
        e = "x0 * x0"
    elif name in ("isnan",):
        e = "x0 != x0" if f else "false"
    elif name == "isinf":
        e = "isinf(x0)" if f else "false"
    elif name == "isfinite":
        e = "isfinite(x0)" if f else "true"
    elif name == "signbit":
        e = "signbit(x0)" if f else ("x0 < 0" if dt.kind == "i" else "false")
    elif name == "logical_and":
        e = "x0 && x1"
    elif name == "logical_or":
        e = "x0 || x1"
    elif name == "logical_xor":
        e = "(bool)x0 != (bool)x1"
    elif name == "logical_not":
        e = "!x0"
    elif name in ("bitwise_and", "bitwise_or", "bitwise_xor") and (i or b):
        e = "x0 " + {"bitwise_and": "&", "bitwise_or": "|", "bitwise_xor": "^"}[name] + " x1"
    elif name in ("invert", "bitwise_not") and (i or b):
        e = "!x0" if b else "~x0"
    elif name == "left_shift" and i:
        e = "x0 << x1"
    elif name == "right_shift" and i:
        e = "x0 >> x1"
    elif name in ("floor_divide", "remainder", "mod", "divmod") and f:
        pre = _FLOAT_DIVMOD
        if name == "floor_divide":
            return pre, f"{{ {T} m; o0 = npy_divmod(({T})x0, ({T})x1, &m); }}"
        if name == "divmod":
            return pre, f"{{ {T} m; o0 = npy_divmod(({T})x0, ({T})x1, &m); o1 = m; }}"
        return pre, f"{{ {T} m; npy_divmod(({T})x0, ({T})x1, &m); o0 = m; }}"
    elif name == "floor_divide" and i:
        e = "(x1 == 0) ? (T)0 : ((x0 % x1 != 0 && ((x0 < 0) != (x1 < 0))) ? x0 / x1 - 1 : x0 / x1)" if dt.kind == "i" \
            else "(x1 == 0) ? (T)0 : x0 / x1"
    elif name in ("remainder", "mod") and i:
        e = "(x1 == 0) ? (T)0 : (((x0 % x1) != 0 && (((x0 % x1) < 0) != (x1 < 0))) ? (x0 % x1) + x1 : (x0 % x1))" \
            if dt.kind == "i" else "(x1 == 0) ? (T)0 : x0 % x1"
    elif name == "fmod" and i:
        e = "(x1 == 0) ? (T)0 : x0 % x1"
    elif name == "power" and i:
        pre = """template <class T> __device__ __forceinline__ T ipow(T b, T e) {
    if (e < 0) return (T)0; T r = 1; while (e) { if (e & 1) r *= b; b *= b; e >>= 1; } return r; }
"""
        e = "ipow(x0, x1)"
    elif f and name in _UNARY_FLOAT:
        e = _UNARY_FLOAT[name].replace("(T)", f"({T})")
    elif f and name in _BINARY_FLOAT:
        e = _BINARY_FLOAT[name]
    else:
        raise NotImplementedError(f"ufunc {name} for dtype {dt} on the Metal backend")
    return pre, "o0 = " + e.replace("(T)", f"({T})") + ";"


def _scalar_value(x):
    if isinstance(x, np.generic):
        return x.item(), x.dtype
    if isinstance(x, _np_ndarray) and x.ndim == 0 and not is_device(x):
        return x.item(), x.dtype
    if isinstance(x, (bool, np.bool_)):
        return bool(x), np.dtype(np.bool_)
    if isinstance(x, numbers.Integral):
        return int(x), int
    if isinstance(x, numbers.Real):
        return float(x), float
    raise TypeError(f"unsupported operand type {type(x).__name__} for a device array operation")


def call(ufunc, inputs, out=None, dtype=None, where=True, casting="same_kind", order="K", signature=None, subok=True):
    name = ufunc.__name__
    if where is not True:
        raise NotImplementedError(f"{name}(where=...) on the Metal backend")
    arrs, kinds = [], []
    for x in inputs:
        if is_device(x):
            arrs.append(x)
            kinds.append(x.dtype)
        elif isinstance(x, _np_ndarray) and x.ndim > 0:
            raise TypeError(f"{name}: unsupported operand: host numpy.ndarray mixed with device arrays "
                            "(convert with asarray first, as CuPy requires)")
        else:
            v, t = _scalar_value(x)
            arrs.append(("scalar", v, t))
            kinds.append(t)
    nout = ufunc.nout
    outs = list(out) if isinstance(out, tuple) else ([out] if out is not None else [None] * nout)
    sig = None
    if dtype is not None:
        sig = (None,) * ufunc.nin + (np.dtype(dtype),) * nout if signature is None else signature
    extra = {"signature": sig} if sig is not None else {}
    try:
        loop = ufunc.resolve_dtypes(tuple(kinds) + tuple(None if o is None else o.dtype for o in outs),
                                    casting=casting, **extra)
    except (TypeError, np._core._exceptions._UFuncNoLoopError if hasattr(np, "_core") else TypeError):
        # an out= of another dtype: resolve the loop without it and cast on store (numpy's unsafe out path)
        loop = ufunc.resolve_dtypes(tuple(kinds) + (None,) * nout, **extra)
    in_loop, out_loop = loop[:ufunc.nin], loop[ufunc.nin:]
    shapes = [a.shape for a in arrs if not isinstance(a, tuple)] + [o.shape for o in outs if o is not None]
    shape = np.broadcast_shapes(*shapes) if shapes else ()
    res = []
    for k in range(nout):
        o = outs[k]
        if o is None:
            o = _empty(shape, out_loop[k])
        elif not is_device(o):
            raise TypeError(f"{name}: out= must be a device array")
        elif o.shape != shape:
            raise ValueError(f"{name}: output shape {o.shape} does not match broadcast shape {shape}")
        res.append(o)
    pre, stmt = _expr(name, in_loop, ufunc.nin)
    dev_ins, scalars, loads = [], [], []
    for j, a in enumerate(arrs):
        lt = K.ctype(in_loop[j])
        if isinstance(a, tuple):
            loads.append(f"const {lt} x{j} = s{len(scalars)};")
            scalars.append((np.dtype(in_loop[j]), _cast(a[1], in_loop[j])))
        else:
            loads.append(f"const {lt} x{j} = ({lt})a{len(dev_ins)};")
            dev_ins.append(a)
    # outputs: loop-typed temporaries cast to the output arrays' dtypes
    stmt_out = stmt
    for k in range(nout):
        stmt_out = stmt_out.replace(f"o{k}", f"r{k}")
    decl = " ".join(f"{K.ctype(out_loop[k])} r{k};" for k in range(nout))
    store = " ".join(f"o{k} = ({K.ctype(res[k].dtype)})r{k};" for k in range(nout))
    expr = " ".join(loads) + f" {decl} {stmt_out} {store}"
    elementwise(expr, res, dev_ins, scalars, pre)
    return res[0] if nout == 1 else tuple(res)


def _cast(v, dt):
    dt = np.dtype(dt)
    if dt.kind == "f":
        return float(v)
    if dt.kind == "b":
        return bool(v)
    return int(v)


# ------------------------------------------------------------------ reductions
_RED = {
    # name: (init(T), step, merge)
    "add": ("({A})0", "a.v = a.v + ({A})x;", "a.v = a.v + b.v;"),
    "multiply": ("({A})1", "a.v = a.v * ({A})x;", "a.v = a.v * b.v;"),
    "logical_or": ("false", "a.v = a.v || (x != 0);", "a.v = a.v || b.v;"),
    "logical_and": ("true", "a.v = a.v && (x != 0);", "a.v = a.v && b.v;"),
    "bitwise_or": ("({A})0", "a.v = a.v | ({A})x;", "a.v = a.v | b.v;"),
    "bitwise_and": ("({A})~({A})0", "a.v = a.v & ({A})x;", "a.v = a.v & b.v;"),
}


def _red_minmax(name, f):
    gt = ">" if name in ("maximum", "fmax") else "<"
    if name in ("maximum", "minimum") and f:
        step = f"if (a.j < 0) {{ a.v = ({{A}})x; a.j = 0; }} else if (a.v == a.v && (x != x || ({{A}})x {gt} a.v)) a.v = ({{A}})x;"
        merge = f"if (a.j < 0) {{ a.v = b.v; a.j = b.j; }} else if (b.j >= 0 && a.v == a.v && (b.v != b.v || b.v {gt} a.v)) a.v = b.v;"
    elif f:   # fmax/fmin ignore NaN
        step = f"if (x == x && (a.j < 0 || a.v != a.v || ({{A}})x {gt} a.v)) {{ a.v = ({{A}})x; a.j = 0; }} else if (a.j < 0) {{ a.v = ({{A}})x; a.j = 0; }}"
        merge = f"if (b.j >= 0 && (a.j < 0 || a.v != a.v || (b.v == b.v && b.v {gt} a.v))) {{ a.v = b.v; a.j = b.j; }}"
    else:
        step = f"if (a.j < 0 || ({{A}})x {gt} a.v) {{ a.v = ({{A}})x; a.j = 0; }}"
        merge = f"if (b.j >= 0 && (a.j < 0 || b.v {gt} a.v)) {{ a.v = b.v; a.j = b.j; }}"
    return "({A})0", step, merge


def _red_arg(name, f):
    gt = ">" if name == "argmax" else "<"
    # first occurrence; NaN wins (numpy): once a NaN is held, keep it
    step = (f"if (a.j < 0) {{ a.v = ({{A}})x; a.j = j; }} else if (a.v == a.v && ((x != x) || ({{A}})x {gt} a.v)) "
            f"{{ a.v = ({{A}})x; a.j = j; }}") if f else \
        f"if (a.j < 0 || ({{A}})x {gt} a.v) {{ a.v = ({{A}})x; a.j = j; }}"
    merge = (f"if (b.j >= 0) {{ if (a.j < 0) {{ a.v = b.v; a.j = b.j; }} else if (a.v != a.v) {{ if (b.v != b.v && b.j < a.j) a.j = b.j; }} "
             f"else if (b.v != b.v || b.v {gt} a.v || (b.v == a.v && b.j < a.j)) {{ a.v = b.v; a.j = b.j; }} }}") if f else \
        f"if (b.j >= 0) {{ if (a.j < 0 || b.v {gt} a.v || (b.v == a.v && b.j < a.j)) {{ a.v = b.v; a.j = b.j; }} }}"
    return "({A})0", step, merge


def _norm_axes(axis, nd):
    if axis is None:
        return tuple(range(nd))
    if isinstance(axis, (numbers.Integral, np.integer)):
        axis = (int(axis),)
    out = []
    for a in axis:
        a = int(a)
        if a < -nd or a >= nd:
            raise np.exceptions.AxisError(a, nd)
        out.append(a % nd)
    return tuple(sorted(set(out)))


def reduce(name, a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, arg=False):
    if where is not True:
        raise NotImplementedError(f"{name}.reduce(where=...) on the Metal backend")
    if not is_device(a):
        a = __import__("gpuwm.metal.xp._core", fromlist=["asarray"]).asarray(a)
    axes = _norm_axes(axis, a.ndim)
    f = a.dtype.kind == "f"
    if arg:
        acc_dt, out_dt = a.dtype, np.dtype(np.int64)
    elif name in ("logical_or", "logical_and"):
        acc_dt = out_dt = np.dtype(np.bool_)
    elif name in ("add", "multiply"):
        if dtype is not None:
            acc_dt = out_dt = np.dtype(dtype)
        else:
            out_dt = getattr(np, name).reduce(np.zeros(1, dtype=a.dtype)).dtype
            acc_dt = out_dt
    else:
        acc_dt = out_dt = np.dtype(dtype) if dtype is not None else a.dtype
    oshape = tuple(s for k, s in enumerate(a.shape) if k not in axes)
    kshape = tuple(1 if k in axes else s for k, s in enumerate(a.shape))
    rshape = tuple(a.shape[k] for k in axes)
    rn = 1
    for s in rshape:
        rn *= s
    if out is None:
        res = _empty(kshape if keepdims else oshape, out_dt)
    else:
        res = out
    if rn == 0:
        if name in ("add", "multiply", "logical_or", "logical_and", "bitwise_or", "bitwise_and") and not arg:
            ident = {"add": 0, "multiply": 1, "logical_or": False, "logical_and": True, "bitwise_or": 0, "bitwise_and": -1}[name]
            res.fill(ident if initial is None else initial)
            return res
        raise ValueError(f"zero-size array to reduction operation {name} which has no identity")
    on = 1
    for s in oshape:
        on *= s
    A = K.ctype(acc_dt)
    if arg:
        init, step, merge = _red_arg(name, f)
    elif name in ("maximum", "minimum", "fmax", "fmin"):
        init, step, merge = _red_minmax(name, f)
    elif name in _RED:
        init, step, merge = _RED[name]
    else:
        raise NotImplementedError(f"{name}.reduce on the Metal backend")
    init, step, merge = (s.replace("{A}", A) for s in (init, step, merge))
    if not arg and name not in ("maximum", "minimum", "fmax", "fmin"):
        step = step.replace("a.v =", "a.j = 0; a.v =") if "a.j" not in step else step
    final = "a.j" if arg else "a.v"
    src = K.reduction_source(A, K.ctype(a.dtype), K.ctype(out_dt), init, step, merge, final, arg)
    # output-space layout: op0 = result (over oshape), op1 = input (over oshape)
    tgt = res if not keepdims or out is not None else res
    rv = res.reshape(oshape) if res.shape != oshape else res
    ost = [list(rv.strides), [a.strides[k] for k in range(a.ndim) if k not in axes]]
    oshp, ost = K.collapse(oshape, ost) if oshape else ((), [(), ()])
    rst = [[a.strides[k] for k in axes]]
    rshp, rst = K.collapse(rshape, rst)
    LO = K.pack_layout(on, oshp, ost)
    LR = K.pack_layout(rn, rshp, rst)
    base = a
    if on >= 2048 or rn <= 64:
        k = K.kernel(src, "red_thread")
        k(((on + 127) // 128,), (128,), (rv, base, LO, LR))
    else:
        k = K.kernel(src, "red_block")
        k((on,), (256,), (rv, base, LO, LR))
    if initial is not None and not arg:
        call(getattr(np, name), (res, initial), out=res)
    return res


def accumulate(name, a, axis=0, dtype=None, out=None):
    if name not in ("add", "multiply", "maximum", "minimum"):
        raise NotImplementedError(f"{name}.accumulate on the Metal backend")
    if not is_device(a):
        raise TypeError("accumulate on a non-device array")
    axis = int(axis) % max(a.ndim, 1)
    out_dt = np.dtype(dtype) if dtype is not None else (
        getattr(np, name).accumulate(np.zeros(1, dtype=a.dtype)).dtype)
    res = out if out is not None else _empty(a.shape, out_dt)
    A = K.ctype(out_dt)
    op = {"add": "acc + x", "multiply": "acc * x", "maximum": "(x > acc || x != x) ? x : acc",
          "minimum": "(x < acc || x != x) ? x : acc"}[name]
    init = {"add": f"({A})0", "multiply": f"({A})1", "maximum": "x", "minimum": "x"}[name]
    if name in ("maximum", "minimum"):
        op = op if a.dtype.kind == "f" else op.replace(" || x != x", "")
    src = K.SCAN_SOURCE_TMPL.format(lay=K._layout_struct(), offs=K._OFFSETS, acc_t=A, in_t=K.ctype(a.dtype),
                                    out_t=K.ctype(out_dt), init=(f"({A})*(const {K.ctype(a.dtype)}*)(in + off[1])"
                                                                  if init == "x" else init), op=op)
    rows_shape = tuple(s for k, s in enumerate(a.shape) if k != axis)
    rn = 1
    for s in rows_shape:
        rn *= s
    if rn == 0 or a.shape[axis] == 0:
        return res
    ost = [[res.strides[k] for k in range(a.ndim) if k != axis], [a.strides[k] for k in range(a.ndim) if k != axis]]
    shp, ost = K.collapse(rows_shape, ost) if rows_shape else ((), [(), ()])
    LO = K.pack_layout(rn, shp, ost)
    k = K.kernel(src, "scan_rows")
    k(((rn + 127) // 128,), (128,), (res, a, LO, a.shape[axis], res.strides[axis], a.strides[axis]))
    return res


# ------------------------------------------------------------------ dispatch from __array_ufunc__
def dispatch(ufunc, method, inputs, kwargs):
    out = kwargs.pop("out", None)
    if method == "__call__":
        return call(ufunc, inputs, out=out, **kwargs)
    if method == "reduce":
        a = inputs[0]
        o = out[0] if isinstance(out, tuple) else out
        return reduce(ufunc.__name__, a, axis=kwargs.get("axis", 0), dtype=kwargs.get("dtype"), out=o,
                      keepdims=kwargs.get("keepdims", False), initial=kwargs.get("initial"),
                      where=kwargs.get("where", True))
    if method == "accumulate":
        o = out[0] if isinstance(out, tuple) else out
        return accumulate(ufunc.__name__, inputs[0], axis=kwargs.get("axis", 0), dtype=kwargs.get("dtype"), out=o)
    raise NotImplementedError(f"{ufunc.__name__}.{method} on the Metal backend")
