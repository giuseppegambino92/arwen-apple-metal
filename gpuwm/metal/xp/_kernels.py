"""GPU kernels behind the array API: elementwise maps, reductions, scans.

Every kernel is generated as CUDA C++ and goes through the same translator as
gpuwm's own kernels, so float64 has the same exact binary64 semantics.  One
kernel per (operation, dtypes): shapes and strides travel in the argument
struct, so a new shape never recompiles.
"""
from __future__ import annotations

import functools
import struct

import numpy as np

from .. import runtime

MAXD = 8          # dimensions after collapsing
MAXOPS = 48       # array operands of one kernel (the layout struct is sized per kernel)

CTYPE = {
    np.dtype(np.bool_): "bool", np.dtype(np.int8): "signed char", np.dtype(np.uint8): "unsigned char",
    np.dtype(np.int16): "short", np.dtype(np.uint16): "unsigned short", np.dtype(np.int32): "int",
    np.dtype(np.uint32): "unsigned int", np.dtype(np.int64): "long long", np.dtype(np.uint64): "unsigned long long",
    np.dtype(np.float32): "float", np.dtype(np.float64): "double",
}


def ctype(dt) -> str:
    dt = np.dtype(dt)
    try:
        return CTYPE[dt.newbyteorder("=") if dt.byteorder not in "=|" else dt]
    except KeyError:
        raise TypeError(f"dtype {dt} is not supported on the Metal backend") from None


@functools.lru_cache(maxsize=None)
def _module(source: str) -> runtime.Module:
    # CuPy compiles its elementwise/reduction kernels with -ftz=true: so do we
    from .cuda import effective_options
    return runtime.Module(source, effective_options(source, ("-std=c++17",)), cupy_raw_module=True)


def kernel(source: str, name: str) -> runtime.Kernel:
    return _module(source).get_function(name)


def _layout_struct(nops):
    """The shape/stride block of one kernel, sized for exactly ``nops`` operands."""
    if not 1 <= nops <= MAXOPS:
        raise NotImplementedError(f"{nops} array operands in one generated kernel (limit {MAXOPS})")
    return f"struct Lay {{ long long n; int nd; int pad; long long shape[{MAXD}]; long long st[{nops}][{MAXD}]; }};"


def pack_layout(n, shape, strides_list, nops) -> bytes:
    nd = len(shape)
    if nd > MAXD:
        raise ValueError(f"more than {MAXD} dimensions after collapsing")
    if len(strides_list) > nops:
        raise ValueError(f"{len(strides_list)} operand stride rows for a layout of {nops}")
    shp = list(shape) + [1] * (MAXD - nd)
    out = struct.pack(f"<qii{MAXD}q", n, nd, 0, *shp)
    for k in range(nops):
        st = list(strides_list[k]) if k < len(strides_list) else []
        st = st + [0] * (MAXD - len(st))
        out += struct.pack(f"<{MAXD}q", *st)
    return out


def collapse(shape, strides_list):
    """Merge adjacent dimensions that are contiguous for every operand (C order)."""
    dims = [(s, [st[k] for st in strides_list]) for k, s in enumerate(shape) if s != 1]
    if not dims:
        return (), [()] * len(strides_list)
    out = [list(dims[0])]
    for s, sts in dims[1:]:
        ps, psts = out[-1]
        if all(psts[i] == sts[i] * s for i in range(len(sts))):
            out[-1] = [ps * s, sts]
        else:
            out.append([s, sts])
    shape = tuple(d[0] for d in out)
    strides = [tuple(d[1][i] for d in out) for i in range(len(strides_list))]
    return shape, strides


_OFFSETS = """
__device__ __forceinline__ void lay_offsets(const Lay& L, long long i, long long* off, int nops) {
    for (int k = 0; k < nops; ++k) off[k] = 0;
    for (int d = L.nd - 1; d >= 0; --d) {
        long long s = L.shape[d];
        long long q = i / s, r = i - q * s;
        i = q;
        for (int k = 0; k < nops; ++k) off[k] += r * L.st[k][d];
    }
}
"""


@functools.lru_cache(maxsize=None)
def elementwise_source(expr: str, out_types: tuple, in_types: tuple, scalar_types: tuple, preamble: str = "") -> str:
    """CUDA source of a strided elementwise map.

    ``expr`` is C code assigning outputs ``o0..`` from inputs ``a0..`` (arrays)
    and ``s0..`` (scalars); ``i`` is the flat element index.
    """
    nops = len(out_types) + len(in_types)
    params = [f"char* p{k}" for k in range(len(out_types))]
    params += [f"const char* p{len(out_types) + k}" for k in range(len(in_types))]
    params += [f"{t} s{k}" for k, t in enumerate(scalar_types)]
    params += ["Lay L"]
    loads = "".join(f"        const {t} a{k} = *(const {t}*)(p{len(out_types) + k} + off[{len(out_types) + k}]);\n"
                    for k, t in enumerate(in_types))
    decls = "".join(f"        {t} o{k};\n" for k, t in enumerate(out_types))
    stores = "".join(f"        *({t}*)(p{k} + off[{k}]) = o{k};\n" for k, t in enumerate(out_types))
    return f"""{_layout_struct(max(nops, 1))}
{_OFFSETS}
{preamble}
extern "C" __global__ void ew({", ".join(params)}) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long step = (long long)blockDim.x * gridDim.x;
    long long off[{max(nops, 1)}];
    for (; i < L.n; i += step) {{
        lay_offsets(L, i, off, {nops});
{loads}{decls}        {{ {expr} }}
{stores}    }}
}}
"""


def launch_1d(k: runtime.Kernel, n: int, args, block=256):
    blocks = max(1, min((n + block - 1) // block, 65535 * 8))
    k((blocks,), (block,), args)


@functools.lru_cache(maxsize=None)
def reduction_source(acc_t: str, in_t: str, out_t: str, init: str, step: str, merge: str, final: str,
                     with_index: bool) -> str:
    """Reduce the trailing ``R`` elements of every output position.

    Layout: operand 0 = output (output dims), operand 1 = input over output dims,
    operand 2 = input over reduced dims (L2).  ``step(acc, x, j)`` folds element
    ``x`` (flat reduced index ``j``); ``merge(acc, other)`` combines partials in a
    fixed tree order (deterministic)."""
    idx = "long long" if with_index else "int"
    return f"""{_layout_struct(2)}
{_OFFSETS}
struct Acc {{ {acc_t} v; {idx} j; }};
__device__ __forceinline__ void acc_step(Acc& a, {in_t} x, long long j) {{ {step} }}
__device__ __forceinline__ void acc_merge(Acc& a, const Acc& b) {{ {merge} }}
extern "C" __global__ void red_thread(char* out, const char* in, Lay LO, Lay LR) {{
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= LO.n) return;
    long long off[2];
    lay_offsets(LO, o, off, 2);
    Acc a; a.v = {init}; a.j = -1;
    for (long long j = 0; j < LR.n; ++j) {{
        long long r[1];
        lay_offsets(LR, j, r, 1);
        acc_step(a, *(const {in_t}*)(in + off[1] + r[0]), j);
    }}
    *({out_t}*)(out + off[0]) = ({out_t})({final});
}}
extern "C" __global__ void __launch_bounds__(256) red_block(char* out, const char* in, Lay LO, Lay LR) {{
    __shared__ {acc_t} sv[256];
    __shared__ {idx} sj[256];
    long long o = blockIdx.x;
    long long off[2];
    lay_offsets(LO, o, off, 2);
    Acc a; a.v = {init}; a.j = -1;
    for (long long j = threadIdx.x; j < LR.n; j += blockDim.x) {{
        long long r[1];
        lay_offsets(LR, j, r, 1);
        acc_step(a, *(const {in_t}*)(in + off[1] + r[0]), j);
    }}
    sv[threadIdx.x] = a.v; sj[threadIdx.x] = a.j;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) {{
            Acc x; x.v = sv[threadIdx.x]; x.j = sj[threadIdx.x];
            Acc y; y.v = sv[threadIdx.x + s]; y.j = sj[threadIdx.x + s];
            acc_merge(x, y);
            sv[threadIdx.x] = x.v; sj[threadIdx.x] = x.j;
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) {{ Acc a2; a2.v = sv[0]; a2.j = sj[0]; a = a2; *({out_t}*)(out + off[0]) = ({out_t})({final}); }}
}}
"""


SCAN_SOURCE_TMPL = """{lay}
{offs}
// inclusive scan along the last (collapsed) axis; one thread per row, sequential (deterministic)
extern "C" __global__ void scan_rows(char* out, const char* in, Lay LO, long long len, long long so, long long si) {{
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= LO.n) return;
    long long off[2];
    lay_offsets(LO, row, off, 2);
    {acc_t} acc = {init};
    for (long long j = 0; j < len; ++j) {{
        {acc_t} x = ({acc_t})*(const {in_t}*)(in + off[1] + j * si);
        acc = {op};
        *({out_t}*)(out + off[0] + j * so) = ({out_t})acc;
    }}
}}
"""
