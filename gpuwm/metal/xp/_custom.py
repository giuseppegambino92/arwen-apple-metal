"""CuPy's user-kernel classes on Metal: RawModule, RawKernel, ElementwiseKernel,
ReductionKernel.  All of them end in CUDA source run through the translator."""
from __future__ import annotations

import re

import numpy as np

from .. import runtime
from . import _kernels as K
from ._core import _empty, asarray, broadcast, is_device
from . import _ufuncs as U

# ------------------------------------------------------------------ RawModule / RawKernel / driver Function


import weakref

_BY_PTR = weakref.WeakValueDictionary()


class Function:
    """A loaded kernel, callable like cupy.RawKernel / cupy.cuda.Function."""

    def __init__(self, kernel: runtime.Kernel):
        self._k = kernel
        self.kernel = kernel
        _BY_PTR[id(kernel)] = kernel      # driver.funcGetAttribute(attr, function.ptr)

    def __call__(self, grid, block, args, shared_mem=0, stream=None, enable_cooperative_groups=False):
        if enable_cooperative_groups:
            raise NotImplementedError("cooperative launches on the Metal backend")
        self._k.launch(grid, block, tuple(args), shared_mem)

    @property
    def max_threads_per_block(self):
        return self._k.max_threads

    @property
    def num_regs(self):
        return 0

    @property
    def local_size_bytes(self):
        return 0

    @property
    def shared_size_bytes(self):
        return self._k.pso.static_threadgroup_memory

    @property
    def const_size_bytes(self):
        return 0

    @property
    def max_dynamic_shared_size_bytes(self):
        return 32768 - self._k.pso.static_threadgroup_memory

    @property
    def attributes(self):
        return {"max_threads_per_block": self.max_threads_per_block, "num_regs": 0, "local_size_bytes": 0,
                "shared_size_bytes": self.shared_size_bytes, "const_size_bytes": 0,
                "max_dynamic_shared_size_bytes": self.max_dynamic_shared_size_bytes, "ptx_version": 0,
                "binary_version": 0, "cache_mode_ca": 0, "preferred_shared_memory_carveout": -1}

    @property
    def ptr(self):
        return id(self._k)

    def __getattr__(self, name):
        # attribute assignment knobs (max_dynamic_shared_size_bytes = ...) are accepted as no-ops below
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("_k", "kernel"):
            object.__setattr__(self, name, value)
        elif name in ("max_dynamic_shared_size_bytes", "preferred_shared_memory_carveout"):
            pass  # threadgroup memory is set per launch on Metal
        else:
            raise AttributeError(f"cannot set {name} on a Metal kernel")


class RawModule:
    def __init__(self, code=None, *, path=None, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, name_expressions=None, jitify=False):
        if path is not None:
            raise NotImplementedError("RawModule(path=cubin) cannot run on Metal; pass the CUDA source")
        if jitify or enable_cooperative_groups:
            raise NotImplementedError("jitify / cooperative groups on the Metal backend")
        self.code = code
        self.options = tuple(options)
        from .cuda import effective_options
        self.module = runtime.Module(code, effective_options(code, self.options), cupy_raw_module=True)
        self.name_expressions = name_expressions

    def get_function(self, name):
        return Function(self.module.get_function(name))

    def get_global(self, name):
        mem, g = self.module.get_global(name)
        from ._core import MemoryPointer
        return MemoryPointer(mem, mem.gpu + g["offset"])

    def compile(self, *a, **k):
        return None


class RawKernel(Function):
    def __init__(self, code, name, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, jitify=False):
        m = RawModule(code=code, options=options)
        object.__setattr__(self, "_module", m)
        super().__init__(m.module.get_function(name))
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "name", name)

    def compile(self, log_stream=None):
        return None          # compiled (translated and built) at construction

    def __setattr__(self, name, value):
        if name in ("_k", "kernel", "_module", "code", "name"):
            object.__setattr__(self, name, value)
        else:
            Function.__setattr__(self, name, value)


# ------------------------------------------------------------------ ElementwiseKernel
_DT = {"float32": np.float32, "float64": np.float64, "float": np.float32, "double": np.float64,
       "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16, "int32": np.int32,
       "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64, "bool": np.bool_, "int": np.int32}


_C_TYPENAME = {"bool": "bool", "int8": "signed char", "uint8": "unsigned char", "int16": "short",
               "uint16": "unsigned short", "int32": "int", "uint32": "unsigned int", "int64": "long long",
               "uint64": "unsigned long long", "float16": "float16", "float32": "float", "float64": "double"}


class _Param(tuple):
    """(ctype, name, raw), unpacked so internally, with CuPy's ParameterInfo
    attributes (name, ctype, raw, is_const) for code that reads the contract."""
    is_const = False
    # CuPy reports the C type name of a concrete dtype (int32 -> int); placeholders stay
    ctype = property(lambda self: _C_TYPENAME.get(self[0], self[0]))
    name = property(lambda self: self[1])
    raw = property(lambda self: self[2])


def _params(s, is_const):
    out = []
    for p in _parse_params(s):
        q = _Param(p)
        q.is_const = is_const
        out.append(q)
    return out


def _parse_params(s):
    out = []
    for p in [x.strip() for x in s.split(",") if x.strip()]:
        toks = p.split()
        raw = toks[0] == "raw"
        if raw:
            toks = toks[1:]
        if len(toks) != 2:
            raise ValueError(f"cannot parse kernel parameter {p!r}")
        out.append((toks[0], toks[1], raw))
    return out


class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name="kernel", reduce_dims=True, preamble="",
                 no_return=False, return_tuple=False, loop_prep="", after_loop="", options=(), **kw):
        self.in_params = _params(in_params, True)
        self.out_params = _params(out_params, False)
        self.operation = operation
        self.name = name
        self.preamble = preamble
        self.loop_prep = loop_prep
        self.after_loop = after_loop
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout

    def _types(self, args):
        tvars = {}
        for (t, n, raw), a in zip(self.in_params + self.out_params, args):
            if a is None:
                continue
            if t not in _DT:
                dt = a.dtype if is_device(a) or isinstance(a, np.generic) else np.asarray(a).dtype
                if t in tvars and tvars[t] != dt and not (not is_device(a)):
                    raise TypeError(f"{self.name}: type placeholder {t} bound to {tvars[t]} and {dt}")
                tvars.setdefault(t, np.dtype(dt) if is_device(a) else tvars.get(t))
        for (t, n, raw), a in zip(self.in_params + self.out_params, args):
            if t not in _DT and tvars.get(t) is None and a is not None:
                tvars[t] = np.dtype(a.dtype if isinstance(a, np.generic) else np.asarray(a).dtype)
        def res(t):
            if t in _DT:
                return np.dtype(_DT[t])
            if tvars.get(t) is None:
                raise TypeError(f"{self.name}: cannot determine type {t}")
            return tvars[t]
        return res

    def __call__(self, *args, size=None, block_size=None, stream=None, **kw):
        args = list(args)
        if len(args) == self.nin:
            args += [None] * self.nout
        if len(args) != self.nargs:
            raise TypeError(f"{self.name}: takes {self.nin} or {self.nargs} arguments ({len(args)} given)")
        res = self._types(args)
        # broadcast shape over non-raw array operands
        shapes = [a.shape for (t, n, raw), a in zip(self.in_params + self.out_params, args)
                  if not raw and a is not None and is_device(a)]
        if shapes:
            shape = np.broadcast_shapes(*shapes)
        elif size is not None:
            shape = (int(size),)
        else:
            raise ValueError(f"{self.name}: size= is required when every array is raw")
        nelem = int(np.prod(shape, dtype=np.int64))
        outs = []
        for j, (t, n, raw) in enumerate(self.out_params):
            a = args[self.nin + j]
            if a is None:
                if raw:
                    raise ValueError(f"{self.name}: raw output {n} must be given")
                a = _empty(shape, res(t))
                args[self.nin + j] = a
            outs.append(a)
        params, loads, scalars, arrays, lay_ops = [], [], [], [], []
        for (t, n, raw), a in zip(self.in_params + self.out_params, args):
            ct = K.ctype(res(t))
            is_out = any(a is o for o in outs)
            if raw:
                if not is_device(a):
                    raise TypeError(f"{self.name}: raw argument {n} must be a device array")
                if not a.flags.c_contiguous:
                    raise NotImplementedError(f"{self.name}: non-contiguous raw argument {n}")
                params.append(f"{'' if is_out else 'const '}{ct}* {n}")
                arrays.append(a)
            elif is_device(a):
                k = len(lay_ops)
                lay_ops.append(broadcast(a, shape))
                params.append(f"{'' if is_out else 'const '}char* __p_{n}")
                arrays.append(None)
                if is_out:
                    loads.append(f"{ct}& {n} = *({ct}*)(__p_{n} + __off[{k}]);")
                else:
                    loads.append(f"const {ct} {n} = *(const {ct}*)(__p_{n} + __off[{k}]);")
            else:
                params.append(f"{ct} {n}")
                arrays.append(("scalar", U._cast(np.asarray(a).item(), res(t))))
        op = self.operation.replace("_ind.size()", "__n")
        typedefs = "".join(f"typedef {K.ctype(res(t))} {t};\n" for t in dict.fromkeys(
            t for t, _, _ in self.in_params + self.out_params if t not in _DT))
        nops = max(len(lay_ops), 1)
        src = f"""{K._layout_struct(max(len(lay_ops), 1))}
{K._OFFSETS}
{typedefs}{self.preamble}
extern "C" __global__ void ek({", ".join(params + ["long long __n", "Lay __L"])}) {{
    {self.loop_prep}
    long long __off[{nops}];
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < __n; i += (long long)blockDim.x * gridDim.x) {{
        lay_offsets(__L, i, __off, {len(lay_ops)});
        {" ".join(loads)}
        {op};
    }}
    {self.after_loop}
}}
"""
        kern = K.kernel(src, "ek")
        strides = [U._strides_for(a, shape) for a in lay_ops]
        cshape, cstrides = K.collapse(shape, strides) if lay_ops else ((nelem,), [])
        lay = K.pack_layout(nelem, cshape, cstrides, max(len(lay_ops), 1))
        kargs = []
        li = 0
        for (t, n, raw), a, spec in zip(self.in_params + self.out_params, args, arrays):
            if isinstance(spec, tuple):
                kargs.append(spec[1])
            elif raw:
                kargs.append(a)
            else:
                kargs.append(lay_ops[li])
                li += 1
        kargs += [nelem, lay]
        if nelem:
            K.launch_1d(kern, nelem, kargs, block=block_size or 256)
        if self.no_return:
            return None
        if self.nout == 1 and not self.return_tuple:
            return outs[0]
        return tuple(outs)


# ------------------------------------------------------------------ ReductionKernel
class ReductionKernel:
    def __init__(self, in_params, out_params, map_expr, reduce_expr, post_map_expr, identity, name="reduce_kernel",
                 reduce_type=None, reduce_dims=True, preamble="", options=()):
        self.in_params = _params(in_params, True)
        self.out_params = _params(out_params, False)
        if len(self.out_params) != 1 or len(self.in_params) > 3:
            raise NotImplementedError("ReductionKernel with several outputs or more than 3 inputs on Metal")
        self.map_expr, self.reduce_expr, self.post_map_expr = map_expr, reduce_expr, post_map_expr
        self.identity, self.name, self.reduce_type, self.preamble = identity, name, reduce_type, preamble

    def __call__(self, *args, out=None, axis=None, keepdims=False, stream=None):
        ek = ElementwiseKernel("", "", "", self.name)
        ek.in_params, ek.out_params = self.in_params, self.out_params
        arrs = [a if is_device(a) else asarray(a) for a in args[:len(self.in_params)]]
        res_t = ek._types(arrs + [out])
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
        nd = len(shape)
        axes = U._norm_axes(axis, nd)
        oshape = tuple(s for k, s in enumerate(shape) if k not in axes)
        kshape = tuple(1 if k in axes else s for k, s in enumerate(shape))
        rshape = tuple(shape[k] for k in axes)
        on = int(np.prod(oshape, dtype=np.int64))
        rn = int(np.prod(rshape, dtype=np.int64))
        ot, on_name, _ = self.out_params[0]
        out_dt = res_t(ot)
        red_t = K.ctype(np.dtype(_DT[self.reduce_type])) if self.reduce_type in _DT else (
            self.reduce_type if self.reduce_type else K.ctype(out_dt))
        res = out if out is not None else _empty(kshape if keepdims else oshape, out_dt)
        rv = res.reshape(oshape) if res.shape != oshape else res
        b = [broadcast(a, shape) for a in arrs]
        loads = " ".join(f"const {K.ctype(a.dtype)} {n} = *(const {K.ctype(a.dtype)}*)(p{j} + o{j} + r[{j}]);"
                         for j, ((t, n, raw), a) in enumerate(zip(self.in_params, b)))
        pre_t = "".join(f"typedef {K.ctype(res_t(t))} {t};\n" for t in {t for t, _, _ in self.in_params + self.out_params}
                        if t not in _DT)
        np_ = len(b)
        src = f"""{K._layout_struct(np_ + 1)}
{K._OFFSETS}
{pre_t}{self.preamble}
__device__ __forceinline__ {red_t} __red({red_t} a, {red_t} b) {{ return ({self.reduce_expr}); }}
extern "C" __global__ void __launch_bounds__(256) rk({", ".join(f"const char* p{j}" for j in range(np_))}, char* out, Lay LO, Lay LR) {{
    __shared__ {red_t} s[256];
    long long o = blockIdx.x;
    long long oo[{np_ + 1}];
    lay_offsets(LO, o, oo, {np_ + 1});
    {" ".join(f"long long o{j} = oo[{j + 1}];" for j in range(np_))}
    {red_t} a = ({red_t})({self.identity});
    for (long long j = threadIdx.x; j < LR.n; j += blockDim.x) {{
        long long r[{np_}];
        lay_offsets(LR, j, r, {np_});
        {loads}
        a = __red(a, ({red_t})({self.map_expr}));
    }}
    s[threadIdx.x] = a;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {{
        if (threadIdx.x < k) s[threadIdx.x] = __red(s[threadIdx.x], s[threadIdx.x + k]);
        __syncthreads();
    }}
    if (threadIdx.x == 0) {{
        a = s[0];
        {K.ctype(out_dt)}& {on_name} = *({K.ctype(out_dt)}*)(out + oo[0]);
        {self.post_map_expr};
    }}
}}
"""
        kern = K.kernel(src, "rk")
        ost = [list(rv.strides)] + [[a.strides[k] for k in range(nd) if k not in axes] for a in b]
        oshp, ost = K.collapse(oshape, ost) if oshape else ((), [()] * (np_ + 1))
        rst = [[a.strides[k] for k in axes] for a in b]
        rshp, rst = K.collapse(rshape, rst)
        if on:
            kern((on,), (256,), tuple(b) + (rv, K.pack_layout(on, oshp, ost, np_ + 1), K.pack_layout(rn, rshp, rst, np_ + 1)))
        return res
