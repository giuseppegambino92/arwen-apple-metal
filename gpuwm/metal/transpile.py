"""CUDA -> Metal Shading Language translator for ArWen kernel modules.

The CUDA text gpuwm hands to NVRTC is canonical.  This translator is a pure
function of (source, NVRTC options, FP64 mode) -> (MSL text, kernel ABI
metadata); nothing is maintained by hand on the Metal side except the
compatibility headers in ``include/``.

Pipeline: preprocess (frontend.preprocess) -> parse with libclang in CUDA
device mode (frontend.parse) -> collect edits from the AST -> apply them.

Semantics carried across, and how (see docs/metal-backend.md):

* kernels      ``__global__`` functions become ``kernel`` functions whose CUDA
               parameters arrive in one ``constant`` argument struct; pointers
               are 64-bit GPU addresses (bindless), so an array view is just
               an address + byte offset, as in CUDA.
* indices      ``threadIdx``/``blockIdx``/``blockDim``/``gridDim`` are read
               from a per-thread context built in the kernel prologue; device
               functions that need it (transitively) receive it as their first
               parameter.
* pointers     CUDA pointers are generic, MSL pointers carry an address space.
               Device functions become templates over their pointer
               parameters (deduction monomorphises per address space);
               non-const reference parameters become pointers; local pointer
               variables are declared ``auto``.
* double       Every ``double`` becomes ``__f64`` (compute type) or
               ``__f64m`` (memory type, always IEEE binary64 bits so host and
               device views agree).  Every implicit conversion involving
               double is made explicit from the AST, and every double literal
               is emitted as its exact binary64 bit pattern.  The arithmetic
               behind __f64 is chosen per module: sf64 (IEEE soft-float, bit
               exact), df64 (float-float), fp32 (demoted).
* globals      ``__constant__``/``__device__`` variables that are constant
               initialised become MSL ``constant`` data; mutable ones live in
               a per-module globals buffer addressed through the context.
* inline PTX   only the instruction patterns found in ArWen are recognised;
               anything else is a TranslationError.
"""
from __future__ import annotations

import dataclasses
import os
import hashlib
import json
import re
import struct
from collections import defaultdict

import clang.cindex as ci

from .frontend import NvrtcOptions, TranslationError, parse, preprocess

K = ci.CursorKind
T = ci.TypeKind
TRANSLATOR_REVISION = "arwen-metal-transpile-1"
F64_MODES = ("sf64", "df64", "fp32")
BUILTIN_VARS = {"threadIdx": "tid", "blockIdx": "bid", "blockDim": "bdim", "gridDim": "gdim"}
WARP_FUNCS = {"__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync",
              "__ballot_sync", "__any_sync", "__all_sync", "__activemask", "__trap",
              "__syncthreads_or", "__syncthreads_and", "__syncthreads_count"}
SYNC_REDUCTIONS = {"__syncthreads_or", "__syncthreads_and", "__syncthreads_count"}
FN_KINDS = (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD, K.CONSTRUCTOR)


# ----------------------------------------------------------------- rewriting
class Rewriter:
    """Non-overlapping replacements plus ordered insertions on one text."""

    def __init__(self, text: str):
        self.text = text
        self.repl: dict[int, tuple[int, str]] = {}
        self.ins: list[tuple[int, float, int, str]] = []
        self.opaque: list[tuple[int, int]] = []
        self._n = 0

    def in_opaque(self, start: int, end: int) -> bool:
        return any(a <= start and end <= b for a, b in self.opaque)

    def replace(self, start: int, end: int, new: str, *, opaque: bool = False):
        """Replace [start, end).  An ``opaque`` replacement rewrites a whole
        expression; later edits strictly inside it are moot and dropped."""
        if self.in_opaque(start, end) and (start, end) not in self.opaque:
            return
        old = self.repl.get(start)
        if old is not None and old != (end, new) and (start, end) in self.opaque and old[0] == end:
            return          # an opaque rewrite of exactly this expression came first (the outer edit)
        if old is not None and old != (end, new):
            raise TranslationError(f"conflicting rewrites at offset {start}: {self.text[start:end]!r}")
        self.repl[start] = (end, new)
        if opaque:
            self.opaque.append((start, end))
            # edits already registered strictly inside are superseded by this one
            for s0 in [s0 for s0, (e0, _) in self.repl.items()
                       if start <= s0 and e0 <= end and (s0, e0) != (start, end)]:
                del self.repl[s0]
            self.ins = [x for x in self.ins if not (start < x[0] < end)]

    def insert(self, pos: int, new: str, order: float = 0.0):
        if any(a < pos < b for a, b in self.opaque):
            return
        self._n += 1
        self.ins.append((pos, order, self._n, new))

    def apply(self) -> str:
        spans = sorted((s, e) for s, (e, _) in self.repl.items())
        for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
            if s1 < e0:
                raise TranslationError(f"overlapping rewrites {self.text[s0:e0]!r} / {self.text[s1:e1]!r}")
        ins = defaultdict(list)
        for pos, order, n, new in self.ins:
            for s, e in spans:
                if s < pos < e:
                    raise TranslationError(f"insertion inside a replaced span: {self.text[s:e]!r}")
            ins[pos].append((order, n, new))
        out, i = [], 0
        points = sorted(set([s for s, _ in spans] + list(ins)))
        for p in points:
            out.append(self.text[i:p])
            i = p
            for _, _, new in sorted(ins.get(p, ())):
                out.append(new)
            if p in self.repl:
                e, new = self.repl[p]
                out.append(new)
                i = e
        out.append(self.text[i:])
        return "".join(out)


# ----------------------------------------------------------------- helpers
def _canon(t):
    return t.get_canonical()


def _is_double(t) -> bool:
    return _canon(t).kind in (T.DOUBLE, T.LONGDOUBLE)


def _arith_kind(t):
    c = _canon(t)
    if c.kind in (T.DOUBLE, T.LONGDOUBLE):
        return "f64"
    if c.kind == T.FLOAT:
        return "f32"
    if c.kind in (T.BOOL,):
        return "bool"
    if c.kind in (T.INT, T.UINT, T.SHORT, T.USHORT, T.CHAR_S, T.CHAR_U, T.SCHAR, T.UCHAR,
                  T.LONG, T.ULONG, T.LONGLONG, T.ULONGLONG, T.ENUM, T.CHAR16, T.CHAR32, T.WCHAR):
        return "int"
    return None


_SCALAR_MSL = {
    T.BOOL: "bool", T.CHAR_S: "char", T.SCHAR: "char", T.CHAR_U: "uchar", T.UCHAR: "uchar",
    T.SHORT: "short", T.USHORT: "ushort", T.INT: "int", T.UINT: "uint",
    T.LONG: "long", T.ULONG: "ulong", T.LONGLONG: "long", T.ULONGLONG: "ulong",
    T.FLOAT: "float", T.DOUBLE: "__f64", T.LONGDOUBLE: "__f64",
}
_SCALAR_ABI = {  # (size, align, numpy-like code) of the CUDA/host representation
    "bool": (1, 1, "?"), "char": (1, 1, "b"), "uchar": (1, 1, "B"), "short": (2, 2, "h"),
    "ushort": (2, 2, "H"), "int": (4, 4, "i"), "uint": (4, 4, "I"), "long": (8, 8, "q"),
    "ulong": (8, 8, "Q"), "float": (4, 4, "f"), "__f64m": (8, 8, "d"),
}


class _Off:
    __slots__ = ("offset",)

    def __init__(self, o):
        self.offset = o


class _Ext:
    __slots__ = ("start", "end")

    def __init__(self, s, e):
        self.start, self.end = _Off(s), _Off(e)


class _Tok:
    """A token of the main file, addressed by byte offsets.

    libclang's own tokenizer returns nothing for a cursor whose extent starts
    inside a macro expansion (``__device__`` comes from the parse header), so
    the main file is tokenized once and sliced by offsets."""
    __slots__ = ("spelling", "kind", "extent")

    def __init__(self, spelling, kind, s, e):
        self.spelling, self.kind, self.extent = spelling, kind, _Ext(s, e)


_FILE_TOKENS: dict = {}


def _file_tokens(tu, filename):
    key = id(tu)
    hit = _FILE_TOKENS.get(key)
    if hit is None or hit[0] is not tu:
        f = tu.get_file(filename)
        size = len(tu._arwen_text.encode()) if hasattr(tu, "_arwen_text") else 1 << 30
        rng = ci.SourceRange.from_locations(ci.SourceLocation.from_offset(tu, f, 0),
                                            ci.SourceLocation.from_offset(tu, f, size))
        toks = [_Tok(t.spelling, t.kind, t.extent.start.offset, t.extent.end.offset) for t in tu.get_tokens(extent=rng)]
        hit = (tu, toks, [t.extent.start.offset for t in toks])
        _FILE_TOKENS.clear()
        _FILE_TOKENS[key] = hit
    return hit


def _tokens(cursor):
    import bisect
    tu = cursor._tu
    _, toks, starts = _file_tokens(tu, tu._arwen_filename)
    s, e = cursor.extent.start.offset, cursor.extent.end.offset
    i = bisect.bisect_left(starts, s)
    out = []
    while i < len(toks) and toks[i].extent.start.offset < e:
        out.append(toks[i])
        i += 1
    return out


def _hexf32(v: float) -> str:
    f = struct.unpack("<f", struct.pack("<f", v))[0]
    if f != f:
        return "__arwen_qnanf()"
    if f in (float("inf"), float("-inf")):
        return "INFINITY" if f > 0 else "(-INFINITY)"
    return (f.hex() + "f") if f != 0 or str(f)[0] != "-" else "-0.0f"


def _f64_lit(v: float, storage: bool) -> str:
    bits = struct.unpack("<Q", struct.pack("<d", v))[0]
    hi = struct.unpack("<f", struct.pack("<f", v))[0] if abs(v) < 3.4e38 or v != v else v
    try:
        hi32 = struct.unpack("<f", struct.pack("<f", v))[0]
        lo32 = struct.unpack("<f", struct.pack("<f", v - hi32))[0] if hi32 == hi32 and abs(hi32) != float("inf") else 0.0
    except OverflowError:
        hi32, lo32 = (float("inf") if v > 0 else float("-inf")), 0.0
    macro = "__F64M_LIT" if storage else "__F64_LIT"
    return f"{macro}(0x{bits:016x}UL, {_hexf32(hi32)}, {_hexf32(lo32)})"


_MACRO_VALUES = {
    "DBL_MAX": 1.7976931348623157e308, "DBL_MIN": 2.2250738585072014e-308, "DBL_EPSILON": 2.220446049250313e-16,
    "FLT_MAX": float.fromhex("0x1.fffffep+127"), "FLT_MIN": float.fromhex("0x1p-126"),
    "FLT_EPSILON": float.fromhex("0x1p-23"),
}


def _np32(v):
    import numpy as np
    return np.float32(v)


def _decimal_to_f32(text: str) -> float:
    """Correctly rounded decimal -> binary32 (no double rounding through binary64)."""
    from fractions import Fraction
    import struct
    q = Fraction(text)
    if q == 0:
        return 0.0
    neg = q < 0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)
    m = q * Fraction(2) ** (23 - e)
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return float("-inf") if neg else float("inf")
    bits = n if n < (1 << 23) else ((e + 127) << 23) | (n - (1 << 23))
    v = struct.unpack("<f", struct.pack("<I", bits))[0]
    return -v if neg else v


def _literal_value(cursor):
    """Numeric value of a (possibly negated / parenthesised) literal, else None."""
    k = cursor.kind
    if k == K.PAREN_EXPR or (k == K.UNEXPOSED_EXPR and len(list(cursor.get_children())) == 1):
        ch = list(cursor.get_children())
        return _literal_value(ch[0]) if ch else None
    if k == K.UNARY_OPERATOR:
        toks = _tokens(cursor)
        ch = list(cursor.get_children())
        if toks and toks[0].spelling in ("-", "+") and len(ch) == 1:
            v = _literal_value(ch[0])
            if v is not None:
                return -v if toks[0].spelling == "-" else v
        return None
    if k == K.FLOATING_LITERAL:
        toks = _tokens(cursor)
        if not toks:
            return None
        if toks[0].spelling in _MACRO_VALUES:
            return _MACRO_VALUES[toks[0].spelling]
        s = toks[0].spelling.rstrip("fFlL")
        if not re.fullmatch(r"[-+]?(0[xX][0-9a-fA-F.]+[pP][-+]?\d+|[0-9.]+([eE][-+]?\d+)?)", s):
            return None
        v = float.fromhex(s) if s.lower().startswith(("0x", "-0x")) else float(s)
        if _canon(cursor.type).kind == T.FLOAT:
            # a binary32 literal (``1.373f``) is its float32 rounding, also when it is
            # later widened to binary64; decimal -> binary32 directly (C semantics)
            v = _decimal_to_f32(s) if not s.lower().startswith(("0x", "-0x")) else float(_np32(v))
        return v
    if k == K.INTEGER_LITERAL:
        toks = _tokens(cursor)
        if not toks:
            return None
        s = toks[0].spelling.rstrip("uUlL")
        if not re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", s):
            return None
        return float(int(s, 0) if not re.fullmatch(r"0[0-7]+", s) else int(s, 8))
    return None


# Identifiers that are reserved words or type names in MSL but ordinary names in CUDA C++.
# CUDA vector types whose MSL namesake has a different layout, or no MSL namesake.
CUDA_VECTOR_RENAMES = {f"{b}{n}": f"__arwen_{b}{n}" for b in ("double", "longlong", "ulonglong") for n in (1, 2, 3, 4)}
CUDA_VECTOR_RENAMES.update({f"make_{b}{n}": f"__arwen_make_{b}{n}" for b in ("double", "longlong", "ulonglong") for n in (1, 2, 3, 4)})
CUDA_VECTOR_RENAMES.update({"float3": "packed_float3", "int3": "packed_int3", "uint3": "packed_uint3", "dim3": "__arwen_dim3"})
MSL_RESERVED = {"half", "kernel", "vertex", "fragment", "device", "constant", "thread", "threadgroup",
                "threadgroup_imageblock", "sampler", "texture", "vec", "ray_data", "object_data", "visible",
                "stage_in", "patch", "mesh", "object", "uniform", "packed", "coherent", "as_type", "bfloat"}


def _ident(name: str) -> str:
    return f"__arwen_id_{name}" if name in MSL_RESERVED else name


def _array_dims(t):
    """(innermost element type, [dims]) of a (multi-dimensional) constant array type."""
    dims = []
    while _canon(t).kind == T.CONSTANTARRAY:
        dims.append(_canon(t).element_count)
        t = t.get_array_element_type()
    return t, dims


def _deep_const(t) -> bool:
    while True:
        if t.is_const_qualified():
            return True
        if _canon(t).kind not in (T.CONSTANTARRAY, T.INCOMPLETEARRAY):
            return False
        t = t.get_array_element_type()


@dataclasses.dataclass
class KernelParam:
    name: str
    kind: str            # "ptr" | "scalar" | "struct"
    msl_type: str
    size: int
    align: int
    offset: int
    code: str = ""       # struct-format code for scalars ("f", "i", "d", ...)


@dataclasses.dataclass
class KernelInfo:
    name: str
    params: list
    args_size: int
    max_threads: int | None
    uses_barrier: bool
    uses_shared: bool
    uses_warp: bool
    dyn_shared: bool


@dataclasses.dataclass
class GlobalVar:
    name: str
    msl_type: str
    size: int
    align: int
    offset: int
    init: str | None


@dataclasses.dataclass
class Translation:
    msl: str
    kernels: dict
    globals: dict
    globals_size: int
    f64_mode: str
    options: NvrtcOptions
    metal_flags: tuple

    def metadata(self) -> dict:
        return dict(revision=TRANSLATOR_REVISION, f64_mode=self.f64_mode, metal_flags=list(self.metal_flags),
                    globals_size=self.globals_size,
                    globals={k: dataclasses.asdict(v) for k, v in self.globals.items()},
                    kernels={k: dict(dataclasses.asdict(v), params=[dataclasses.asdict(p) for p in v.params])
                             for k, v in self.kernels.items()})


# ----------------------------------------------------------------- translator
class _Module:
    def __init__(self, text: str, opts: NvrtcOptions, f64_mode: str, filename: str):
        self.text = text
        self.opts = opts
        self.mode = f64_mode
        self.filename = filename
        tu, errors = parse(text, filename)
        if errors:
            msg = "\n".join(f"{e.location.line}:{e.location.column}: {e.spelling}" for e in errors[:20])
            raise TranslationError(f"{filename}: CUDA parse errors:\n{msg}")
        tu._arwen_text = text
        tu._arwen_filename = filename
        self.tu = tu
        self.rw = Rewriter(text)
        self.parent: dict = {}
        self.nodes: list = []
        self.fn_of: dict = {}            # node hash -> enclosing function cursor
        self.functions: dict = {}        # usr -> [cursor decls]
        self.kernels: list = []
        self.globals_mut: dict = {}      # usr -> GlobalVar
        self.global_cursors: dict = {}
        self.needs_ctx: set = set()
        self.ref_params: dict = {}       # usr -> {param index: "ptr"|"value"}
        self.kernel_infos: dict = {}
        self.dyn_shared_used = False
        self.module_uses_globals = False
        self.ssmem_size = 0
        self.syncred_off = None
        self._tmpl_defaults_done: set = set()
        self._pointee_params_done: set = set()
        self.block_protos: dict = {}       # DECL_STMT hash -> block-scope function prototypes
        self.destructors: list = []
        self.ssmem_globals: dict = {}      # program-scope __shared__ VAR_DECL hash -> ssmem offset
        self.dep_shared: dict = {}         # fn usr -> (fn, [__shared__ VAR_DECLs of dependent size])
        self.dep_shared_decl: dict = {}    # VAR_DECL hash -> (element MSL type, dims text, offset expression)
        self.ssmem_vars: dict = {}
        self.ssmem_users: set = set()
        self.field_space: dict = {}

    # -- traversal ------------------------------------------------------
    def _main(self, c) -> bool:
        f = c.location.file
        return f is not None and f.name == self.filename

    def collect(self):
        stack = [(self.tu.cursor, None, None)]
        while stack:
            c, parent, fn = stack.pop()
            if parent is not None and not self._main(c):
                if parent.kind != K.TRANSLATION_UNIT and c.location.file is None:
                    # an implicit node located nowhere (the member expression of a
                    # parse-only base conversion): transparent, its operands are ours
                    for ch in reversed(list(c.get_children())):
                        stack.append((ch, parent, fn))
                continue
            if parent is not None:
                self.parent[c.hash] = parent
                self.nodes.append(c)
                self.fn_of[c.hash] = fn
            nfn = fn
            if c.kind in FN_KINDS or c.kind == K.LAMBDA_EXPR:
                if c.kind != K.LAMBDA_EXPR and parent is not None and parent.kind == K.DECL_STMT:
                    # a block-scope prototype (f2c ``extern int f_(...), g_(...);``): removed
                    self.block_protos.setdefault(parent.hash, parent)
                elif c.kind != K.LAMBDA_EXPR:
                    nfn = c
                    usr = c.get_usr()
                    self.functions.setdefault(usr, []).append(c)
            elif c.kind == K.DESTRUCTOR:
                nfn = c                      # its body's ctx needs, see analyse / edit_destructors
                self.destructors.append(c)
            for ch in reversed(list(c.get_children())):
                stack.append((ch, c, nfn))
        self.usr_by_loc = {}
        names = defaultdict(set)
        for usr, decls in self.functions.items():
            for d in decls:
                names[d.spelling].add(usr)
        self.fn_names = names
        self.fn_by_name = {n: next(iter(u)) for n, u in names.items() if len(u) == 1}
        for usr, decls in self.functions.items():
            for d in decls:
                self.usr_by_loc.setdefault(d.location.offset, usr)
        self.nodes.sort(key=lambda c: (c.extent.start.offset, -c.extent.end.offset))

    @staticmethod
    def call_args(call):
        args = list(call.get_arguments())
        if args or call.referenced is not None:
            return args
        # unresolved call inside a template: callee expression, then the arguments
        return [k for k in call.get_children() if k.kind.is_expression()][1:]

    def callee(self, call):
        """(usr, declaration) of a call to a main-file function, also for calls
        clang leaves unresolved inside templates (resolved by unique name)."""
        ref = call.referenced
        if ref is not None:
            if not self._main(ref):
                return None, None
            return self.fusr(ref), ref
        name = call.spelling
        if not name:
            kids = list(call.get_children())
            if kids:
                toks = [t.spelling for t in _tokens(kids[0])]
                if "<" in toks:
                    toks = toks[:toks.index("<")]
                name = kids[0].spelling or (toks[-1] if toks else "")
        usr = self.fn_by_name.get(name)
        if usr is None and len(self.fn_names.get(name, ())) > 1:
            # same-named functions (e.g. two scheme variants' methods): usable when their
            # parameters have the same shape, since only that shape drives the rewrite
            cands = sorted(self.fn_names[name])
            shapes = {tuple(_canon(p.type).kind for p in self._params(self.functions[u][0])) for u in cands}
            ctx = {u in self.needs_ctx for u in cands} if hasattr(self, "needs_ctx") else {False}
            if len(shapes) == 1 and len(ctx) == 1:
                usr = cands[0]
        if usr is None:
            return None, None
        decls = self.functions[usr]
        d = next((x for x in decls if self._body(x) is not None), decls[0])
        return usr, d

    def fusr(self, ref) -> str:
        """USR of a main-file function; specialisations map to their template."""
        if ref is not None and self._main(ref):
            u = self.usr_by_loc.get(ref.location.offset)
            if u is not None:
                return u
        return ref.get_usr()

    @staticmethod
    def _attrs(c):
        return {ch.kind for ch in c.get_children()}

    def is_kernel(self, c) -> bool:
        return c.kind == K.FUNCTION_DECL and K.CUDAGLOBAL_ATTR in self._attrs(c)

    def _body(self, fn):
        for ch in fn.get_children():
            if ch.kind == K.COMPOUND_STMT:
                return ch
        return None

    def _params(self, fn):
        return [ch for ch in fn.get_children() if ch.kind == K.PARM_DECL]

    def _program_scope(self, c) -> bool:
        p = self.parent.get(c.hash)
        while p is not None and p.kind in (K.LINKAGE_SPEC, K.NAMESPACE, K.UNEXPOSED_DECL):
            p = self.parent.get(p.hash)
        return p is None or p.kind == K.TRANSLATION_UNIT

    # -- MSL type spelling ---------------------------------------------
    def msl_type(self, t, *, space="device", storage=False) -> str:
        """Spell clang type ``t`` in MSL; pointers get address space ``space``."""
        c = _canon(t)
        const = "const " if t.is_const_qualified() else ""
        if c.kind == T.POINTER:
            pt = t.get_pointee()
            if _canon(pt).kind == T.CONSTANTARRAY:
                # pointer to array (a ``T a[][N]`` parameter): ``space T (*)[N]``
                elem, dims = _array_dims(pt)
                cst = " const" if t.is_const_qualified() else ""
                return f"{space} {self.msl_type(elem, storage=True)} (*{cst})" + "".join(f"[{d}]" for d in dims)
            inner = self.msl_type(pt, space=space, storage=True)
            cst = " const" if t.is_const_qualified() else ""
            if _canon(pt).kind == T.POINTER:
                # pointer to a pointer object stored in ``space``: T* const device*
                return f"{inner} {space}*{cst}"
            return (f"{space} {inner}*" if space else f"{inner}*") + cst
        if c.kind in (T.DOUBLE, T.LONGDOUBLE):
            return const + ("__f64m" if storage else "__f64")
        if c.kind in _SCALAR_MSL:
            # keep typedef names the module defines (e.g. ``real``) unless they hide a double
            if t.kind == T.TYPEDEF and not _is_double(t) and t.spelling not in ("size_t", "int64_t", "uint64_t"):
                return const + t.spelling.replace("const ", "")
            return const + _SCALAR_MSL[c.kind]
        if c.kind == T.CONSTANTARRAY:
            raise TranslationError(f"array type {t.spelling} where a scalar/pointer/record type was expected")
        if c.kind in (T.RECORD, T.ELABORATED, T.ENUM):
            name = t.spelling.replace("const ", "").replace("struct ", "")
            name = CUDA_VECTOR_RENAMES.get(name, name)      # double2 -> __arwen_double2 ...
            return const + name
        return t.spelling

    def scalar_abi(self, t):
        c = _canon(t)
        if c.kind in (T.DOUBLE, T.LONGDOUBLE):
            return _SCALAR_ABI["__f64m"]
        name = _SCALAR_MSL.get(c.kind)
        if name is None:
            return None
        return _SCALAR_ABI[name]

    # -- analysis -------------------------------------------------------
    def _ssmem_global_slot(self, v):
        if v.hash in self.ssmem_globals:
            return self.ssmem_globals[v.hash]
        ct = _canon(v.type)
        size, align = ct.get_size(), max(16, ct.get_align())
        if size <= 0:
            raise TranslationError(f"__shared__ variable {v.spelling}: unsized")
        self.ssmem_size = (self.ssmem_size + align - 1) // align * align
        self.ssmem_globals[v.hash] = self.ssmem_size
        self.ssmem_size += size
        return self.ssmem_globals[v.hash]

    def edit_ctx_records(self):
        for rec in getattr(self, "ctx_records", {}).values():
            ropen = self.text.find("{", rec.location.offset)
            ctors = [m for m in rec.get_children() if m.kind == K.CONSTRUCTOR
                     and not re.search(r"=\s*delete\s*$", self.text[m.extent.start.offset:m.extent.end.offset])]
            # the ctx reference is the first member: initialised before the others
            self.rw.insert(ropen + 1, " thread const __arwen_ctx_t& __arwen_ctx; "
                           + ("" if ctors else f"__device__ explicit {rec.spelling}(thread const __arwen_ctx_t& __c) "
                                                ": __arwen_ctx(__c) {} "))
            for m in ctors:
                # the constructor got the ctx parameter (named like the member): store it
                body = next((ch for ch in m.get_children() if ch.kind == K.COMPOUND_STMT), None)
                if body is None:
                    raise TranslationError(f"{rec.spelling}: constructor without a body")
                rp = _ctor_body(self.text, m)[0]
                init = self.text[rp + 1:body.extent.start.offset]
                if ":" in init:
                    k = rp + 1 + init.index(":")
                    self.rw.insert(k + 1, " __arwen_ctx(__arwen_ctx),")
                else:
                    self.rw.insert(body.extent.start.offset, " : __arwen_ctx(__arwen_ctx) ")
        if not getattr(self, "ctx_records", None):
            return
        # records with a user constructor: edit_call passes the ctx at ``Rec x;``
        names = {r.spelling for r in self.ctx_records.values()
                 if not any(m.kind == K.CONSTRUCTOR for m in r.get_children())}
        for c in self.nodes:
            if c.kind != K.VAR_DECL or self._program_scope(c):
                continue
            decl = _canon(c.type).get_declaration()
            if decl is None or decl.spelling not in names:
                continue
            toks = _tokens(c)
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None or any(t.spelling in ("=", "(", "{") for t in toks
                                       if t.extent.start.offset > name_tok.extent.start.offset):
                raise TranslationError(f"{c.spelling}: initialised {decl.spelling} needs the ctx")
            self.rw.insert(name_tok.extent.end.offset, "(__arwen_ctx)")

    _ELEM_BYTES = {"float": 4, "int": 4, "uint": 4, "bool": 1, "char": 1, "uchar": 1, "short": 2,
                   "ushort": 2, "long": 8, "ulong": 8, "__f64": 8, "__f64m": 8, "half": 2}

    def _layout_dependent_shared(self):
        """Dependent-size ``__shared__`` arrays of a function template: offsets are
        expressions of the template parameter (each slot 16-byte aligned), after a
        base reserved in the static threadgroup block for the LARGEST instantiation
        the module names (``f<32>(...)`` / ``f<64>(...)``)."""
        for usr, (fn, vars_) in self.dep_shared.items():
            nttp = [ch for ch in fn.get_children() if ch.kind == K.TEMPLATE_NON_TYPE_PARAMETER]
            if len(nttp) != 1:
                raise TranslationError(f"{fn.spelling}: dependent __shared__ sizes need one integer template parameter")
            pname = nttp[0].spelling
            vals = {int(v) for v in re.findall(r"\b%s\s*<\s*(\d+)\s*>\s*\(" % re.escape(fn.spelling), self.text)}
            if not vals or re.search(r"\b%s\s*<\s*[^\d\s>]" % re.escape(fn.spelling), self.text):
                raise TranslationError(f"{fn.spelling}: dependent __shared__ sizes need literal instantiations")
            entries = []
            for v in vars_:
                toks = [t.spelling for t in _tokens(v)]
                k = toks.index(v.spelling)
                j = k + 1
                dims = []
                while j < len(toks) and toks[j] == "[":
                    e = toks.index("]", j)
                    dims.append(" ".join(toks[j + 1:e]))
                    j = e + 1
                elem = v.type
                while elem.kind in (T.CONSTANTARRAY, T.DEPENDENTSIZEDARRAY, T.INCOMPLETEARRAY):
                    elem = elem.get_array_element_type()
                ety = self.msl_type(elem)
                nb = self._ELEM_BYTES.get(ety.replace("const ", ""))
                if nb is None or not dims:
                    raise TranslationError(f"__shared__ {v.spelling}: unsupported dependent element {ety}")
                entries.append((v, ety, nb, dims))
            def total(val):
                off = 0
                for _, _, nb, dims in entries:
                    n = nb
                    for d in dims:
                        n *= int(eval(d.replace(pname, str(val)), {"__builtins__": {}}))
                    off = (off + 15) // 16 * 16 + n
                return off
            base = (self.ssmem_size + 15) // 16 * 16
            self.ssmem_size = base + max(total(x) for x in vals)
            off_expr = "0"
            for v, ety, nb, dims in entries:
                off_expr = f"((({off_expr}) + 15) / 16 * 16)"
                self.dep_shared_decl[v.hash] = (ety, "".join(f"[{d}]" for d in dims), f"{base} + {off_expr}")
                off_expr = f"{off_expr} + {nb} * " + " * ".join(f"({d})" for d in dims)

    def edit_destructors(self):
        for d in getattr(self, "ctx_dtors", []):
            rec = self.parent.get(d.hash)
            body = next((ch for ch in d.get_children() if ch.kind == K.COMPOUND_STMT), None)
            if rec is None or body is None:
                raise TranslationError(f"{d.spelling}: destructor needing the ctx without a body")
            self.rw.insert(body.extent.start.offset + 1, " thread const __arwen_ctx_t& __arwen_ctx = *__arwen_ctxp; ")
            ropen = self.text.find("{", rec.location.offset)
            self.rw.insert(ropen + 1, " thread const __arwen_ctx_t* __arwen_ctxp; ")
            for m in rec.get_children():
                if m.kind == K.CONSTRUCTOR:
                    mb = next((ch for ch in m.get_children() if ch.kind == K.COMPOUND_STMT), None)
                    if mb is None and re.search(r"=\s*delete\s*$", self.text[m.extent.start.offset:m.extent.end.offset]):
                        continue
                    if mb is None:
                        raise TranslationError(f"{rec.spelling}: constructor without a body")
                    self.rw.insert(mb.extent.start.offset + 1, " __arwen_ctxp = &__arwen_ctx; ", order=-1e6)

    def analyse(self):
        builtin_users, warp_users, callers = set(), set(), defaultdict(set)
        for c in self.nodes:
            fn = self.fn_of.get(c.hash)
            fusr = fn.get_usr() if fn is not None else None
            if c.kind == K.DECL_REF_EXPR:
                ref = c.referenced
                if ref is not None and ref.kind == K.VAR_DECL and self._main(ref) and self._program_scope(ref) \
                        and K.CUDASHARED_ATTR in self._attrs(ref) and not self._is_extern_shared(ref):
                    self._ssmem_global_slot(ref)
                    self.ssmem_users.add(fusr)
                    builtin_users.add(fusr)
                if c.spelling in BUILTIN_VARS and ref is not None and not self._main(ref):
                    builtin_users.add(fusr)
                elif ref is not None and ref.kind == K.VAR_DECL and self._main(ref) and self._is_mutable_global(ref):
                    builtin_users.add(fusr)
            elif c.kind == K.CALL_EXPR:
                if c.spelling in WARP_FUNCS and c.referenced is not None and not self._main(c.referenced):
                    warp_users.add(fusr)
                    if c.spelling in SYNC_REDUCTIONS:
                        # one threadgroup word for the block-wide vote, in the static shared area
                        if self.syncred_off is None:
                            self.ssmem_size = (self.ssmem_size + 15) // 16 * 16
                            self.syncred_off = self.ssmem_size
                            self.ssmem_size += 16
                        self.ssmem_users.add(fusr)
                cu, _ = self.callee(c)
                if cu is not None and fusr is not None:
                    callers[cu].add(fusr)
            elif c.kind == K.ASM_STMT and re.search(r'"\s*trap\s*;\s*"', self.text[c.extent.start.offset:c.extent.end.offset]):
                warp_users.add(fusr)
            elif c.kind == K.VAR_DECL and self._is_extern_shared(c):
                self.dyn_shared_used = True
                builtin_users.add(fusr)
            elif c.kind == K.VAR_DECL and K.CUDASHARED_ATTR in self._attrs(c) and fn is not None \
                    and not self.is_kernel(fn) and not self._program_scope(c):
                ct = _canon(c.type)
                size, align = ct.get_size(), max(16, ct.get_align())
                if size <= 0 or ct.get_align() <= 0:
                    # ``__shared__ float a[KMAX]`` in ``template <int KMAX>``: laid out per
                    # instantiation (see _layout_dependent_shared)
                    self.dep_shared.setdefault(fusr, (fn, []))[1].append(c)
                    self.ssmem_users.add(fusr)
                    builtin_users.add(fusr)
                    continue
                self.ssmem_size = (self.ssmem_size + align - 1) // align * align
                self.ssmem_vars[c.hash] = self.ssmem_size
                self.ssmem_size += size
                self.ssmem_users.add(fusr)
                builtin_users.add(fusr)
        self._layout_dependent_shared()
        self.callers = callers
        need = set(u for u in builtin_users | warp_users if u)

        def propagate():
            work = list(need)
            while work:
                u = work.pop()
                for caller in callers.get(u, ()):
                    if caller not in need:
                        need.add(caller)
                        work.append(caller)
        propagate()
        # a destructor needing the ctx reads it from a pointer its constructors store
        self.ctx_dtors = [d for d in self.destructors if d.get_usr() in need]
        for d in self.ctx_dtors:
            rec = self.parent.get(d.hash)
            for m in (rec.get_children() if rec is not None else ()):
                if m.kind == K.CONSTRUCTOR:
                    need.add(m.get_usr())
        propagate()
        # records whose default member initialisers construct ctx-needing objects
        # (``LakeStorage<double,1> t{1,1};`` members): they hold the ctx by reference
        self.ctx_records = {}
        for c in self.nodes:
            if c.kind != K.FIELD_DECL:
                continue
            for d in c.walk_preorder():
                if d.kind == K.CALL_EXPR:
                    u, _ = self.callee(d)
                    if u in need:
                        rec = self.parent.get(c.hash)
                        if rec is not None:
                            self.ctx_records[rec.hash] = rec
                        break
        if self.ctx_records:
            for rec in self.ctx_records.values():
                for m in rec.get_children():
                    if m.kind == K.CONSTRUCTOR:
                        need.add(m.get_usr())      # gets the ctx parameter, stores it first
            for c in self.nodes:
                if c.kind == K.VAR_DECL and not self._program_scope(c):
                    decl = _canon(c.type).get_declaration()
                    if decl is not None and decl.hash in self.ctx_records or any(
                            r.spelling == decl.spelling for r in self.ctx_records.values()):
                        fn = self.fn_of.get(c.hash)
                        if fn is not None:
                            need.add(fn.get_usr())
            propagate()
        self.needs_ctx = need

    def _is_extern_shared(self, c):
        if c.kind != K.VAR_DECL or K.CUDASHARED_ATTR not in self._attrs(c):
            return False
        toks = [t.spelling for t in _tokens(c)]
        p = self.parent.get(c.hash)
        prev = self.text[max(0, c.extent.start.offset - 16):c.extent.start.offset]
        return "extern" in toks or "extern" in prev.split()[-1:]

    def _is_mutable_global(self, c):
        if c.hash in self.global_cursors:
            return self.global_cursors[c.hash]
        res = False
        if c.kind == K.VAR_DECL and self._program_scope(c):
            attrs = self._attrs(c)
            const = _deep_const(c.type)
            # an initialiser follows the declarator name ('=' or a brace list); the
            # array bound (``[64]``) is a child expression too but initialises nothing
            toks = [t.spelling for t in _tokens(c)]
            k = toks.index(c.spelling) if c.spelling in toks else -1
            has_init = k >= 0 and any(t in ("=", "{") for t in toks[k + 1:])
            if K.CUDACONSTANT_ATTR in attrs:
                res = not has_init
            elif K.CUDADEVICE_ATTR in attrs:
                res = not (const and has_init)
            else:
                res = False
        self.global_cursors[c.hash] = res
        return res

    # -- edits ----------------------------------------------------------
    def tok_replace(self, tok, new):
        e = tok.extent
        self.rw.replace(e.start.offset, e.end.offset, new)

    def _expr_space(self, e, depth=0):
        """Address space a pointer expression points into, followed through local
        pointer variables and, for parameters, every call site; None if unknown."""
        if depth > 8:
            return None
        while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR) and \
                [k for k in e.get_children() if k.kind.is_expression()]:
            e = [k for k in e.get_children() if k.kind.is_expression()][-1]
        if e.kind == K.BINARY_OPERATOR:
            ptrs = [k for k in e.get_children() if _canon(k.type).kind in (T.POINTER, T.CONSTANTARRAY,
                                                                              T.INCOMPLETEARRAY)]
            return self._expr_space(ptrs[0], depth + 1) if len(ptrs) == 1 else None
        if e.kind == K.UNARY_OPERATOR and _tokens(e) and _tokens(e)[0].spelling == "&":
            ch = list(e.get_children())
            b = ch[0] if ch else None
            # ``&w.a.b`` / ``&arr[i]``: the object the member chain starts from
            while b is not None and b.kind in (K.MEMBER_REF_EXPR, K.ARRAY_SUBSCRIPT_EXPR, K.UNEXPOSED_EXPR,
                                               K.PAREN_EXPR) and b.kind != K.DECL_REF_EXPR:
                if b.kind == K.MEMBER_REF_EXPR and _tokens(b) and "->" in [t.spelling for t in _tokens(b)]:
                    return None             # through a pointer: its own space, unknown here
                kids = [k for k in b.get_children() if k.kind.is_expression()]
                b = kids[0] if kids else None
            if b is not None and b is not ch[0] and b.kind == K.DECL_REF_EXPR and b.referenced is not None \
                    and b.referenced.kind == K.VAR_DECL and _canon(b.referenced.type).kind != T.POINTER:
                return self._var_space(b.referenced, depth)
            r = ch[0].referenced if ch and ch[0].kind == K.DECL_REF_EXPR else None
            if r is not None and r.kind == K.VAR_DECL:
                return self._var_space(r, depth)
            return None
        r = e.referenced if e.kind == K.DECL_REF_EXPR else None
        if r is None:
            return None
        if r.kind == K.VAR_DECL:
            if _canon(r.type).kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY):
                return self._var_space(r, depth)
            if _canon(r.type).kind == T.POINTER and not self._program_scope(r):
                init = [k for k in r.get_children() if k.kind.is_expression()]
                return self._expr_space(init[-1], depth + 1) if init else None
            return None
        if r.kind == K.PARM_DECL:
            fn = self.parent.get(r.hash)
            if fn is None:
                return None
            if self.is_kernel(fn):
                return "device"
            idx = [p.hash for p in self._params(fn)].index(r.hash) if r.hash in [p.hash for p in self._params(fn)] else -1
            args = self._callee_args(fn.get_usr(), idx) if idx >= 0 else []
            sp = {self._expr_space(a, depth + 1) for a in args}
            return sp.pop() if len(sp) == 1 else None
        return None

    def _var_space(self, v, depth):
        if K.CUDASHARED_ATTR in self._attrs(v):
            return "threadgroup"
        if self._program_scope(v):
            return "device" if self._is_mutable_global(v) else "constant"
        return "thread"

    def _param_written(self, p) -> bool:
        """Is by-value parameter ``p`` (or one of its members) assigned in its body?"""
        for c in self.nodes:
            if c.kind not in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR, K.UNARY_OPERATOR):
                continue
            kids = list(c.get_children())
            if not kids:
                continue
            lhs = kids[0]
            if c.kind == K.BINARY_OPERATOR:
                op = [t.spelling for t in _tokens(c) if len(kids) == 2 and lhs.extent.end.offset <= t.extent.start.offset
                      < kids[1].extent.start.offset]
                if op != ["="]:
                    continue
            elif c.kind == K.UNARY_OPERATOR and not any(t.spelling in ("++", "--") for t in _tokens(c)[:1] + _tokens(c)[-1:]):
                continue
            base = lhs
            while base.kind in (K.MEMBER_REF_EXPR, K.ARRAY_SUBSCRIPT_EXPR, K.UNEXPOSED_EXPR, K.PAREN_EXPR):
                ch = [k for k in base.get_children() if k.kind.is_expression()]
                if not ch:
                    break
                base = ch[0]
            if base.kind == K.DECL_REF_EXPR and base.referenced is not None and base.referenced.hash == p.hash:
                return True
        return False

    def _thread_ref_params(self) -> set:
        """Reference parameters bound, at every call site in the module, to a local
        object (or to another such parameter): their address is a thread pointer."""
        fparams = {}
        for c in self.nodes:
            if c.kind in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD):
                ps = [p for p in c.get_children() if p.kind == K.PARM_DECL]
                fparams.setdefault(c.spelling, []).append(ps)
        args = defaultdict(list)          # (fn name, index) -> argument cursors
        for c in self.nodes:
            if c.kind != K.CALL_EXPR or not _tokens(c):
                continue
            name = _tokens(c)[0].spelling
            if name in fparams:
                for i, a in enumerate(x for x in c.get_children() if x.kind.is_expression()
                                       and x.extent.start.offset > _tokens(c)[0].extent.start.offset):
                    args[(name, i)].append(a)
        cand = {}
        for name, lists in fparams.items():
            for ps in lists:
                for i, p in enumerate(ps):
                    if _canon(p.type).kind == T.LVALUEREFERENCE and args.get((name, i)):
                        cand[p.hash] = args[(name, i)]
        res, changed = set(), True

        def local(a):
            while a.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(a.get_children())) == 1:
                a = list(a.get_children())[0]
            r = a.referenced if a.kind == K.DECL_REF_EXPR else None
            if r is None:
                return False
            if r.kind == K.VAR_DECL:
                return not self._program_scope(r) and _canon(r.type).kind not in (T.POINTER, T.LVALUEREFERENCE)
            return r.kind == K.PARM_DECL and r.hash in res
        while changed:
            changed = False
            for h, al in cand.items():
                if h not in res and all(local(a) for a in al):
                    res.add(h)
                    changed = True
        return res

    def edit_fields(self):
        """Pointer members get an address space: ``thread`` when the struct points
        into one of its own array members (storage idiom), else ``device``."""
        thread_fields = set()
        for c in self.nodes:
            if c.kind == K.BINARY_OPERATOR or c.kind == K.MEMBER_REF:
                kids = list(c.get_children())
                if c.kind == K.MEMBER_REF and kids:
                    lhs_ref, rhs = c, kids[0]
                elif len(kids) == 2:
                    lhs_ref, rhs = kids
                else:
                    continue
                lref = lhs_ref.referenced if lhs_ref.kind in (K.MEMBER_REF_EXPR, K.MEMBER_REF) else None
                if lref is None or lref.kind != K.FIELD_DECL or _canon(lref.type).kind != T.POINTER:
                    continue
                r = rhs
                while r.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(r.get_children())) == 1:
                    r = list(r.get_children())[0]
                rr = r.referenced if r.kind in (K.MEMBER_REF_EXPR, K.DECL_REF_EXPR) else None
                if rr is not None and _canon(rr.type).kind == T.CONSTANTARRAY and (
                        rr.kind == K.FIELD_DECL or (rr.kind == K.VAR_DECL and not self._program_scope(rr))):
                    thread_fields.add(lref.hash)
        # brace-initialised records: a pointer field given the address of a local
        # variable (``Ws ws{pool, &dummy, ...}``) points into thread memory, and so does
        # one initialised from such a field (``VI v{i4base + k, ...}``): to a fixed point
        inits = []
        for c in self.nodes:
            if c.kind != K.INIT_LIST_EXPR:
                continue
            decl = _canon(c.type).get_declaration()
            if decl.kind not in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE):
                decl = c.type.get_declaration()          # a dependent ``S<R>{...}``: the template
                if decl.kind not in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE):
                    p = self.parent.get(c.hash)
                    decl = p.type.get_declaration() if p is not None and p.kind == K.VAR_DECL else decl
                    if decl.kind not in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE):
                        continue
            fields = [f for f in decl.get_children() if f.kind == K.FIELD_DECL]
            inits += [(e, f) for e, f in zip([k for k in c.get_children() if k.kind.is_expression()], fields)
                      if _canon(f.type).kind == T.POINTER]

        thread_ref_params = self._thread_ref_params()

        def points_to_thread(e):
            for d in e.walk_preorder():
                if d.kind == K.UNARY_OPERATOR and _tokens(d) and _tokens(d)[0].spelling == "&":
                    ch = list(d.get_children())
                    r = ch[0].referenced if ch and ch[0].kind == K.DECL_REF_EXPR else None
                    if r is not None and r.kind == K.VAR_DECL and not self._program_scope(r):
                        return True
                    if r is not None and r.kind == K.PARM_DECL and r.hash in thread_ref_params:
                        return True
                if d.kind == K.MEMBER_REF_EXPR and d.referenced is not None and d.referenced.hash in thread_fields:
                    return True
            return False
        changed = True
        while changed:
            changed = False
            for e, f in inits:
                if f.hash not in thread_fields and points_to_thread(e):
                    thread_fields.add(f.hash)
                    changed = True
        # pointer members assigned (``w.a.p = base + k``) from expressions whose
        # address space the data flow fixes (a __shared__ workspace passed down)
        assigned_spaces = defaultdict(set)
        for c in self.nodes:
            if c.kind != K.BINARY_OPERATOR:
                continue
            kids = list(c.get_children())
            if len(kids) != 2 or kids[0].kind != K.MEMBER_REF_EXPR or kids[0].referenced is None \
                    or kids[0].referenced.kind != K.FIELD_DECL or _canon(kids[0].referenced.type).kind != T.POINTER:
                continue
            op = [t.spelling for t in _tokens(c) if kids[0].extent.end.offset <= t.extent.start.offset
                  < kids[1].extent.start.offset]
            if op == ["="]:
                assigned_spaces[kids[0].referenced.hash].add(self._expr_space(kids[1]))
        # a pointer member brace-initialised from a __shared__ array: threadgroup
        tg_fields = {h for h, sp in assigned_spaces.items() if sp == {"threadgroup"}}
        for e, f in inits:
            x = e
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            r = x.referenced if x.kind == K.DECL_REF_EXPR else None
            if r is not None and r.kind == K.VAR_DECL and K.CUDASHARED_ATTR in self._attrs(r) \
                    and f.hash not in thread_fields:
                tg_fields.add(f.hash)
        # a member whose default initialiser is a __constant__ array and that nothing
        # re-points (no assignment, no brace initialiser element): constant memory
        assigned = {f.hash for _, f in inits}
        for c in self.nodes:
            if c.kind == K.BINARY_OPERATOR:
                kids = list(c.get_children())
                if len(kids) == 2 and kids[0].kind == K.MEMBER_REF_EXPR and kids[0].referenced is not None:
                    assigned.add(kids[0].referenced.hash)
        constant_fields = set()
        for c in self.nodes:
            if c.kind != K.FIELD_DECL or _canon(c.type).kind != T.POINTER or c.hash in assigned:
                continue
            ex = [x for x in c.get_children() if x.kind.is_expression()]
            if not ex:
                continue
            refs = [d.referenced for d in ex[-1].walk_preorder() if d.kind == K.DECL_REF_EXPR and d.referenced is not None]
            if refs and all(r.kind == K.VAR_DECL and self._program_scope(r) and not self._is_mutable_global(r)
                            and _canon(r.type).kind == T.CONSTANTARRAY for r in refs):
                constant_fields.add(c.hash)
        # inside class templates clang leaves ``this->p = arr`` unresolved: match by name
        records = [c for c in self.nodes if c.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE)]
        for r in records:
            fields = [f for f in r.get_children() if f.kind == K.FIELD_DECL]
            arrays = {f.spelling for f in fields if _canon(f.type).kind in (T.CONSTANTARRAY, T.DEPENDENTSIZEDARRAY,
                                                                              T.INCOMPLETEARRAY)}
            ptrs = {f.spelling: f for f in fields if _canon(f.type).kind == T.POINTER}
            if not arrays or not ptrs:
                continue
            body = self.text[r.extent.start.offset:r.extent.end.offset]
            for m in re.finditer(r"(?:this\s*->\s*)?\b(\w+)\s*=\s*(\w+)\s*[;,)]", body):
                if m.group(1) in ptrs and m.group(2) in arrays:
                    thread_fields.add(ptrs[m.group(1)].hash)
        seen_starts = set()
        # ``T *a, b, *c;`` member declarations mixing pointers and non-pointers: one per member
        groups = defaultdict(list)
        for c in self.nodes:
            if c.kind == K.FIELD_DECL:
                groups[c.extent.start.offset].append(c)
        for start, fs in groups.items():
            kinds = {_canon(f.type).kind == T.POINTER for f in fs}
            if len(fs) < 2 or kinds == {False}:
                continue
            fs.sort(key=lambda f: f.location.offset)
            stoks = _tokens(self.parent.get(fs[0].hash)) if self.parent.get(fs[0].hash) is not None else []
            for k, f in enumerate(fs):
                name_tok = next((t for t in _tokens(f) if t.spelling == f.spelling
                                 and t.extent.start.offset >= f.location.offset), None)
                if name_tok is None:
                    raise TranslationError(f"member {f.spelling}: name not found")
                ft = _canon(f.type).kind
                if ft == T.POINTER:
                    space = ("thread" if f.hash in thread_fields else "threadgroup" if f.hash in tg_fields
                             else "constant" if f.hash in constant_fields else "device")
                    self.field_space[f.hash] = space
                    ty = self.msl_type(f.type, space=space)
                elif ft in (T.CONSTANTARRAY, T.INCOMPLETEARRAY):
                    ty = self.msl_type(_array_dims(f.type)[0], storage=True)
                else:
                    ty = self.msl_type(f.type, storage=True)
                if k == 0:
                    self.rw.replace(start, name_tok.extent.start.offset, ty + " ")
                else:
                    comma = max((t for t in stoks if t.spelling == "," and t.extent.start.offset < name_tok.extent.start.offset),
                                key=lambda t: t.extent.start.offset)
                    self.rw.replace(comma.extent.start.offset, name_tok.extent.start.offset, f"; {ty} ")
            seen_starts.add(start)
        for c in self.nodes:
            if c.kind != K.FIELD_DECL or _canon(c.type).kind != T.POINTER:
                continue
            if c.extent.start.offset in seen_starts and len(groups.get(c.extent.start.offset, ())) > 1:
                continue
            toks = _tokens(c)
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None:
                continue
            space = ("thread" if c.hash in thread_fields else "threadgroup" if c.hash in tg_fields
                     else "constant" if c.hash in constant_fields else "device")
            self.field_space[c.hash] = space
            if _canon(c.type.get_pointee()).kind == T.POINTER:
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, self.msl_type(c.type, space=space) + " ")
            elif c.extent.start.offset not in seen_starts:
                # ``const T *a, *b;``: an address-space specifier distributes over the declarators
                seen_starts.add(c.extent.start.offset)
                self.rw.insert(c.extent.start.offset, space + " ")
        # reference members (``Ws& w;``, RAII markers): a thread pointer, its uses
        # dereferenced and its constructor initialiser taking the address
        ref_fields = set()
        for c in self.nodes:
            if c.kind != K.FIELD_DECL or _canon(c.type).kind != T.LVALUEREFERENCE:
                continue
            name_tok = next((t for t in _tokens(c) if t.spelling == c.spelling), None)
            if name_tok is None:
                continue
            pointee = self.msl_type(c.type.get_pointee())
            self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, f"thread {pointee}* ")
            ref_fields.add(c.hash)
        if ref_fields:
            for c in self.nodes:
                if c.kind == K.MEMBER_REF_EXPR and c.referenced is not None and c.referenced.hash in ref_fields:
                    self.rw.insert(c.extent.start.offset, "(*", order=-7)
                    self.rw.insert(c.extent.end.offset, ")", order=7)
                elif c.kind == K.CONSTRUCTOR:
                    kids = list(c.get_children())
                    for a_, b_ in zip(kids, kids[1:]):
                        if a_.kind == K.MEMBER_REF and a_.referenced is not None and a_.referenced.hash in ref_fields \
                                and b_.kind.is_expression():
                            self.rw.insert(b_.extent.start.offset, "&(", order=-8)
                            self.rw.insert(b_.extent.end.offset, ")", order=8)

    def edit_init_lists(self):
        """Element types of brace initialisers (libclang shows their syntactic form,
        without the implicit conversions): walk the declared type alongside."""
        for c in self.nodes:
            if c.kind != K.VAR_DECL:
                continue
            inits = [ch for ch in c.get_children() if ch.kind == K.INIT_LIST_EXPR]
            if not inits:
                par = self.parent.get(c.hash)
                static_member = par is not None and par.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE)
                if (self._program_scope(c) and not self._is_mutable_global(c)) or static_member:
                    exprs = [ch for ch in c.get_children() if ch.kind.is_expression()]
                    if exprs:
                        self._fold_constant(exprs[-1], c.type, True)
                continue
            storage = self._program_scope(c)
            self._typed_init(inits[0], c.type, storage)

    # -- compile-time evaluation of constant arithmetic (as the CUDA front end folds it)
    def _const_eval(self, c, depth=0):
        """(value, kind) of a constant arithmetic expression, or None.  kind is f64/f32/int;
        f64 is evaluated in Python floats (IEEE binary64, round to nearest), f32 rounds
        every operation to binary32, int follows C (truncating division)."""
        import numpy as np
        if depth > 64:
            return None
        kind = _arith_kind(c.type)
        k = c.kind
        kids = [x for x in c.get_children() if x.kind.is_expression() or x.kind == K.TYPE_REF]
        ex = [x for x in kids if x.kind.is_expression()]

        def cast(v, frm, to):
            if v is None or to is None:
                return None
            if to == "f64":
                return float(v)
            if to == "f32":
                return float(np.float32(v))
            if to in ("int", "bool"):
                if frm in ("f64", "f32"):
                    import math
                    if not math.isfinite(v):
                        return None
                    return int(v) if to == "int" else bool(v)
                return int(v) if to == "int" else bool(v)
            return None
        if k in (K.FLOATING_LITERAL, K.INTEGER_LITERAL):
            v = _literal_value(c)
            return None if v is None else (cast(v, "f64" if k == K.FLOATING_LITERAL else "int", kind), kind)
        if k in (K.PAREN_EXPR, K.UNEXPOSED_EXPR, K.CSTYLE_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR,
                 K.CXX_STATIC_CAST_EXPR) and ex:
            r = self._const_eval(ex[-1], depth + 1)
            if r is None:
                return None
            return cast(r[0], r[1], kind), kind
        if k == K.UNARY_OPERATOR and len(ex) == 1:
            toks = _tokens(c)
            r = self._const_eval(ex[0], depth + 1)
            if r is None or not toks or toks[0].spelling not in ("-", "+"):
                return None
            v = cast(r[0], r[1], kind)
            return (-v if toks[0].spelling == "-" else v), kind
        if k == K.BINARY_OPERATOR and len(ex) == 2:
            a, b = (self._const_eval(x, depth + 1) for x in ex)
            if a is None or b is None or kind not in ("f64", "f32", "int"):
                return None
            op = next((t.spelling for t in _tokens(c)
                       if t.extent.start.offset >= ex[0].extent.end.offset and t.extent.end.offset <= ex[1].extent.start.offset), None)
            x, y = cast(a[0], a[1], kind), cast(b[0], b[1], kind)
            if kind == "int" and op in ("<<", ">>", "&", "|", "^", "%") and x is not None and y is not None:
                if op in ("<<", ">>") and not 0 <= y < 64:
                    return None
                if op == "%":
                    if y == 0:
                        return None
                    r = abs(x) % abs(y)
                    return (r if x >= 0 else -r), kind
                return {"<<": x << y, ">>": x >> y, "&": x & y, "|": x | y, "^": x ^ y}[op], kind
            if op not in ("+", "-", "*", "/") or x is None or y is None:
                return None
            if kind == "int":
                if op == "/":
                    if y == 0:
                        return None
                    q = abs(x) // abs(y)
                    return (q if (x >= 0) == (y >= 0) else -q), kind
                return {"+": x + y, "-": x - y, "*": x * y}[op], kind
            if kind == "f32":
                fx, fy = np.float32(x), np.float32(y)
                with np.errstate(all="ignore"):
                    v = {"+": fx + fy, "-": fx - fy, "*": fx * fy, "/": fx / fy}[op]
                return float(v), kind
            try:
                return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op], kind
            except ZeroDivisionError:
                return None
        if k == K.DECL_REF_EXPR and c.referenced is not None:
            r = c.referenced
            if r.kind == K.ENUM_CONSTANT_DECL:
                return cast(r.enum_value, "int", kind), kind
            if r.kind == K.VAR_DECL and r.type.is_const_qualified():
                inits = [x for x in r.get_children() if x.kind.is_expression()]
                if inits and inits[-1].kind != K.INIT_LIST_EXPR:
                    v = self._const_eval(inits[-1], depth + 1)
                    if v is not None:
                        return cast(v[0], v[1], _arith_kind(r.type)), _arith_kind(r.type)
        return None

    def _fold_constant(self, e, t, storage) -> bool:
        """Replace a non-literal constant floating expression initialising storage of type
        ``t`` by its value: binary64 arithmetic on __f64 is not a constant expression in
        MSL, and program-scope data cannot be computed at run time."""
        to = _arith_kind(t)
        if to not in ("f64", "f32") or _literal_value(e) is not None:
            return False
        if self.rw.in_opaque(e.extent.start.offset, e.extent.end.offset):
            return False
        r = self._const_eval(e)
        if r is None:
            return False
        import numpy as np
        v = float(r[0]) if to == "f64" else float(np.float32(r[0]))
        self.rw.replace(e.extent.start.offset, e.extent.end.offset,
                        _f64_lit(v, storage) if to == "f64" else _hexf32(v), opaque=True)
        for d in e.walk_preorder():
            self._literal_done.add(d.hash)
        return True

    def _typed_init(self, node, t, storage):
        """Walk a brace initialiser alongside the declared type, typing each leaf.
        Brace elision (``{1, 2, 3, 4}`` for ``float m[2][2]`` or an array of
        records) follows C: a sub-aggregate met by a non-brace expression of
        another type takes its elements from the following initialisers."""
        kids = [k for k in node.get_children() if k.kind.is_expression()]
        ct = _canon(t)
        if ct.kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY, T.RECORD):
            it = [0]
            self._fill_aggregate(t, kids, it, storage, top=True)
        elif kids:
            self._typed_leaf(kids[0], t, storage)

    def _aggregate_members(self, t):
        ct = _canon(t)
        if ct.kind == T.CONSTANTARRAY:
            return [t.get_array_element_type()] * ct.element_count
        if ct.kind == T.RECORD:
            decl = ct.get_declaration()
            return [f.type for f in decl.get_children() if f.kind == K.FIELD_DECL]
        return None

    def _fill_aggregate(self, t, kids, it, storage, top=False):
        ct = _canon(t)
        if ct.kind == T.INCOMPLETEARRAY:
            members = None                          # as many elements as initialisers
            et = t.get_array_element_type()
        else:
            members = self._aggregate_members(t)
        i = 0
        while it[0] < len(kids) and (members is None or i < len(members)):
            mt = et if members is None else members[i]
            i += 1
            k = kids[it[0]]
            mct = _canon(mt)
            if k.kind == K.INIT_LIST_EXPR:
                it[0] += 1
                self._typed_init(k, mt, storage)
            elif mct.kind in (T.CONSTANTARRAY, T.RECORD):
                if _canon(k.type).kind == mct.kind and _canon(k.type).spelling == mct.spelling:
                    it[0] += 1                      # copy-initialised from an object
                else:
                    self._fill_aggregate(mt, kids, it, storage)    # brace elision
            else:
                it[0] += 1
                self._typed_leaf(k, mt, storage)

    def _typed_leaf(self, k, t, storage):
        if storage and self._fold_constant(k, t, storage):
            return
        to, frm = _arith_kind(t), _arith_kind(k.type)
        if to is None or frm is None or to == frm or "f64" not in (to, frm):
            return
        v = _literal_value(k)
        s0, s1 = k.extent.start.offset, k.extent.end.offset
        if v is not None:
            self.rw.replace(s0, s1, _f64_lit(v, storage) if to == "f64" else _hexf32(v) if to == "f32"
                            else str(int(v)), opaque=True)
        else:
            tgt = ("__f64m" if storage else "__f64") if to == "f64" else self.msl_type(t).replace("const ", "")
            self.rw.insert(s0, f"__arwen_cvt<{tgt}>(", order=-(s1 - s0))
            self.rw.insert(s1, ")", order=(s1 - s0))
        for d in k.walk_preorder():
            self._literal_done.add(d.hash)
            self._cast_done.add(d.hash)

    def find_reassigned(self):
        self.reassigned = set()
        for c in self.nodes:
            if c.kind == K.BINARY_OPERATOR:
                kids = list(c.get_children())
                if len(kids) == 2 and kids[0].kind == K.DECL_REF_EXPR:
                    toks = _tokens(c)
                    op = [t.spelling for t in toks if kids[0].extent.end.offset <= t.extent.start.offset < kids[1].extent.start.offset]
                    if op == ["="] and kids[0].referenced is not None and kids[0].referenced.kind == K.PARM_DECL:
                        self.reassigned.add(kids[0].referenced.hash)

    def eliminate_gotos(self):
        """Metal has no goto.  Translated Fortran uses two shapes, both rewritten
        exactly:

        * backward ``goto L`` (an iteration written with a label): the statements
          from ``L`` through the last statement holding such a goto are wrapped in
          ``for (;;) { ... break; }`` and each backward goto becomes ``continue``;
        * forward ``goto L``, also out of nested loops (real or the ones above):
          one flag per label, set by the goto, cleared at the label; the goto
          breaks out of the innermost loop it crosses, after each crossed loop
          ``if (flag) break;`` leaves the next one, and above the outermost
          crossed loop every following statement up to the label runs only while
          the flag is clear."""
        gotos = [c for c in self.nodes if c.kind == K.GOTO_STMT]
        if not gotos:
            return

        def fnkey(c):
            f = self.fn_of.get(c.hash)
            return f.hash if f is not None else None
        labels = {(fnkey(c), _tokens(c)[0].spelling): c for c in self.nodes if c.kind == K.LABEL_STMT}
        loops = (K.FOR_STMT, K.WHILE_STMT, K.DO_STMT, K.SWITCH_STMT)

        def target(g):
            t = next((ch for ch in g.get_children() if ch.kind == K.LABEL_REF), None)
            name = t.spelling if t is not None else _tokens(g)[1].spelling
            L = labels.get((fnkey(g), name))
            if L is None:
                raise TranslationError(f"unresolved goto {name}")
            # ``L160: L150: stmt``: consecutive labels name one position; the outermost
            # label statement is the block member, and its name identifies the target
            while self.parent.get(L.hash) is not None and self.parent.get(L.hash).kind == K.LABEL_STMT:
                L = self.parent.get(L.hash)
                name = _tokens(L)[0].spelling
            return name, L

        def path_to(node, block):
            """[node, ..., child of block] or None."""
            out, a = [node], node
            while True:
                p = self.parent.get(a.hash)
                if p is None or p.kind in FN_KINDS:
                    return None
                if p.hash == block.hash:
                    return out
                out.append(p)
                a = p

        def index_in(block, node):
            return next(k for k, ch in enumerate(block.get_children()) if ch.hash == node.hash)

        def stmt_end(st):
            endo = st.extent.end.offset
            return endo if self.text[endo - 1] in ";}" else self.text.find(";", endo) + 1

        # ---- phase A: backward gotos -> synthesized loops
        synth = {}            # block hash -> [(i0, i1, label name)]
        back = {}
        for g in gotos:
            name, L = target(g)
            if L.extent.start.offset < g.extent.start.offset:
                back.setdefault((fnkey(g), name), (L, []))[1].append(g)
        regions = []          # (B, j, kmax, name, L, gs, in_loops)
        for (fk, name), (L, gs) in back.items():
            B = self.parent.get(L.hash)
            if B is None or B.kind != K.COMPOUND_STMT:
                raise TranslationError(f"backward goto {name}: label not in a block")
            j = index_in(B, L)
            kmax = j
            in_loops = []
            for g in gs:
                pth = path_to(g, B)
                if pth is None:
                    raise TranslationError(f"backward goto {name}: not inside the label's block")
                if any(a.kind in loops for a in pth[1:]):
                    in_loops.append((g, [a for a in pth[1:] if a.kind in loops]))
                kmax = max(kmax, index_in(B, pth[-1]))
            regions.append((B, j, kmax, name, L, gs, in_loops))
        # regions of one block that intersect (overlapping or nested) become ONE loop
        # with a restart target: a ``continue`` must reach the right label, and loops
        # synthesized per label would interleave (overlap) or capture each other's
        # ``continue`` (nesting)
        groups = []
        for B in {r[0].hash: r[0] for r in regions}.values():
            rs = sorted((r for r in regions if r[0].hash == B.hash), key=lambda r: r[1])
            cur = [rs[0]]
            for r in rs[1:]:
                if r[1] <= max(x[2] for x in cur):
                    cur.append(r)
                else:
                    groups.append(cur)
                    cur = [r]
            groups.append(cur)
        for grp in groups:
            B = grp[0][0]
            sib = list(B.get_children())
            j = min(r[1] for r in grp)
            kmax = max(r[2] for r in grp)
            gname = "_".join(r[3] for r in grp)
            region = sib[j:kmax + 1]
            for st in region:
                for d in st.walk_preorder():
                    if d.kind == K.DECL_STMT and self.parent.get(d.hash) is not None \
                            and self.parent.get(d.hash).hash == B.hash:
                        raise TranslationError(f"backward goto {gname}: declaration inside the loop it forms")
                    if d.kind in (K.BREAK_STMT, K.CONTINUE_STMT):
                        anc = path_to(d, B) or []
                        if not any(a.kind in loops for a in anc[1:]):
                            raise TranslationError(f"backward goto {gname}: break/continue would bind to the new loop")
            tag = f"{grp[0][4].location.offset}"
            flag = f"__arwen_back_{tag}"
            tgt = f"__arwen_tgt_{tag}"
            multi = len(grp) > 1
            order_of = {r[3]: m + 1 for m, r in enumerate(sorted(grp, key=lambda r: r[1]))}
            label_pos = sorted(r[1] for r in grp)
            in_loops = [x for r in grp for x in r[6]]
            pre = (f"bool {flag} = false; " if in_loops else "") + (f"int {tgt} = 0; " if multi else "")
            if pre:
                self.rw.insert(sib[j].extent.start.offset, pre, order=-3e9)
            if in_loops:
                # a restart from inside loops (``goto L`` to a label before the loop
                # nest): set the flag, leave each crossed loop, and the outermost one is
                # followed by ``continue`` of the synthesized loop (flag cleared there)
                done = set()
                for g, crossed in in_loops:
                    for lp in crossed[:-1]:
                        if lp.hash not in done:
                            done.add(lp.hash)
                            self.rw.insert(stmt_end(lp), f" if ({flag}) break;", order=1e9)
                    top = crossed[-1]
                    if top.hash not in done:
                        done.add(top.hash)
                        self.rw.insert(stmt_end(top), f" if ({flag}) {{ {flag} = false; continue; }}", order=1e9)
            self.rw.insert(sib[j].extent.start.offset, "for (;;) { ", order=-2e9)
            self.rw.insert(stmt_end(sib[kmax]), " break; }", order=2e9)
            if multi:
                # statement i runs when the restart target is at or before it
                for i in range(j, kmax + 1):
                    o = sum(1 for p_ in label_pos if p_ <= i)
                    if o < len(label_pos):
                        self.rw.insert(sib[i].extent.start.offset, f"if ({tgt} <= {o}) {{ ", order=1e8)
                        self.rw.insert(stmt_end(sib[i]), " }", order=-1e8)
            looped = {g.hash for g, _ in in_loops}
            for r in grp:
                m = order_of[r[3]]
                set_t = f"{tgt} = {m}; " if multi else ""
                for g in r[5]:
                    self.rw.replace(g.extent.start.offset, g.extent.end.offset,
                                    f"{{ {flag} = true; {set_t}break; }}" if g.hash in looped
                                    else (f"{{ {set_t}continue; }}" if multi else "continue"))
            synth.setdefault(B.hash, []).append((j, kmax, gname))

        # ---- phase B: forward gotos
        fwd = {}
        for g in gotos:
            name, L = target(g)
            if L.extent.start.offset > g.extent.start.offset:
                fwd.setdefault((fnkey(g), name), (L, []))[1].append(g)
        guarded = set()
        for (fk, name), (L, gs) in fwd.items():
            Lb = self.parent.get(L.hash)
            if Lb is None or Lb.kind != K.COMPOUND_STMT:
                raise TranslationError(f"goto {name}: target not in an enclosing block")
            j = index_in(Lb, L)
            fn = self.fn_of.get(gs[0].hash)
            body = next((ch for ch in fn.get_children() if ch.kind == K.COMPOUND_STMT), None) if fn is not None else None
            if body is None:
                raise TranslationError(f"goto {name}: function body not found")
            flag = f"__arwen_goto_{name}_{L.extent.start.offset}"
            first = list(body.get_children())[0]
            self.rw.insert(first.extent.start.offset, f"bool {flag} = false; ", order=-3e9)
            self.rw.insert(L.extent.start.offset, f"{flag} = false; ", order=-1e9)
            for g in gs:
                pth = path_to(g, Lb)
                if pth is None:
                    raise TranslationError(f"goto {name}: target not in an enclosing block")
                # crossed loop levels, innermost first: ("real", loop) or ("synth", block, i1)
                levels = []
                for a, par in zip(pth, pth[1:] + [Lb]):
                    if par.kind in loops and par.hash != Lb.hash:
                        levels.append(("real", par))
                    if par.kind == K.COMPOUND_STMT:
                        k = index_in(par, a)
                        for (i0, i1, nm) in synth.get(par.hash, []):
                            if i0 <= k <= i1 and not (par.hash == Lb.hash and i0 <= j <= i1):
                                levels.append(("synth", par, i1))
                if not levels:
                    self.rw.replace(g.extent.start.offset, g.extent.end.offset, f"{flag} = true")
                else:
                    self.rw.replace(g.extent.start.offset, g.extent.end.offset, f"{{ {flag} = true; break; }}")
                crossed = 0
                for a, par in zip(pth, pth[1:] + [Lb]):
                    if par.kind in loops and par.hash != Lb.hash:
                        crossed += 1              # exited at its parent block, where a == the loop
                        continue
                    if par.kind != K.COMPOUND_STMT:
                        continue
                    kids = list(par.get_children())
                    k = index_in(par, a)
                    just_exited = a.kind in loops and a.hash != Lb.hash
                    sl = next(((i0, i1) for (i0, i1, nm) in synth.get(par.hash, [])
                               if i0 <= k <= i1 and not (par.hash == Lb.hash and i0 <= j <= i1)), None)
                    if sl is not None:
                        crossed += 1
                        k = sl[1]                 # control leaves after the loop's last statement
                        just_exited = True
                    if levels and crossed == 0:
                        continue                  # inside the innermost crossed loop: its break skips the rest
                    if crossed < len(levels):
                        if just_exited:           # out of one loop, still inside the next crossed one
                            self.rw.insert(stmt_end(kids[k]), f" if ({flag}) break;", order=1.5e9)
                        continue
                    stop = j if par.hash == Lb.hash else len(kids)
                    self._guard_statements(kids[k + 1:stop], flag, guarded, name)
        for L in labels.values():
            toks = _tokens(L)
            if len(toks) >= 2 and toks[1].spelling == ":":
                self.rw.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, "")

    def _guard_statements(self, stmts, flag, guarded, name):
        """Wrap each statement in ``if (!flag) { ... }`` (declarations cannot be wrapped)."""
        for st in stmts:
            if (flag, st.hash) in guarded or st.kind == K.NULL_STMT:
                continue
            if st.kind == K.DECL_STMT:
                raise TranslationError(f"goto {name}: declarations between the goto and its label")
            guarded.add((flag, st.hash))
            endo = st.extent.end.offset
            semi = endo if self.text[endo - 1] in ";}" else self.text.find(";", endo) + 1
            self.rw.insert(st.extent.start.offset, f"if (!{flag}) {{ ", order=-1e8)
            self.rw.insert(semi, " }", order=1e8)

    def run(self) -> Translation:
        self.collect()
        self.analyse()
        self.find_reassigned()
        self.eliminate_gotos()
        self.edit_fields()
        self.edit_init_lists()
        self.edit_linkage()
        self.edit_globals()
        self.edit_block_prototypes()
        self.edit_destructors()
        self.edit_ctx_records()
        self.edit_functions()
        self.edit_lambdas()
        self.edit_dependent_conversions()
        self.edit_dependent_conditionals()
        self.edit_conditional_lvalues()
        self.edit_expressions()
        self.edit_select_minmax()
        if not self.opts.ftz:
            self.edit_float_ops()
        self.edit_tokens()
        body = self.rw.apply()
        return self.finish(body)

    def edit_linkage(self):
        for c in self.nodes:
            if c.kind == K.LINKAGE_SPEC:
                toks = _tokens(c)
                if len(toks) >= 2 and toks[0].spelling == "extern" and toks[1].spelling == '"C"':
                    self.tok_replace(toks[0], "")
                    self.tok_replace(toks[1], "")
                    if len(toks) > 2 and toks[2].spelling == "{":
                        self.tok_replace(toks[2], "")
                        self.tok_replace(toks[-1], "")

    def _global_float_init(self, e, elem):
        """Literal text of a constant floating initialiser (scalar or brace list,
        nested lists flattened in order), each value converted to ``elem`` as C does;
        None when some element is not a constant expression."""
        kind = _arith_kind(elem)
        vals = []

        def walk(x):
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1 \
                    and list(x.get_children())[0].kind == K.INIT_LIST_EXPR:
                x = list(x.get_children())[0]
            if x.kind == K.INIT_LIST_EXPR:
                for ch in x.get_children():
                    if ch.kind.is_expression() and not walk(ch):
                        return False
                return True
            r = self._const_eval(x)
            if r is None:
                return False
            vals.append(float(r[0]))
            return True
        if not walk(e):
            return None
        lits = [_f64_lit(v, True) if kind == "f64" else _hexf32(float(_np32(v))) for v in vals]
        return "{" + ", ".join(lits) + "}" if e.kind == K.INIT_LIST_EXPR or len(lits) != 1 else lits[0]

    def edit_globals(self):
        offset = 0
        for c in self.nodes:
            if c.kind != K.VAR_DECL or not self._program_scope(c):
                continue
            toks = _tokens(c)
            if c.hash in self.ssmem_globals or (K.CUDASHARED_ATTR in self._attrs(c) and not self._is_extern_shared(c)):
                # program-scope __shared__: a slot in the static threadgroup area
                end = c.extent.end.offset
                semi = self.text.find(";", end)
                self.rw.replace(c.extent.start.offset, semi + 1, f"/* __shared__ {c.spelling} -> threadgroup slot */",
                                opaque=True)
                continue
            if self._is_mutable_global(c):
                self.module_uses_globals = True
                t = c.type
                ct = _canon(t)
                size, align = ct.get_size(), ct.get_align()
                if size <= 0:
                    raise TranslationError(f"global {c.spelling}: unsized type")
                offset = (offset + align - 1) // align * align
                init = None
                children = [ch for ch in c.get_children() if ch.kind.is_expression()]
                ttoks = [t.spelling for t in toks]
                kname = ttoks.index(c.spelling) if c.spelling in ttoks else -1
                if children and kname >= 0 and any(x in ("=", "{") for x in ttoks[kname + 1:]):
                    # the initialiser, not an array bound (``T a[64];`` has 64 as a child)
                    init = self.text[children[-1].extent.start.offset:children[-1].extent.end.offset]
                elem, dims = _array_dims(t)
                if init is not None and _arith_kind(elem) in ("f64", "f32"):
                    # floating initialisers are emitted as exact literals of the storage type
                    init = self._global_float_init(children[-1], elem) or init
                mty = self.msl_type(elem, storage=True).replace("const ", "")
                gv = GlobalVar(c.spelling, mty, size, align, offset, init)
                gv.dims = dims
                self.globals_mut[c.get_usr()] = gv
                offset += size
                # remove the declaration (through its terminating semicolon)
                end = c.extent.end.offset
                semi = self.text.find(";", end)
                self.rw.replace(c.extent.start.offset, semi + 1, f"/* global {c.spelling} -> globals buffer */",
                                opaque=True)
            else:
                spellings = [t.spelling for t in toks]
                done = False
                for t in toks:
                    if t.spelling in ("__device__", "__constant__"):
                        self.tok_replace(t, "constant" if not done else "")
                        done = True
                if not done:
                    first = toks[0]
                    self.rw.insert(first.extent.start.offset, "constant ")
        self.globals_size = offset

    def _call_lparen(self, call):
        """Offset just after the '(' that opens a call's argument list."""
        ch = list(call.get_children())
        ref = call.referenced
        if ref is not None and ref.kind == K.CONSTRUCTOR:
            # ``T x(a, b)`` / ``T x{a, b}`` / ``T(a)``: the opener before the first argument
            args = [k for k in ch if k.kind.is_expression()]
            limit = args[0].extent.start.offset if args else call.extent.end.offset
            opener = None
            for t in _tokens(call):
                if t.extent.start.offset < limit and t.spelling in ("(", "{"):
                    opener = t
            return opener.extent.end.offset if opener is not None else None
        if not ch:
            return None
        callee_end = ch[0].extent.end.offset
        for t in _tokens(call):
            if t.extent.start.offset >= callee_end and t.spelling == "(":
                return t.extent.end.offset
        return None

    def edit_block_prototypes(self):
        """Block-scope function prototypes go: the functions are defined at file scope,
        and the calls, inside translated (template) functions with dependent
        arguments, find them at instantiation."""
        for st in self.block_protos.values():
            kids = list(st.get_children())
            if any(k.kind not in FN_KINDS for k in kids):
                raise TranslationError("block-scope prototype declared together with variables")
            self.rw.replace(st.extent.start.offset, st.extent.end.offset, ";", opaque=True)

    def edit_functions(self):
        for usr, decls in self.functions.items():
            kernel = any(self.is_kernel(d) for d in decls)
            for d in sorted(decls, key=lambda d: d.extent.start.offset):   # first declaration first
                if kernel:
                    self.edit_kernel(d)
                else:
                    self.edit_device_function(d, usr)

    def _reaches_ssmem(self, kusr) -> bool:
        if not self.ssmem_users:
            return False
        # walk callers upward from the ssmem users and see whether the kernel is among them
        seen, work = set(self.ssmem_users), list(self.ssmem_users)
        while work:
            u = work.pop()
            if u == kusr:
                return True
            for c in self.callers.get(u, ()):
                if c not in seen:
                    seen.add(c)
                    work.append(c)
        return kusr in seen

    def edit_kernel(self, fn):
        body = self._body(fn)
        if body is None:
            raise TranslationError(f"kernel {fn.spelling} declared without a body")
        name = fn.spelling
        params = self._params(fn)
        sig = self.text[fn.extent.start.offset:body.extent.start.offset]
        m = re.search(r"__launch_bounds__\s*\(\s*([^,)]+)", sig)
        max_threads = None
        attr = ""
        if m:
            attr = f"[[max_total_threads_per_threadgroup({m.group(1).strip()})]] "
            try:
                max_threads = int(eval(m.group(1), {}, {}))
            except Exception:
                max_threads = None
        fields, unpack, infos, off = [], [], [], 0
        for i, p in enumerate(params):
            pname = _ident(p.spelling) if p.spelling else f"__arwen_p{i}"
            t = p.type
            ct = _canon(t)
            if ct.kind == T.POINTER and _canon(t.get_pointee()).kind == T.POINTER:
                # Metal forbids pointers to pointers in buffers: carry the GPU address
                fty = "ulong"
                size = align = 8
                kind, code = "ptr", "P"
                unpack_ty = self.msl_type(t, space="device")
                infos_cast = True
            elif ct.kind == T.POINTER:
                fty = self.msl_type(t, space="device")
                size = align = 8
                kind, code = "ptr", "P"
                unpack_ty = fty
            elif ct.kind == T.RECORD:
                fty = self.msl_type(t)
                size, align = ct.get_size(), ct.get_align()
                kind, code = "struct", f"{size}s"
                # read in place from the constant argument buffer: a by-value copy of a
                # large struct lands in thread-private memory.  A small struct with
                # member functions is copied (its methods take a thread ``this``).
                rec = ct.get_declaration()
                has_methods = any(m.kind == K.CXX_METHOD and not m.is_static_method() for m in rec.get_children())
                copy = (has_methods and size <= 512) or self._param_written(p)
                unpack_ty = (fty.replace("const ", "") if copy
                             else "constant " + fty.replace("const ", "") + "&")
            else:
                abi = self.scalar_abi(t)
                if abi is None:
                    raise TranslationError(f"kernel {name}: unsupported parameter type {t.spelling}")
                size, align, code = abi
                kind = "scalar"
                fty = "__f64m" if _is_double(t) else self.msl_type(t).replace("const ", "")
                unpack_ty = "__f64" if _is_double(t) else fty
            off = (off + align - 1) // align * align
            fields.append(f"    {fty} {pname};")
            unpack.append(f"{unpack_ty} {pname} = ({unpack_ty})__arwen_a.{pname};" if fty == "ulong" and kind == "ptr"
                          else f"{unpack_ty} {pname} = __arwen_a.{pname};")
            infos.append(KernelParam(pname, kind, fty, size, align, off, code))
            off += size
        args_size = (off + 7) // 8 * 8 if off else 0
        struct_def = (f"struct __arwen_args_{name} {{\n" + ("\n".join(fields) if fields else "    int __arwen_empty;")
                      + "\n};\n")
        extra = []
        extra.append("device uchar* __arwen_g [[buffer(2)]]")
        extra.append("device metal::atomic_uint* __arwen_status [[buffer(3)]]")
        extra.append("threadgroup uchar* __arwen_dsmem [[threadgroup(0)]]" if self.dyn_shared_used else "")
        sig_new = (f"{struct_def}{attr}kernel void {name}(constant __arwen_args_{name}& __arwen_a [[buffer(0)]], "
                   f"constant __arwen_launch_t& __arwen_L [[buffer(1)]], "
                   + ", ".join(x for x in extra if x) +
                   ", uint3 __arwen_tid [[thread_position_in_threadgroup]], uint3 __arwen_bid [[threadgroup_position_in_grid]], "
                   "uint3 __arwen_bdim [[threads_per_threadgroup]], uint3 __arwen_gdim [[threadgroups_per_grid]], "
                   "uint __arwen_lane [[thread_index_in_simdgroup]])\n")
        self.rw.replace(fn.extent.start.offset, body.extent.start.offset, sig_new)
        ss = self._reaches_ssmem(fn.get_usr())
        prologue = (("\n    threadgroup uchar __arwen_ssmem[%d] __attribute__((aligned(16)));" % self.ssmem_size if ss else "")
                    + "\n    const __arwen_ctx_t __arwen_ctx = __arwen_make_ctx(__arwen_L, __arwen_tid, __arwen_bid, "
                    "__arwen_bdim, __arwen_gdim, __arwen_lane, __arwen_g, __arwen_status"
                    + (", __arwen_dsmem" if self.dyn_shared_used else ", (threadgroup uchar*)0")
                    + (", __arwen_ssmem" if ss else ", (threadgroup uchar*)0") + ");\n    "
                    + " ".join(unpack) + "\n    if (__arwen_ctx.inactive) return;\n")
        self.rw.insert(body.extent.start.offset + 1, prologue)
        text = self.text[body.extent.start.offset:body.extent.end.offset]
        self.kernel_infos[name] = KernelInfo(
            name, infos, args_size, max_threads,
            uses_barrier=bool(re.search(r"__syncthreads|__syncwarp", text)),
            uses_shared=bool(re.search(r"__shared__", text)),
            uses_warp=bool(re.search(r"__shfl|__ballot|__any_sync|__all_sync", text)),
            dyn_shared=self.dyn_shared_used)

    def _record_has_pointers(self, t) -> bool:
        ct = _canon(t)
        if ct.kind != T.RECORD:
            return False
        decl = ct.get_declaration()
        fields = [f for f in decl.get_children() if f.kind == K.FIELD_DECL]
        if not fields:
            # an implicit template specialisation: look at the primary template
            name = ct.spelling.split("<")[0].replace("const ", "").replace("struct ", "").strip().split("::")[-1]
            for c in self.nodes:
                if c.kind == K.CLASS_TEMPLATE and c.spelling == name:
                    fields = [f for f in c.get_children() if f.kind == K.FIELD_DECL]
                    break
        for f in fields:
            fk = _canon(f.type).kind
            if fk == T.POINTER or (fk == T.RECORD and self._record_has_pointers(f.type)):
                return True
            if "__ARWEN_AP" in f.type.spelling:       # a parametrized view (parametrize_views)
                return True
        return False

    def _deducing_type_param(self, fn, usr, pointee):
        """The template type parameter ``R`` a ``[const] R*`` parameter deduces, when
        the function is defined once and never called with explicit arguments."""
        if fn.kind != K.FUNCTION_TEMPLATE or len(self.functions.get(usr, [])) != 1:
            return None
        if self._call_shapes_of(fn) & {"<"}:
            return None
        name = re.sub(r"\bconst\b|\bvolatile\b", "", pointee.spelling).strip()
        return next((ch for ch in fn.get_children() if ch.kind == K.TEMPLATE_TYPE_PARAMETER
                     and ch.spelling == name), None)

    def _call_shapes_of(self, fn):
        self._explicit_targs_everywhere(fn)
        return self._call_shapes.get(fn.spelling, set())

    def _explicit_targs_everywhere(self, fn) -> bool:
        """Every call of ``fn`` in the module names its template arguments
        (``f<R>(...)``): its parameters need not deduce them."""
        if not hasattr(self, "_call_shapes"):
            self._call_shapes = defaultdict(set)
            for c in self.nodes:
                if c.kind == K.CALL_EXPR:
                    toks = _tokens(c)
                    if len(toks) > 1:
                        self._call_shapes[toks[0].spelling].add(toks[1].spelling)
        shapes = self._call_shapes.get(fn.spelling)
        return bool(shapes) and shapes == {"<"}

    def _deducible_elsewhere(self, params, i) -> bool:
        """Is the dependent pointee type of params[i] also named by a non-pointer parameter?"""
        name = re.sub(r"\b(const|volatile)\b|[*&\s]", "", params[i].type.get_pointee().spelling)
        for j, q in enumerate(params):
            if j == i or _canon(q.type).kind == T.POINTER:
                continue
            if re.search(r"\b%s\b" % re.escape(name), q.type.spelling):
                return True
        return False

    @staticmethod
    def _dependent(t) -> bool:
        k, ck = t.kind, _canon(t).kind
        if k in (T.UNEXPOSED, T.DEPENDENTSIZEDARRAY) or ck in (T.UNEXPOSED, T.DEPENDENT):
            return True
        sp = _canon(t).spelling
        return "type-parameter" in sp or "decltype" in sp

    def _param_default(self, p) -> str:
        kids = [k for k in p.get_children() if k.kind.is_expression()]
        toks = [t.spelling for t in _tokens(p)]
        k = toks.index(p.spelling) if p.spelling in toks else -1
        if not kids or k < 0 or "=" not in toks[k + 1:]:
            return ""                   # an array bound (``T (*a)[202]``) is no default
        d = kids[-1]
        return " = " + self.text[d.extent.start.offset:d.extent.end.offset]

    def _overload_constraints(self, fn, params, tparams):
        if not hasattr(self, "_overloaded"):
            names = defaultdict(set)
            for u, ds in self.functions.items():
                names[ds[0].spelling].add(u)
            self._overloaded = {n for n, us in names.items() if len(us) > 1}
        if fn.spelling not in self._overloaded:
            return []
        tps = {t for t, _ in tparams}
        cons = []
        for i, p in enumerate(params):
            tp = f"__arwen_P{i}"
            ck = _canon(p.type).kind
            if tp not in tps or ck not in (T.POINTER, T.LVALUEREFERENCE):
                continue
            pt = p.type.get_pointee()
            # scalar pointees only: a record parameter also binds derived (flattened) records
            if self._dependent(pt) or _arith_kind(pt) is None:
                continue
            cons.append(f"__arwen_points_to<{tp}, {self.msl_type(pt, storage=True).replace('const ', '')}>()")
        return cons

    def edit_device_function(self, fn, usr):
        params = self._params(fn)
        tparams, refs = [], {}
        for i, p in enumerate(params):
            t = p.type
            ct = _canon(t)
            pname = _ident(p.spelling)
            if ct.kind == T.POINTER and t.get_pointee().is_const_qualified() and p.hash in self.reassigned \
                    and _canon(t.get_pointee()).kind != T.POINTER:
                # re-pointed between global and threadgroup memory at run time
                ety = self.msl_type(t.get_pointee(), storage=True)
                self.rw.replace(p.extent.start.offset, p.extent.end.offset,
                                f"__arwen_gptr<{ety}> {pname}{self._param_default(p)}")
            elif ct.kind == T.POINTER and self._dependent(t.get_pointee()) and not self._deducible_elsewhere(params, i) \
                    and not self._explicit_targs_everywhere(fn):
                # the function template deduces its own parameter from T*
                tpar = self._deducing_type_param(fn, usr, t.get_pointee())
                if tpar is not None and tpar.hash not in self._pointee_params_done:
                    # ``template <class R> f(const R* p)``: p generic (any address space),
                    # R its pointee: ``template <typename P, class R = __arwen_pointee_t<P>>``
                    self._pointee_params_done.add(tpar.hash)
                    tp = f"__arwen_P{i}"
                    self.rw.insert(tpar.extent.start.offset, f"typename {tp}, ")
                    self.rw.insert(tpar.extent.end.offset, f" = __arwen_pointee_t<{tp}>")
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}{self._param_default(p)}")
                else:
                    # keep it deducible, in device memory
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset,
                                    f"{self.msl_type(t, space='device')} {pname}{self._param_default(p)}")
            elif ct.kind == T.POINTER:
                tp = f"__arwen_P{i}"
                tparams.append((tp, self.msl_type(t, space="device")))
                self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}{self._param_default(p)}")
            elif ct.kind in (T.INCOMPLETEARRAY, T.CONSTANTARRAY):
                # an array-declarator parameter (``const double c[][2]``) is a pointer:
                # its address space comes from the argument, as for a pointer
                tp = f"__arwen_P{i}"
                tparams.append((tp, None))
                self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}")
            elif ct.kind == T.RECORD and self._record_has_pointers(t):
                # views holding pointers: duck-typed, so a flattened derived view or a
                # view over another address space binds too
                tp = f"__arwen_P{i}"
                tparams.append((tp, None))
                self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}")
            elif ct.kind == T.LVALUEREFERENCE and _canon(t.get_pointee()).kind in (
                    T.CONSTANTARRAY, T.DEPENDENTSIZEDARRAY, T.INCOMPLETEARRAY) and self._array_arg_space(usr, i):
                # ``const int (&a)[N]``: stays a reference to array (N keeps deducing),
                # in the address space every call site's array lives in
                self.rw.insert(p.extent.start.offset, self._array_arg_space(usr, i) + " ")
                refs[i] = "value"
            elif ct.kind in (T.LVALUEREFERENCE, T.RVALUEREFERENCE):
                pt = t.get_pointee()
                if pt.is_const_qualified() and ct.kind == T.LVALUEREFERENCE and _canon(pt).kind != T.RECORD:
                    ty = self.msl_type(pt).replace("const ", "")
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{ty} {pname}{self._param_default(p)}")
                    refs[i] = "value"
                elif ct.kind == T.LVALUEREFERENCE and not self._has_resolved_calls(usr) \
                        and _canon(pt).kind not in (T.CONSTANTARRAY, T.DEPENDENTSIZEDARRAY, T.INCOMPLETEARRAY):
                    # only called through dependent expressions (a functor invoked by a
                    # template): call sites cannot take addresses, so a true reference,
                    # to thread memory (locals); any other space fails to compile
                    ty = self.msl_type(pt, storage=True)
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"thread {ty}& {pname}")
                    refs[i] = "value"
                elif ct.kind == T.LVALUEREFERENCE or pt.is_const_qualified():
                    tp = f"__arwen_P{i}"
                    tparams.append((tp, None))
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}")
                    refs[i] = "ptr"
                else:
                    raise TranslationError(f"{fn.spelling}: rvalue reference parameter")
        if refs:
            self.ref_params[usr] = refs
        rt = fn.result_type
        if fn.kind != K.CONSTRUCTOR and _canon(rt).kind in (T.POINTER, T.LVALUEREFERENCE):
            # return type: spelled between the declaration start and the name
            toks = _tokens(fn)
            nm = "operator" if fn.spelling.startswith("operator") else fn.spelling
            name_tok = next((t for t in toks if t.spelling == nm), None)
            if name_tok is not None:
                rstart = None
                depth = 0                         # inside a ``template <...>`` parameter list
                for i_t, t in enumerate(toks):
                    if t.extent.start.offset >= name_tok.extent.start.offset:
                        break
                    if t.spelling == "template" and i_t + 1 < len(toks) and toks[i_t + 1].spelling == "<":
                        continue
                    if t.spelling == "<" and (depth or (i_t and toks[i_t - 1].spelling == "template")):
                        depth += 1
                        continue
                    if depth:
                        depth += {"<": 1, ">": -1}.get(t.spelling, 0)
                        if t.spelling == ">>":
                            depth -= 2
                        continue
                    if t.spelling in ("static", "inline", "__device__", "__host__", "__forceinline__",
                                      "__noinline__", "template", "constexpr", "extern"):
                        continue
                    if rstart is None:
                        rstart = t.extent.start.offset
                if rstart is not None:
                    new = "decltype(auto)" if _canon(rt).kind == T.LVALUEREFERENCE else "auto"
                    self.rw.replace(rstart, name_tok.extent.start.offset, new + " ")
        ctx = usr in self.needs_ctx
        if ctx:
            lp = None
            for t in _tokens(fn):
                if t.spelling == "(" and t.extent.start.offset > fn.location.offset:
                    lp = t.extent.end.offset
                    break
            if lp is None:
                raise TranslationError(f"{fn.spelling}: no parameter list")
            self.rw.insert(lp, "thread const __arwen_ctx_t& __arwen_ctx" + (", " if params else ""), order=-1)
        if tparams:
            # a default type lets calls that omit a defaulted (null) pointer argument deduce
            # default template arguments may appear on the first declaration only
            first_decl = usr not in self._tmpl_defaults_done
            self._tmpl_defaults_done.add(usr)
            hdr = ", ".join(f"typename {t}" + (f" = {d}" if d and first_decl else "") for t, d in tparams)
            cons = self._overload_constraints(fn, params, tparams)
            if cons:
                # overloads told apart by pointee type (``float*`` / ``double*``) stay
                # distinct once their pointer parameters are generic
                # (a non-type parameter: templates differing only in a default type
                # argument would be one template)
                hdr += f", metal::enable_if_t<{' && '.join(cons)}, int>" + (" = 0" if first_decl else "")
            if fn.kind == K.FUNCTION_TEMPLATE:
                toks = _tokens(fn)
                # extend the existing template parameter list
                depth, close = 0, None
                for t in toks:
                    if t.spelling == "<":
                        depth += 1
                    elif t.spelling == ">":
                        depth -= 1
                        if depth == 0:
                            close = t
                            break
                    elif t.spelling == ">>":
                        depth -= 2
                        if depth <= 0:
                            close = t
                            break
                if close is None:
                    raise TranslationError(f"{fn.spelling}: template header not found")
                self.rw.insert(close.extent.start.offset, ", " + hdr)
            else:
                self.rw.insert(fn.extent.start.offset, f"template <{hdr}> ", order=-10)

    def edit_lambdas(self):
        """Lambdas: pointer / reference parameters become generic (``auto``), so the
        address space is the argument's; a reference return type becomes
        ``decltype(auto)`` and a pointer return type is deduced."""
        for lam in self.nodes:
            if lam.kind != K.LAMBDA_EXPR:
                continue
            params = [p for p in lam.get_children() if p.kind == K.PARM_DECL]
            for p in params:
                ck = _canon(p.type).kind
                if ck not in (T.POINTER, T.LVALUEREFERENCE):
                    continue
                name_tok = next((t for t in _tokens(p) if t.spelling == p.spelling), None)
                if name_tok is None:
                    continue
                # MSL needs an address space on ``auto&``; a forwarding reference binds an
                # lvalue in any space as ``AS T&`` (same semantics for lvalue arguments)
                self.rw.replace(p.extent.start.offset, name_tok.extent.start.offset,
                                "auto " if ck == T.POINTER else "thread auto& ")
            toks = _tokens(lam)
            body = next((ch for ch in lam.get_children() if ch.kind == K.COMPOUND_STMT), None)
            if body is None:
                continue
            arrow = next((t for t in toks if t.spelling == "->" and t.extent.start.offset < body.extent.start.offset
                          and (not params or t.extent.start.offset > params[-1].extent.end.offset)), None)
            if arrow is not None:
                rk = _canon(lam.type).kind
                text = self.text[arrow.extent.end.offset:body.extent.start.offset]
                if "&" in text:
                    self.rw.replace(arrow.extent.start.offset, body.extent.start.offset, "-> decltype(auto) ")
                elif "*" in text:
                    self.rw.replace(arrow.extent.start.offset, body.extent.start.offset, " ")

    _ARITH_KEYWORDS = {"double", "float", "int", "unsigned", "long", "short", "char", "bool"}

    def _edit_dependent_construct(self, c) -> bool:
        """``double(e)`` with a type-dependent ``e`` (template code) is an unresolved
        construct expression, which libclang shows as a call: the same conversion
        as a cast, spelled with __arwen_cvt (right for every source type)."""
        toks = _tokens(c)
        if len(toks) < 3 or toks[1].spelling != "(" or toks[-1].spelling != ")":
            return False
        kids = list(c.get_children())
        if toks[0].spelling in self._ARITH_KEYWORDS and _arith_kind(c.type) is not None:
            target = self.msl_type(c.type).replace("const ", "")
        elif kids and kids[0].kind == K.TYPE_REF and kids[0].spelling.split()[-1] == toks[0].spelling \
                and len([k for k in kids if k.kind.is_expression()]) == 1 and self._dependent(c.type):
            # ``T(a)`` with a dependent alias ``using T = decltype(a + b)``
            target = toks[0].spelling
        else:
            return False
        self.rw.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, f"__arwen_cvt<{target}>(")
        return True

    def _balanced_end(self, start, end):
        """Extend ``end`` past closing parentheses the extent left out (libclang ends an
        unresolved ``T(x)`` construct before its ``)``)."""
        depth = 0
        for ch in self.text[start:end]:
            depth += {"(": 1, ")": -1}.get(ch, 0)
        while depth > 0:
            j = self.text.find(")", end)
            if j < 0:
                break
            depth -= 1 + self.text[end:j].count(")") - self.text[end:j].count("(")
            end = j + 1
        return end

    def _resolved_arith(self, e):
        """Arithmetic kind of ``e`` when it is known despite a dependent context
        (``a[k]`` on ``float a[KMAX]`` is type-dependent to clang, its element is not)."""
        while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(e.get_children())) == 1:
            e = list(e.get_children())[0]
        if not self._dependent(e.type):
            return _arith_kind(e.type)
        if e.kind == K.ARRAY_SUBSCRIPT_EXPR:
            base = list(e.get_children())[0]
            while base.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(base.get_children())) == 1:
                base = list(base.get_children())[0]
            t = base.referenced.type if base.kind == K.DECL_REF_EXPR and base.referenced is not None else base.type
            if t.kind in (T.CONSTANTARRAY, T.DEPENDENTSIZEDARRAY, T.INCOMPLETEARRAY):
                el = t.get_array_element_type()
            elif _canon(t).kind == T.POINTER:
                el = t.get_pointee()
            else:
                return None
            return None if self._dependent(el) else _arith_kind(el)
        if e.kind == K.DECL_REF_EXPR and e.referenced is not None and not self._dependent(e.referenced.type):
            return _arith_kind(e.referenced.type)
        return None

    def edit_dependent_conditionals(self):
        """``c ? a : b`` in template code with a type-dependent branch: the branches
        go through __arwen_tsel (double when either branch is double, else identity),
        unless the result is assigned to (an lvalue conditional)."""
        for c in self.nodes:
            if c.kind != K.CONDITIONAL_OPERATOR:
                continue
            kids = [k for k in c.get_children() if k.kind.is_expression()]
            if len(kids) != 3:
                continue
            _, a, b = kids
            if not (self._dependent(a.type) or self._dependent(b.type)):
                continue
            ka, kb = self._resolved_arith(a), self._resolved_arith(b)
            if ka and kb and (ka == kb or "f64" not in (ka, kb)):
                continue                    # the conditional's type is already right
            if _canon(a.type).kind == T.POINTER or _canon(b.type).kind == T.POINTER:
                continue
            p = self.parent.get(c.hash)
            while p is not None and p.kind == K.PAREN_EXPR:
                p = self.parent.get(p.hash)
            if p is not None and p.kind in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
                lhs = list(p.get_children())[0]
                if lhs.extent.start.offset <= c.extent.start.offset and c.extent.end.offset <= lhs.extent.end.offset:
                    continue
            toks = _tokens(c)
            q = next((t for t in toks if t.spelling == "?" and kids[0].extent.end.offset <= t.extent.start.offset
                      < a.extent.start.offset), None)
            col = next((t for t in toks if t.spelling == ":" and a.extent.end.offset <= t.extent.start.offset
                        < b.extent.start.offset), None)
            if q is None or col is None:
                continue
            # end of the last token (an unresolved construct's extent can stop before
            # its closing parenthesis); nested inside wrappers of exactly this span
            # (order -n), outside those of the condition / last branch (shorter spans)
            end = self._balanced_end(c.extent.start.offset, max(c.extent.end.offset, toks[-1].extent.end.offset))
            n = end - c.extent.start.offset
            self.rw.insert(c.extent.start.offset, "__arwen_csel(", order=-n + 0.5)
            self.rw.replace(q.extent.start.offset, q.extent.end.offset, ", [&]() { return ")
            self.rw.replace(col.extent.start.offset, col.extent.end.offset, "; }, [&]() { return ")
            self.rw.insert(end, "; })", order=n - 0.5)

    def _lvalue_space(self, e):
        while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(e.get_children())) == 1:
            e = list(e.get_children())[0]
        if e.kind == K.DECL_REF_EXPR and e.referenced is not None and e.referenced.kind == K.VAR_DECL:
            return self._var_space(e.referenced, 0)
        if e.kind == K.ARRAY_SUBSCRIPT_EXPR:
            return self._expr_space(list(e.get_children())[0])
        return None

    def edit_conditional_lvalues(self):
        """``(c ? a : b) = v`` with a and b in different address spaces (a local and a
        device element): no common lvalue type in MSL.  As a statement it becomes
        ``__arwen_sel_assign(c, [&](auto x){ a = x; }, [&](auto x){ b = x; }, v)``."""
        for asg in self.nodes:
            if asg.kind != K.BINARY_OPERATOR:
                continue
            kids = list(asg.get_children())
            if len(kids) != 2:
                continue
            lhs, rhs = kids
            x = lhs
            while x.kind in (K.PAREN_EXPR, K.UNEXPOSED_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            if x.kind != K.CONDITIONAL_OPERATOR:
                continue
            eq = [t for t in _tokens(asg) if lhs.extent.end.offset <= t.extent.start.offset < rhs.extent.start.offset]
            if [t.spelling for t in eq] != ["="]:
                continue
            ck = [k for k in x.get_children() if k.kind.is_expression()]
            if len(ck) != 3:
                continue
            cond, a, b = ck
            sa, sb = self._lvalue_space(a), self._lvalue_space(b)
            if sa is None or sb is None or sa == sb:
                continue
            par = self.parent.get(asg.hash)
            if par is None or par.kind not in (K.COMPOUND_STMT, K.FOR_STMT, K.IF_STMT, K.WHILE_STMT, K.DO_STMT):
                raise TranslationError("conditional lvalue across address spaces used as a value")
            toks = _tokens(x)
            q = next(t for t in toks if t.spelling == "?" and cond.extent.end.offset <= t.extent.start.offset < a.extent.start.offset)
            col = next(t for t in toks if t.spelling == ":" and a.extent.end.offset <= t.extent.start.offset < b.extent.start.offset)
            n = asg.extent.end.offset - asg.extent.start.offset
            self.rw.insert(asg.extent.start.offset, "__arwen_sel_assign(", order=-n - 0.5)
            self.rw.replace(q.extent.start.offset, q.extent.end.offset, ", [&](auto __x) { ")
            self.rw.insert(a.extent.end.offset, " = __x; }", order=1e8)
            self.rw.replace(col.extent.start.offset, col.extent.end.offset, ", [&](auto __x) { ")
            self.rw.insert(b.extent.end.offset, " = __x; }", order=1e8)
            self.rw.replace(eq[0].extent.start.offset, eq[0].extent.end.offset, ", ")
            self.rw.insert(asg.extent.end.offset, ")", order=n + 0.5)
            p = lhs
            while p.kind in (K.PAREN_EXPR, K.UNEXPOSED_EXPR) and p.extent.start.offset < x.extent.start.offset:
                if p.kind == K.PAREN_EXPR:      # the parentheses around the conditional go
                    lt = _tokens(p)
                    self.rw.replace(lt[0].extent.start.offset, lt[0].extent.end.offset, "")
                    self.rw.replace(lt[-1].extent.start.offset, lt[-1].extent.end.offset, "")
                p = list(p.get_children())[0]

    def edit_dependent_conversions(self):
        """Template code: a type-dependent value initialising or assigned to an
        arithmetic object converts to the object's type (C++ copy-initialisation),
        which the dependent AST leaves implicit.  Spelled __arwen_cvt<T>, exact for
        every source type.  ``d op= e`` with a double ``d`` promotes ``e`` to double
        first; other compound assignments keep their operator (ARWEN_F32_COMPOUND_F64)."""
        def wrap(e, t):
            target = self.msl_type(t).replace("const ", "")
            toks = _tokens(e)
            end = self._balanced_end(e.extent.start.offset, max(e.extent.end.offset, toks[-1].extent.end.offset if toks else 0))
            n = end - e.extent.start.offset
            self.rw.insert(e.extent.start.offset, f"__arwen_cvt<{target}>(", order=-n - 0.25)
            self.rw.insert(end, ")", order=n + 0.25)

        for c in self.nodes:
            if c.kind == K.VAR_DECL and not self._program_scope(c) and self._dependent(c.type) \
                    and re.fullmatch(r"(const\s+)?\w+", c.type.spelling):
                # ``T r = 1;`` with T a template parameter: copy-initialisation converts
                toks = [x.spelling for x in _tokens(c)]
                kname = toks.index(c.spelling) if c.spelling in toks else -1
                ex = [x for x in c.get_children() if x.kind.is_expression()]
                if ex and kname >= 0 and "=" in toks[kname + 1:kname + 2] and not self._dependent(ex[-1].type) \
                        and _arith_kind(ex[-1].type) is not None:
                    e = ex[-1]
                    tname = c.type.spelling.replace("const ", "").strip()
                    n = e.extent.end.offset - e.extent.start.offset
                    self.rw.insert(e.extent.start.offset, f"__arwen_cvt<{tname}>(", order=-n - 0.25)
                    self.rw.insert(e.extent.end.offset, ")", order=n + 0.25)
                continue
            if c.kind == K.VAR_DECL and not self._program_scope(c):
                t = c.type
                if _arith_kind(t) is None or self._dependent(t):
                    continue
                toks = [x.spelling for x in _tokens(c)]
                kname = toks.index(c.spelling) if c.spelling in toks else -1
                ex = [x for x in c.get_children() if x.kind.is_expression()]
                if ex and kname >= 0 and "=" in toks[kname + 1:kname + 2] and self._dependent(ex[-1].type) \
                        and self._resolved_arith(ex[-1]) != _arith_kind(t):
                    wrap(ex[-1], t)
            elif c.kind in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
                kids = list(c.get_children())
                if len(kids) != 2:
                    continue
                lhs, rhs = kids
                lt = lhs.type
                if _arith_kind(lt) is None or self._dependent(lt) or not self._dependent(rhs.type) \
                        or self._resolved_arith(rhs) == _arith_kind(lt):
                    continue
                op = [x.spelling for x in _tokens(c) if lhs.extent.end.offset <= x.extent.start.offset
                      < rhs.extent.start.offset]
                if op == ["="] or (op and op[0] in ("+=", "-=", "*=", "/=") and _arith_kind(lt) == "f64"):
                    wrap(rhs, lt)

    def edit_expressions(self):
        for c in self.nodes:
            k = c.kind
            if k == K.CALL_EXPR and c.referenced is None and self._edit_dependent_construct(c):
                continue
            if k == K.DECL_REF_EXPR:
                ref = c.referenced
                if ref is None:
                    continue
                if c.spelling in BUILTIN_VARS and not self._main(ref):
                    self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                    f"__arwen_ctx.{BUILTIN_VARS[c.spelling]}")
                elif ref.hash in self.ssmem_globals:
                    elem, dims = _array_dims(ref.type)
                    ety = self.msl_type(elem, space="device")
                    off = self.ssmem_globals[ref.hash]
                    d = "".join(f"[{x}]" for x in dims)
                    # a pointer object stored in threadgroup memory: ``device T* threadgroup``
                    tg = f"{ety} threadgroup" if _canon(elem).kind == T.POINTER else f"threadgroup {ety}"
                    expr = (f"(*({tg} (*){d})(__arwen_ctx.ssmem + {off}))" if dims
                            else f"(*({tg}*)(__arwen_ctx.ssmem + {off}))")
                    self.rw.replace(c.extent.start.offset, c.extent.end.offset, expr)
                elif ref.kind == K.VAR_DECL and self._main(ref) and self._is_mutable_global(ref):
                    g = self.globals_mut[ref.get_usr()]
                    dims = "".join(f"[{d}]" for d in getattr(g, "dims", []))
                    expr = (f"(*(device {g.msl_type}*)(__arwen_ctx.g + {g.offset}))" if not dims
                            else f"(*(device {g.msl_type} (*){dims})(__arwen_ctx.g + {g.offset}))")
                    self.rw.replace(c.extent.start.offset, c.extent.end.offset, expr)
                elif ref.kind == K.PARM_DECL:
                    owner = self.parent.get(ref.hash)
                    if owner is not None:
                        refs = self.ref_params.get(self.fusr(owner), {})
                        idx = [p.hash for p in self._params(owner)].index(ref.hash) if refs else -1
                        if refs.get(idx) == "ptr":
                            self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"(*{c.spelling})")
            elif k == K.CALL_EXPR:
                self.edit_call(c)
            elif k == K.VAR_DECL:
                self.edit_local_var(c)
            elif k == K.UNEXPOSED_EXPR:
                if c.hash not in self._cast_done:
                    self.edit_implicit_cast(c)
            elif k == K.UNARY_OPERATOR and _is_double(c.type) and c.hash not in self._literal_done \
                    and _literal_value(c) is not None and not self.rw.in_opaque(c.extent.start.offset, c.extent.end.offset):
                # ``-1.5`` in binary64: one constant (a constant expression at program scope,
                # where __f64's operator- would need a global constructor)
                self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                _f64_lit(_literal_value(c), self._in_storage_init(c)), opaque=True)
                for d in c.walk_preorder():
                    self._literal_done.add(d.hash)
            elif k == K.FLOATING_LITERAL:
                self.edit_float_literal(c)
            elif k == K.ASM_STMT:
                self.edit_asm(c)
            elif k in (K.CSTYLE_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR, K.CXX_STATIC_CAST_EXPR,
                       K.CXX_REINTERPRET_CAST_EXPR, K.CXX_CONST_CAST_EXPR):
                self.edit_explicit_cast(c)

    def edit_select_minmax(self):
        """``a > b ? a : b`` on floats -> fmaxf(a, b) (and the three sibling forms).

        NVRTC emits setp + selp for these, but ptxas selects them into FMNMX, and
        CUDA results follow FMNMX's semantics (max(+0, -0) = +0, NaN loses);
        ArWen's RRTMG legacy prep preflight checks exactly that on the device.
        Only side-effect-free operands that are textually identical are matched."""
        simple = re.compile(r"^[\w.\[\]\s()>-]+$")
        for c in self.nodes:
            if c.kind != K.CONDITIONAL_OPERATOR or self.fn_of.get(c.hash) is None or _canon(c.type).kind != T.FLOAT:
                continue
            kids = list(c.get_children())
            if len(kids) != 3:
                continue
            cond = kids[0]
            while cond.kind in (K.PAREN_EXPR, K.UNEXPOSED_EXPR) and len(list(cond.get_children())) == 1:
                cond = list(cond.get_children())[0]
            if cond.kind != K.BINARY_OPERATOR:
                continue
            ck = list(cond.get_children())
            if len(ck) != 2 or _canon(ck[0].type).kind != T.FLOAT or _canon(ck[1].type).kind != T.FLOAT:
                continue
            ops = [t.spelling for t in _tokens(cond) if ck[0].extent.end.offset <= t.extent.start.offset < ck[1].extent.start.offset]
            if ops not in (["<"], [">"]):
                continue
            txt = lambda n: re.sub(r"\s+", "", self.text[n.extent.start.offset:n.extent.end.offset])
            a, b, t, f = txt(ck[0]), txt(ck[1]), txt(kids[1]), txt(kids[2])
            if not (simple.match(a) and simple.match(b)) or "++" in a + b or "--" in a + b or "=" in a + b:
                continue
            if (t, f) == (a, b):
                fn = "fmaxf" if ops == [">"] else "fminf"
            elif (t, f) == (b, a):
                fn = "fminf" if ops == [">"] else "fmaxf"
            else:
                continue
            if self.rw.in_opaque(c.extent.start.offset, c.extent.end.offset):
                continue
            # three disjoint replacements around the operands, so edits inside them still apply
            self.rw.replace(c.extent.start.offset, ck[0].extent.start.offset, f"{fn}(")
            self.rw.replace(ck[0].extent.end.offset, ck[1].extent.start.offset, ", ")
            # the dropped "? x : y" tail repeats the operands: nothing inside it survives
            self.rw.replace(ck[1].extent.end.offset, c.extent.end.offset, ")", opaque=True)
            self._select_done.update(n.hash for n in (cond,))

    _FBIN = {"+": "__arwen_fadd", "-": "__arwen_fsubtract", "*": "__arwen_fmul", "/": "__arwen_fdiv",
             "<": "__arwen_flt", "<=": "__arwen_fle", ">": "__arwen_fgt", ">=": "__arwen_fge",
             "==": "__arwen_feq", "!=": "__arwen_fne"}
    _FCOMP = {"+=": "__arwen_fiadd", "-=": "__arwen_fisub", "*=": "__arwen_fimul", "/=": "__arwen_fidiv"}

    def edit_float_ops(self):
        """CUDA --ftz=false: route float arithmetic and comparisons through the
        gradual-underflow-preserving helpers (Apple GPUs flush FP32 subnormals)."""
        def is_f32(t):
            return _canon(t).kind == T.FLOAT
        for c in self.nodes:
            if self.fn_of.get(c.hash) is None:
                continue
            k = c.kind
            if k in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR):
                kids = list(c.get_children())
                if len(kids) != 2:
                    continue
                lhs, rhs = kids
                ops = [t for t in _tokens(c) if lhs.extent.end.offset <= t.extent.start.offset < rhs.extent.start.offset]
                if len(ops) != 1:
                    continue
                op = ops[0]
                if c.hash in self._select_done:
                    continue
                if k == K.BINARY_OPERATOR and op.spelling in self._FBIN and is_f32(lhs.type) and is_f32(rhs.type):
                    fn = self._FBIN[op.spelling]
                elif k == K.COMPOUND_ASSIGNMENT_OPERATOR and op.spelling in self._FCOMP and is_f32(lhs.type) \
                        and is_f32(rhs.type):
                    fn = self._FCOMP[op.spelling]
                else:
                    continue
                s0, s1 = lhs.extent.start.offset, rhs.extent.end.offset
                if s0 == s1 or self.rw.in_opaque(s0, s1):
                    continue
                w = s1 - s0
                if k == K.COMPOUND_ASSIGNMENT_OPERATOR:
                    self.rw.insert(s0, f"{fn}(&(", order=-w)
                    self.rw.replace(op.extent.start.offset, op.extent.end.offset, "), ")
                else:
                    self.rw.insert(s0, f"{fn}(", order=-w)
                    self.rw.replace(op.extent.start.offset, op.extent.end.offset, ", ")
                self.rw.insert(s1, ")", order=w)
            elif k == K.UNARY_OPERATOR and is_f32(c.type):
                toks = _tokens(c)
                kids = list(c.get_children())
                if toks and toks[0].spelling == "-" and kids and toks[0].extent.end.offset <= kids[0].extent.start.offset \
                        and not self.rw.in_opaque(c.extent.start.offset, c.extent.end.offset) \
                        and _literal_value(kids[0]) is None:
                    # (a negated literal is folded exactly by the compiler, and stays a
                    # constant expression)
                    self.rw.replace(toks[0].extent.start.offset, toks[0].extent.end.offset, "__arwen_fneg(")
                    self.rw.insert(c.extent.end.offset, ")", order=c.extent.end.offset - c.extent.start.offset)

    def edit_explicit_cast(self, c):
        if c.kind == K.CXX_CONST_CAST_EXPR and self._dependent(c.type):
            # const_cast<P>(x) in a template changes no value: a template-parameter P is
            # spelled as is; a pointer ``T*`` (pointee dependent) gets device memory, the
            # space of the pointer members it initialises
            toks = _tokens(c)
            lt = next((t for t in toks if t.spelling == "<"), None)
            if lt is not None and _canon(c.type).kind == T.POINTER:
                depth, gt = 0, None
                for t in toks[toks.index(lt):]:
                    depth += {"<": 1, ">": -1}.get(t.spelling, 0)
                    if depth == 0:
                        gt = t
                        break
                if gt is not None:
                    self.rw.replace(lt.extent.end.offset, gt.extent.start.offset,
                                    self.msl_type(c.type, space="device"))
            return
        ops = [ch for ch in c.get_children() if ch.kind.is_expression()]
        if not ops:
            return
        op = ops[-1]
        s0, s1 = op.extent.start.offset, op.extent.end.offset
        to_c, fr_c = _canon(c.type), _canon(op.type)
        if to_c.kind == T.POINTER and self._is_null(op):
            p = self.parent.get(c.hash)
            while p is not None and p.kind in (K.PAREN_EXPR, K.UNEXPOSED_EXPR):
                p = self.parent.get(p.hash)
            if p is not None and p.kind == K.CONDITIONAL_OPERATOR:
                # ``c ? &local : (T*)0``: the null converts to the other operand's space
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, "nullptr", opaque=True)
                return
            if p is not None and p.kind == K.CALL_EXPR and self.callee(p)[0] is not None:
                return                      # a null argument: edit_call picks the space
            self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                            f"(({self.msl_type(c.type, space='device')})nullptr)", opaque=True)
            return
        if to_c.kind == T.POINTER and _arith_kind(op.type) == "int":
            # a GPU address carried as an integer (pointer tables): device memory
            self.rw.replace(c.extent.start.offset, s0, f"(({self.msl_type(c.type, space='device')})(")
            if s1 < c.extent.end.offset:
                self.rw.replace(s1, c.extent.end.offset, "))")
            else:
                self.rw.insert(s1, "))", order=1e9)
            return
        if to_c.kind == T.POINTER and fr_c.kind in (T.POINTER, T.CONSTANTARRAY, T.INCOMPLETEARRAY):
            pt = c.type.get_pointee()
            if _canon(pt).kind == T.CONSTANTARRAY:          # to a pointer to array: T = elem[N]...
                elem, dims = _array_dims(pt)
                pointee = self.msl_type(elem, storage=True) + "".join(f"[{d}]" for d in dims)
            else:
                pointee = self.msl_type(pt, storage=True)
            self.rw.replace(c.extent.start.offset, s0, f"__arwen_pcast<{pointee}>(")
            self.rw.replace(s1, c.extent.end.offset, ")") if s1 < c.extent.end.offset else self.rw.insert(s1, ")")
            return
        to = _arith_kind(c.type)
        src = op
        # libclang exposes the operand as an implicit node already carrying the
        # target type; the conversion's real source is further down the chain.
        while src.kind == K.UNEXPOSED_EXPR and _arith_kind(src.type) == to:
            kids = list(src.get_children())
            if len(kids) != 1:
                break
            src = kids[0]
        frm = _arith_kind(src.type)
        # type-dependent operand or target inside a template: the instantiation may
        # be double or not, and __arwen_cvt is right for every arithmetic pair
        if to is None and self._dependent(c.type) and to_c.kind != T.POINTER and (frm is not None or self._dependent(src.type)):
            to = "dependent"
            frm = frm or "dependent-source"
        elif frm is None and to is not None and self._dependent(src.type):
            frm = "dependent-source"
        if to is None or frm is None or to == frm or ("f64" not in (to, frm) and "dependent" != to and "dependent-source" != frm):
            return
        v = _literal_value(src)
        target = self.msl_type(c.type).replace("const ", "")
        if to == "dependent":
            v = None
        if v is not None and to in ("f64", "f32"):
            lit = _f64_lit(v, self._in_storage_init(c)) if to == "f64" else _hexf32(v)
            self.rw.replace(c.extent.start.offset, c.extent.end.offset, lit, opaque=True)
            for d in c.walk_preorder():
                self._literal_done.add(d.hash)
            return
        self.rw.replace(c.extent.start.offset, s0, f"__arwen_cvt<{target}>(")
        if s1 < c.extent.end.offset:
            self.rw.replace(s1, c.extent.end.offset, ")")
        else:
            # closes after the operand's own wrappers, before those of enclosing spans
            self.rw.insert(s1, ")", order=(s1 - c.extent.start.offset) + 0.1)
        self._cast_done.add(op.hash)

    def _has_resolved_calls(self, usr) -> bool:
        if not hasattr(self, "_resolved_callees"):
            self._resolved_callees = set()
            for c in self.nodes:
                if c.kind == K.CALL_EXPR:
                    u, _ = self.callee(c)
                    if u is not None:
                        self._resolved_callees.add(u)
                        # a call resolved by name stands for every same-named candidate
                        name = self.functions[u][0].spelling if u in self.functions else None
                        self._resolved_callees.update(self.fn_names.get(name, ()))
        return usr in self._resolved_callees

    def _callee_args(self, usr, i):
        if not hasattr(self, "_args_by_callee"):
            self._args_by_callee = defaultdict(list)
            for c in self.nodes:
                if c.kind == K.CALL_EXPR:
                    u, _ = self.callee(c)
                    if u is not None:
                        for j, x in enumerate(self.call_args(c)):
                            self._args_by_callee[(u, j)].append(x)
        return self._args_by_callee.get((usr, i), [])

    def _array_arg_space(self, usr, i):
        """thread / constant when every argument for parameter i is a local / a
        __constant__ (or const program-scope) array, else None."""
        spaces = set()
        args = self._callee_args(usr, i)
        for x in args:
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            if x.kind == K.INIT_LIST_EXPR:
                spaces.add("thread")            # a braced temporary bound to a const reference
                continue
            r = x.referenced if x.kind == K.DECL_REF_EXPR else None
            if r is None or r.kind != K.VAR_DECL or _canon(r.type).kind != T.CONSTANTARRAY:
                return None
            spaces.add("thread" if not self._program_scope(r) else
                       "constant" if not self._is_mutable_global(r) else "device")
        return spaces.pop() if len(spaces) == 1 and "device" not in spaces else None

    def _view_rank_conversion(self, a, ptype):
        """Rank of the view parameter when argument ``a`` reaches it through the
        view's own conversion operator (``V<T,2>`` passed for ``V<T,1>``), else None."""
        x = a
        while x.kind == K.UNEXPOSED_EXPR and len([k for k in x.get_children() if k.kind.is_expression()]) == 1:
            ch = [k for k in x.get_children() if k.kind.is_expression()][0]
            if ch.kind == K.CALL_EXPR and ch.referenced is not None and ch.referenced.kind == K.CONVERSION_FUNCTION:
                m = re.match(r"\s*(?:const\s+)?\w+\s*<\s*[^,<>]+,\s*(\d+)\s*>", ptype.spelling)
                return m.group(1) if m else None
            x = ch
        return None

    def _is_null_view_arg(self, a) -> bool:
        """A null literal converted to a view by its constructor (``V<T>(nullptr)`` implicitly)."""
        def written(k):     # default arguments have an empty extent
            return k.kind.is_expression() and k.extent.end.offset > k.extent.start.offset
        x = a
        while x.kind in (K.UNEXPOSED_EXPR, K.CALL_EXPR) and len([k for k in x.get_children() if written(k)]) == 1:
            if x.kind == K.CALL_EXPR and (x.referenced is None or x.referenced.kind != K.CONSTRUCTOR):
                return False
            x = [k for k in x.get_children() if written(k)][0]
        return self._is_null(x)

    def _null_arg_space(self, usr, i) -> str:
        """Address space for a null pointer argument: thread when every non-null
        argument at that position, across the module's calls, is the address of a
        local object (one instantiation then serves all calls), else device."""
        if not hasattr(self, "_args_by_callee"):
            self._args_by_callee = defaultdict(list)
            for c in self.nodes:
                if c.kind == K.CALL_EXPR:
                    u, _ = self.callee(c)
                    if u is not None:
                        for j, x in enumerate(self.call_args(c)):
                            self._args_by_callee[(u, j)].append(x)
            self._thread_refs = self._thread_ref_params()

        def thread_addr(x):
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            if x.kind == K.UNARY_OPERATOR and _tokens(x) and _tokens(x)[0].spelling == "&":
                ch = list(x.get_children())
                r = ch[0].referenced if ch and ch[0].kind == K.DECL_REF_EXPR else None
                return r is not None and ((r.kind == K.VAR_DECL and not self._program_scope(r)
                                           and _canon(r.type).kind != T.LVALUEREFERENCE)
                                          or (r.kind == K.PARM_DECL and r.hash in self._thread_refs))
            r = x.referenced if x.kind == K.DECL_REF_EXPR else None
            return r is not None and r.kind == K.VAR_DECL and not self._program_scope(r) \
                and _canon(r.type).kind == T.CONSTANTARRAY
        others = [x for x in self._args_by_callee.get((usr, i), []) if not self._is_null(x)]
        return "thread" if others and all(thread_addr(x) for x in others) else "device"

    def edit_call(self, c):
        kids = list(c.get_children())
        if kids and kids[0].kind == K.MEMBER_REF_EXPR:
            toks = _tokens(kids[0])
            for a, b, d in zip(toks, toks[1:], toks[2:]):
                if a.spelling in (".", "->") and b.spelling == c.spelling and d.spelling == "<":
                    self.rw.insert(a.extent.end.offset, "template ")
                    break
        ref = c.referenced
        if ref is not None and c.spelling in WARP_FUNCS and not self._main(ref):
            lp = self._call_lparen(c)
            if lp is None:
                raise TranslationError(f"call to {c.spelling}: argument list not found")
            nargs = len([k for k in c.get_children() if k.kind.is_expression()][1:])
            self.rw.insert(lp, "__arwen_ctx, " if nargs else "__arwen_ctx", order=-100)
            return
        usr, ref = self.callee(c)
        if usr is None:
            return
        if usr in self.needs_ctx:
            lp = self._call_lparen(c)
            if lp is None and ref is not None and ref.kind == K.CONSTRUCTOR:
                # ``T x;`` default-constructed by a constructor that needs the ctx
                v = self.parent.get(c.hash)
                while v is not None and v.kind == K.UNEXPOSED_EXPR:
                    v = self.parent.get(v.hash)
                if v is not None and v.kind == K.VAR_DECL:
                    nt = next((t for t in _tokens(v) if t.spelling == v.spelling), None)
                    if nt is not None and not any(t.spelling in ("=", "(", "{") for t in _tokens(v)
                                                  if t.extent.start.offset > nt.extent.start.offset):
                        self.rw.insert(nt.extent.end.offset, "(__arwen_ctx)")
                        return
            if lp is None:
                raise TranslationError(f"call to {c.spelling}: argument list not found")
            nargs = len(list(self.call_args(c)))
            # before any wrapper the first argument gets (``&(`` for a reference parameter)
            self.rw.insert(lp, "__arwen_ctx" + (", " if nargs else ""), order=-100)
        params = self._params(ref)
        for i, a in enumerate(self.call_args(c)):
            if i < len(params) and _canon(params[i].type).kind == T.RECORD and self._record_has_pointers(params[i].type):
                conv = self._view_rank_conversion(a, params[i].type)
                if conv is not None:
                    n_ = a.extent.end.offset - a.extent.start.offset
                    self.rw.insert(a.extent.start.offset, "(", order=-n_ - 0.75)
                    self.rw.insert(a.extent.end.offset, f").template __arwen_rank<{conv}>()", order=n_ + 0.75)
                    continue
            if i < len(params) and self._is_null_view_arg(a) and _canon(params[i].type).kind == T.RECORD \
                    and self._record_has_pointers(params[i].type):
                # a null view argument (``f(..., nullptr)`` with ``V<T> v`` built implicitly):
                # the parameter is generic, so construct the view explicitly
                ty = self.msl_type(params[i].type).replace("const ", "")
                self.rw.replace(a.extent.start.offset, a.extent.end.offset, f"{ty}(nullptr)", opaque=True)
                continue
            if i < len(params) and _canon(params[i].type).kind == T.POINTER and self._is_null(a):
                space = self._null_arg_space(usr, i)
                if self._dependent(params[i].type) or "__ARWEN_AP" in params[i].type.spelling:
                    self.rw.replace(a.extent.start.offset, a.extent.end.offset, "nullptr", opaque=True)
                    continue
                self.rw.replace(a.extent.start.offset, a.extent.end.offset,
                                f"(({self.msl_type(params[i].type, space=space)})nullptr)", opaque=True)
        refs = self.ref_params.get(usr)
        if refs:
            for i, a in enumerate(self.call_args(c)):
                if refs.get(i) == "ptr":
                    self.rw.insert(a.extent.start.offset, "&(", order=-5)
                    self.rw.insert(a.extent.end.offset, ")", order=5)
        args = list(self.call_args(c))
        if (ref is not None and ref.kind == K.CONSTRUCTOR and len(args) == 1
                and c.extent.start.offset == args[0].extent.start.offset
                and c.extent.end.offset == args[0].extent.end.offset
                and _canon(args[0].type).spelling.replace("const ", "") != _canon(c.type).spelling.replace("const ", "")):
            # an implicit converting construction (an argument bound to a by-value
            # record parameter): spell it out, since that parameter may have become
            # a deduced template parameter and would otherwise take the source type
            rec = re.sub(r"\bdouble\b", "__f64", self.msl_type(c.type).replace("const ", "").strip())
            self.rw.insert(c.extent.start.offset, f"{rec}(", order=-6)
            self.rw.insert(c.extent.end.offset, ")", order=6)

    def edit_local_var(self, c):
        if self._program_scope(c):
            return
        attrs = self._attrs(c)
        toks = _tokens(c)
        par = self.parent.get(c.hash)
        if par is not None and par.kind == K.DECL_STMT:
            st = next((t for t in _tokens(par) if t.spelling == "static"
                       and t.extent.start.offset < c.location.offset), None)
            if st is not None and par.extent.start.offset == c.extent.start.offset:
                spell = [t.spelling for t in _tokens(par)]
                if "constexpr" in spell or (_deep_const(c.type) and "const" in spell):
                    # a function-scope constant: the same value without static storage
                    self.tok_replace(st, "")
                else:
                    raise TranslationError(f"static local {c.spelling}: state kept across calls")
        if K.CUDASHARED_ATTR in attrs:
            fn = self.fn_of.get(c.hash)
            if c.hash in self.dep_shared_decl:
                ety, d, off = self.dep_shared_decl[c.hash]
                name = _ident(c.spelling)
                self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                f"threadgroup {ety} (&{name}){d} = *(threadgroup {ety} (*){d})(__arwen_ctx.ssmem + {off})")
                return
            if c.hash in self.ssmem_vars:
                elem, dims = _array_dims(c.type)
                ety = self.msl_type(elem)
                off = self.ssmem_vars[c.hash]
                name = _ident(c.spelling)
                if dims:
                    d = "".join(f"[{x}]" for x in dims)
                    new = f"threadgroup {ety} (&{name}){d} = *(threadgroup {ety} (*){d})(__arwen_ctx.ssmem + {off})"
                else:
                    new = f"threadgroup {ety}& {name} = *(threadgroup {ety}*)(__arwen_ctx.ssmem + {off})"
                # ``__shared__ float a[3], b[4];``: each declarator its own declaration
                start = c.extent.start.offset
                par = self.parent.get(c.hash)
                if par is not None and par.kind == K.DECL_STMT:
                    decls = [ch for ch in par.get_children() if ch.kind == K.VAR_DECL]
                    if len(decls) > 1 and decls[0].hash != c.hash:
                        name_tok = next(t for t in _tokens(c) if t.spelling == c.spelling
                                        and t.extent.start.offset >= c.location.offset)
                        start = max((t for t in _tokens(par) if t.spelling == ","
                                     and t.extent.start.offset < name_tok.extent.start.offset),
                                    key=lambda t: t.extent.start.offset).extent.start.offset
                        new = "; " + new
                self.rw.replace(start, c.extent.end.offset, new)
                return
            if fn is not None and self._is_extern_shared(c):
                # dynamic shared memory, in a kernel or a device function (which gets the ctx)
                elem = c.type.get_array_element_type() if _canon(c.type).kind in (T.INCOMPLETEARRAY, T.CONSTANTARRAY) else c.type
                ety = self.msl_type(elem)
                start = c.extent.start.offset
                prev = self.text.rfind("extern", 0, start)
                if prev >= 0 and self.text[prev:start].strip() == "extern":
                    start = prev
                self.rw.replace(start, c.extent.end.offset,
                                f"threadgroup {ety}* {c.spelling} = reinterpret_cast<threadgroup {ety}*>(__arwen_ctx.dsmem)")
                return
            if fn is None or not self.is_kernel(fn):
                raise TranslationError(f"__shared__ variable {c.spelling} outside a kernel body")
            for t in toks:
                if t.spelling == "__shared__":
                    self.tok_replace(t, "threadgroup")
            return
        if K.CUDADEVICE_ATTR in attrs or K.CUDACONSTANT_ATTR in attrs:
            raise TranslationError(f"function-scope __device__/__constant__ variable {c.spelling}")
        par = self.parent.get(c.hash)
        if par is not None and par.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE,
                                            K.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION, K.UNION_DECL):
            # static data member: Metal keeps such objects in the constant address space
            kw = next((t for t in toks if t.spelling == "constexpr"), None) or next((t for t in toks if t.spelling == "static"), None)
            if kw is None:
                raise TranslationError(f"static member {c.spelling}: no storage specifier")
            self.rw.insert(kw.extent.end.offset, " constant")
            return
        ct = _canon(c.type)
        if (ct.kind == T.CONSTANTARRAY and _canon(_array_dims(c.type)[0]).kind == T.POINTER) or (
                c.type.kind == T.DEPENDENTSIZEDARRAY and _canon(c.type.get_array_element_type()).kind == T.POINTER):
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            elem = _array_dims(c.type)[0] if ct.kind == T.CONSTANTARRAY else c.type.get_array_element_type()
            if name_tok is not None:
                ref = self._ref_param_table(c)
                ty = f"metal::remove_cv_t<decltype({ref})>" if ref else self.msl_type(elem, space=self._pointer_table_space(c))
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, ty + " ")
            return
        if ct.kind == T.POINTER:
            init = [ch for ch in c.get_children() if ch.kind.is_expression()]
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None:
                return
            p = self.parent.get(c.hash)
            if p is not None and p.kind == K.DECL_STMT:
                decls = [ch for ch in p.get_children() if ch.kind == K.VAR_DECL]
                if len(decls) > 1:
                    # once per statement: at its first pointer declarator
                    first_ptr = next(d for d in decls if _canon(d.type).kind == T.POINTER)
                    if first_ptr.hash == c.hash:
                        self._split_declarators(p, decls)
                    return
            eq = next((t for t in toks if t.spelling == "=" and t.extent.start.offset > name_tok.extent.start.offset), None)
            if _canon(c.type.get_pointee()).kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY) and init \
                    and not self._is_null(init[-1]) and eq is not None:
                # pointer to array, ``const float (*t)[N] = ...``: the whole declarator
                self.rw.replace(c.extent.start.offset, eq.extent.start.offset, f"auto {c.spelling} ")
            elif init and not self._is_null(init[-1]):
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, "auto ")
            elif init:
                rhs = self._first_assignment(c)
                if rhs is None:
                    # never re-pointed: a null pointer of the declared type, in device memory
                    self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                    self.msl_type(c.type, space="device") + " ")
                else:
                    self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                    f"metal::remove_reference_t<decltype({self.text[rhs.extent.start.offset:rhs.extent.end.offset]})> ")
            else:
                # address space from the first assignment to this variable
                rhs = self._first_assignment(c)
                if rhs is None:
                    raise TranslationError(f"pointer {c.spelling} declared without initialiser or assignment")
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                f"metal::remove_reference_t<decltype({self.text[rhs.extent.start.offset:rhs.extent.end.offset]})> ")
        elif ct.kind == T.LVALUEREFERENCE:
            name_tok = next((t for t in toks if t.spelling == c.spelling
                             and t.extent.start.offset >= c.location.offset), None)
            if name_tok is None:
                return
            # ``T &a = x, &b = y;``: from the second declarator on, the prefix to replace
            # is ``, &`` and the declarations are separated with ``;``
            start, sep = c.extent.start.offset, ""
            p = self.parent.get(c.hash)
            if p is not None and p.kind == K.DECL_STMT:
                decls = [ch for ch in p.get_children() if ch.kind == K.VAR_DECL]
                if len(decls) > 1 and decls[0].hash != c.hash:
                    if any(_canon(d.type).kind != T.LVALUEREFERENCE for d in decls):
                        raise TranslationError(f"reference {c.spelling} declared together with non-references")
                    comma = max((t for t in _tokens(p) if t.spelling == "," and t.extent.start.offset
                                 < name_tok.extent.start.offset), key=lambda t: t.extent.start.offset)
                    start, sep = comma.extent.start.offset, "; "
            pt = c.type.get_pointee()
            init = [k for k in c.get_children() if k.kind.is_expression()]
            if _canon(pt).kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY) and init:
                # a reference to an array (``const auto &t = table;``): the reference type,
                # address space included, is that of the initialiser lvalue
                it = self.text[init[-1].extent.start.offset:init[-1].extent.end.offset]
                self.rw.replace(start, name_tok.extent.start.offset, f"{sep}decltype(({it})) ")
            elif pt.is_const_qualified():
                self.rw.replace(start, name_tok.extent.start.offset,
                                sep + self.msl_type(pt).replace("const ", "") + " ")
            else:
                self.rw.replace(start, name_tok.extent.start.offset, f"{sep}decltype(auto) ")

    def _pointer_decl_type(self, c) -> str:
        """The MSL type spelling that replaces a local pointer's declared type."""
        init = [ch for ch in c.get_children() if ch.kind.is_expression()]
        if init and not self._is_null(init[-1]):
            return "auto"
        rhs = self._first_assignment(c)
        if rhs is None:
            if init:
                return self.msl_type(c.type, space="device")
            raise TranslationError(f"pointer {c.spelling} declared without initialiser or assignment")
        return f"metal::remove_reference_t<decltype({self.text[rhs.extent.start.offset:rhs.extent.end.offset]})>"

    def _ref_param_table(self, c):
        """``T* v[] = {&r1, &r2, ...}`` over reference parameters (pointers after
        translation, all of one instantiation type): the first one's name."""
        lists = [x for x in c.get_children() if x.kind == K.INIT_LIST_EXPR]
        if not lists:
            return None
        names = []
        for e in lists[0].get_children():
            x = e
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            if not (x.kind == K.UNARY_OPERATOR and _tokens(x) and _tokens(x)[0].spelling == "&"):
                return None
            ch = list(x.get_children())
            r = ch[0].referenced if ch and ch[0].kind == K.DECL_REF_EXPR else None
            if r is None or r.kind != K.PARM_DECL or _canon(r.type).kind != T.LVALUEREFERENCE:
                return None
            names.append((r, _ident(r.spelling)))
        fns = {self.fn_of.get(r.hash).hash if self.fn_of.get(r.hash) is not None else None for r, _ in names}
        types = {r.type.spelling for r, _ in names}
        return names[0][1] if names and len(fns) == 1 and len(types) == 1 else None

    def _pointer_table_space(self, c) -> str:
        """Address space of a local pointer table ``T* v[N] = {a, b, ...}``: thread when
        every element is a local array, constant when every one is a __constant__
        array, else device."""
        lists = [x for x in c.get_children() if x.kind == K.INIT_LIST_EXPR]
        if not lists:
            return "device"
        spaces = set()
        for e in lists[0].get_children():
            x = e
            while x.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(x.get_children())) == 1:
                x = list(x.get_children())[0]
            if x.kind == K.UNARY_OPERATOR and _tokens(x) and _tokens(x)[0].spelling == "&":
                sp = self._expr_space(x)            # ``{&a, &b}`` over objects
                if sp is None:
                    return "device"
                spaces.add(sp)
                continue
            r = x.referenced if x.kind == K.DECL_REF_EXPR else None
            if r is None or r.kind != K.VAR_DECL or _canon(r.type).kind != T.CONSTANTARRAY:
                return "device"
            if not self._program_scope(r):
                spaces.add("thread")
            elif not self._is_mutable_global(r):
                spaces.add("constant")
            else:
                return "device"
        return spaces.pop() if len(spaces) == 1 else "device"

    def _split_declarators(self, stmt, decls):
        """``T *a, *b = x, c;`` -> one declaration per declarator, each with its own MSL
        type (a pointer's address space comes from its own initialiser or first
        assignment).  Only the type prefix and the ``, *`` between declarators are
        rewritten, so edits inside initialisers do not overlap."""
        stoks = _tokens(stmt)
        for k, d in enumerate(decls):
            name_tok = next((t for t in _tokens(d) if t.spelling == d.spelling), None)
            if name_tok is None:
                raise TranslationError(f"declarator {d.spelling}: name not found")
            dct = _canon(d.type)
            if dct.kind == T.POINTER:
                ty = self._pointer_decl_type(d)
            elif dct.kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY):
                ty = self.msl_type(_array_dims(d.type)[0])
            else:
                ty = self.msl_type(d.type)
            if k == 0:
                self.rw.replace(d.extent.start.offset, name_tok.extent.start.offset, ty + " ")
                continue
            comma = max((t for t in stoks if t.spelling == "," and t.extent.start.offset < name_tok.extent.start.offset),
                        key=lambda t: t.extent.start.offset)
            self.rw.replace(comma.extent.start.offset, name_tok.extent.start.offset, f"; {ty} ")

    @staticmethod
    def _is_null(e) -> bool:
        while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR):
            kids = [k for k in e.get_children() if k.kind.is_expression()]
            if len(kids) != 1:
                return False
            e = kids[0]
        if e.kind in (K.CXX_NULL_PTR_LITERAL_EXPR, K.GNU_NULL_EXPR):
            return True
        return e.kind == K.INTEGER_LITERAL and _literal_value(e) == 0

    def _first_assignment(self, var):
        for c in self.nodes:
            if c.kind == K.BINARY_OPERATOR and c.extent.start.offset > var.extent.end.offset:
                ch = list(c.get_children())
                if len(ch) == 2 and ch[0].kind == K.DECL_REF_EXPR and ch[0].referenced is not None \
                        and ch[0].referenced.hash == var.hash:
                    toks = _tokens(c)
                    ops = [t.spelling for t in toks if t.extent.start.offset == ch[0].extent.end.offset
                           or (ch[0].extent.end.offset <= t.extent.start.offset < ch[1].extent.start.offset)]
                    if "=" in ops:
                        return ch[1]
        return None

    def _in_storage_init(self, c) -> bool:
        p = self.parent.get(c.hash)
        while p is not None:
            if p.kind == K.VAR_DECL:
                return self._program_scope(p)
            if p.kind in FN_KINDS:
                return False
            p = self.parent.get(p.hash)
        return False

    def edit_implicit_cast(self, c):
        ch = list(c.get_children())
        if len(ch) != 1:
            return
        if self.rw.in_opaque(c.extent.start.offset, c.extent.end.offset) and \
                (c.extent.start.offset, c.extent.end.offset) not in self.rw.opaque:
            return                       # inside an already folded constant
        p = self.parent.get(c.hash)
        if p is not None and p.kind in (K.CSTYLE_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR, K.CXX_STATIC_CAST_EXPR):
            return
        child = ch[0]
        to, frm = _arith_kind(c.type), _arith_kind(child.type)
        if to is None or frm is None or to == frm or "f64" not in (to, frm):
            return
        if child.kind == K.UNEXPOSED_EXPR and _arith_kind(child.type) == frm and \
                len(list(child.get_children())) == 1 and _arith_kind(list(child.get_children())[0].type) == frm:
            pass
        v = _literal_value(child)
        s, e = child.extent.start.offset, child.extent.end.offset
        if s == e:
            return
        if v is not None:
            # the whole (possibly negated / parenthesised) literal becomes one constant:
            # later passes must not rewrite inside it (e.g. the FP32 negation helper)
            if to == "f64":
                self.rw.replace(s, e, _f64_lit(v, self._in_storage_init(c)), opaque=True)
            elif to == "f32":
                self.rw.replace(s, e, _hexf32(v), opaque=True)
            else:
                self.rw.insert(s, f"__arwen_cvt<{self.msl_type(c.type).replace('const ', '')}>(", order=-c.extent.end.offset)
                self.rw.insert(e, ")", order=c.extent.start.offset)
            self._literal_done.add(child.hash)
            for d in child.walk_preorder():
                self._literal_done.add(d.hash)
            return
        target = self.msl_type(c.type).replace("const ", "")
        if to == "f64" and self._in_storage_init(c):
            target = "__f64m"
        # nested casts at the same offsets: outer opens first and closes last
        width = e - s
        self.rw.insert(s, f"__arwen_cvt<{target}>(", order=-width)
        self.rw.insert(e, ")", order=width)

    def edit_float_literal(self, c):
        if c.hash in self._literal_done:
            return
        if _is_double(c.type):
            v = _literal_value(c)
            self.rw.replace(c.extent.start.offset, c.extent.end.offset, _f64_lit(v, self._in_storage_init(c)))

    _ASM = [
        (re.compile(r'(add|sub|mul|div)\.rn\.f32\s+%0\s*,\s*%1\s*,\s*%2\s*;'), "binop"),
        (re.compile(r'\{\s*\.reg\s+\.pred\s+%?%?\w+\s*;\s*setp\.(lt|le|gt|ge|eq|ne)\.f32\s+%?%?\w+\s*,\s*%1\s*,\s*%2\s*;\s*selp\.u32\s+%0\s*,\s*1\s*,\s*0\s*,\s*%?%?\w+\s*;\s*\}'), "cmp"),
        # the same compare selecting 1.0f / 0.0f (ftz_probe)
        (re.compile(r'\{\s*\.reg\s+\.pred\s+%?%?\w+\s*;\s*setp\.(lt|le|gt|ge|eq|ne)\.f32\s+%?%?\w+\s*,\s*%1\s*,\s*%2\s*;\s*selp\.f32\s+%0\s*,\s*0f3F800000\s*,\s*0f00000000\s*,\s*%?%?\w+\s*;\s*\}'), "cmpf"),
        (re.compile(r'^\s*(min|max)\.f32\s+%0\s*,\s*%1\s*,\s*%2\s*;\s*$'), "minmax"),
        (re.compile(r'^\s*trap\s*;\s*$'), "trap"),
        (re.compile(r'^\s*cvt\.rn\.f32\.f64\s+%0\s*,\s*%1\s*;\s*$'), "cvt_f32_f64"),
        # %globaltimer (device-side stage timing for receipts): MSL exposes no clock
        (re.compile(r'^\s*mov\.u64\s+%0\s*,\s*%%globaltimer\s*;\s*$'), "globaltimer"),
        (re.compile(r'^\s*$'), "empty"),
    ]

    def edit_asm(self, c):
        text = self.text[c.extent.start.offset:c.extent.end.offset]
        m = re.match(r'asm\s*(volatile\s*)?\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)(.*)\)\s*$', text, re.S)
        if not m:
            raise TranslationError(f"unrecognised inline asm: {text[:120]!r}")
        tmpl = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(2)))
        # C escapes in the template (``\n\t`` between instructions) are whitespace to PTX
        tmpl = re.sub(r"\\[ntrv]", " ", tmpl)
        rest = m.group(3)
        sections = [s.strip() for s in self._split_colons(rest)]
        outs = re.findall(r'"=[a-z]"\s*\(([^)]*)\)', sections[0] if sections else "")
        ins = re.findall(r'"[a-z]"\s*\(([^)]*)\)', sections[1] if len(sections) > 1 else "")
        for e in outs + ins:
            # operands are re-emitted as written: only plain lvalues / values, whose
            # spelling means the same in MSL (no arithmetic left untranslated)
            if not re.fullmatch(r"\s*([A-Za-z_][\w.]*(\[[\w+\-* ]*\])?|[-+]?[0-9.]+([eE][-+]?\d+)?[fF]?)\s*", e):
                raise TranslationError(f"inline asm operand {e.strip()!r}: only plain variables are supported")
        for rx, kind in self._ASM:
            mm = rx.search(tmpl)
            if not mm:
                continue
            if kind == "binop" and len(outs) == 1 and len(ins) == 2:
                fn = {"add": "__fadd_rn", "sub": "__fsub_rn", "mul": "__fmul_rn", "div": "__fdiv_rn"}[mm.group(1)]
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"{outs[0]} = {fn}({ins[0]}, {ins[1]})")
                return
            if kind == "cmp" and len(outs) == 1 and len(ins) == 2:
                # setp.<op>.f32 + selp.u32 1/0: ordered IEEE comparison (NaN -> false, ne too)
                self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                f"{outs[0]} = __arwen_fcmp_u32_{mm.group(1)}({ins[0]}, {ins[1]})")
                return
            if kind == "cmpf" and len(outs) == 1 and len(ins) == 2:
                self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                f"{outs[0]} = (float)__arwen_fcmp_u32_{mm.group(1)}({ins[0]}, {ins[1]})")
                return
            if kind == "minmax" and len(outs) == 1 and len(ins) == 2:
                # PTX min/max.f32: the non-NaN operand when one is NaN, as fminf/fmaxf (FMNMX)
                fn = "fminf" if mm.group(1) == "min" else "fmaxf"
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"{outs[0]} = {fn}({ins[0]}, {ins[1]})")
                return
            if kind == "globaltimer" and len(outs) == 1 and not ins:
                # documented difference: GPU-side timing receipts read 0 elapsed on Metal
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"{outs[0]} = 0UL")
                return
            if kind == "cvt_f32_f64" and len(outs) == 1 and len(ins) == 1:
                # round to nearest, subnormal results kept (no .ftz)
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"{outs[0]} = __double2float_rn({ins[0]})")
                return
            if kind == "trap" and not outs and not ins:
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, "__trap(__arwen_ctx)")
                return
            if kind == "empty" and not outs and not ins:
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, "__arwen_compiler_barrier()")
                return
        raise TranslationError(f"unsupported inline PTX: {tmpl!r}")

    @staticmethod
    def _split_colons(s):
        out, cur, depth = [], [], 0
        for ch in s.lstrip(":"):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == ":" and depth == 0:
                out.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        out.append("".join(cur))
        return out

    def edit_tokens(self):
        """Keyword-level rewrites: double, long long, integer literal suffixes, __restrict__."""
        _, main, _ = _file_tokens(self.tu, self.filename)
        i = 0
        while i < len(main):
            t = main[i]
            s = t.spelling
            off = t.extent.start.offset
            if s == "double":
                if off not in self.rw.repl and not self._inside_replaced(off):
                    self.tok_replace(t, "__f64")
            elif s == "long" and i + 1 < len(main) and main[i + 1].spelling == "long":
                if not self._inside_replaced(off):
                    self.rw.replace(off, main[i + 1].extent.end.offset, "long")
                i += 1
            elif t.kind == ci.TokenKind.LITERAL and re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)([uU]?[lL]{2}[uU]?)", s):
                if not self._inside_replaced(off):
                    num = re.match(r"(0[xX][0-9a-fA-F]+|\d+)", s).group(1)
                    self.tok_replace(t, num + ("UL" if "u" in s.lower() else "L"))
            elif s == "__restrict__":
                if not self._inside_replaced(off):
                    self.tok_replace(t, "")
            elif t.kind == ci.TokenKind.IDENTIFIER and (s in MSL_RESERVED or s in CUDA_VECTOR_RENAMES):
                if not self._inside_replaced(off):
                    self.tok_replace(t, CUDA_VECTOR_RENAMES.get(s) or _ident(s))
            i += 1

    def _inside_replaced(self, off):
        if not hasattr(self, "_spans"):
            self._spans = sorted((s, e) for s, (e, _) in self.rw.repl.items())
        import bisect
        spans = sorted((s, e) for s, (e, _) in self.rw.repl.items())
        j = bisect.bisect_right(spans, (off, float("inf"))) - 1
        return j >= 0 and spans[j][0] <= off < spans[j][1]

    def finish(self, body: str) -> Translation:
        init_kernel = ""
        inits = [g for g in self.globals_mut.values() if g.init]
        if inits:
            lines = []
            for g in inits:
                lines.append(f"    *(device {g.msl_type}*)(__arwen_g + {g.offset}) = {g.init};"
                             if not g.init.lstrip().startswith("{") else
                             f"    {{ const {g.msl_type} __v[] = {g.init}; for (uint i = 0; i < sizeof(__v)/sizeof(__v[0]); ++i) ((device {g.msl_type}*)(__arwen_g + {g.offset}))[i] = __v[i]; }}")
            init_kernel = ("\nkernel void __arwen_init_globals(device uchar* __arwen_g [[buffer(2)]])\n{\n"
                           + "\n".join(lines) + "\n}\n")
        flags = metal_flags(self.opts)
        prelude = (f"// generated by gpuwm.metal.transpile ({TRANSLATOR_REVISION}); do not edit\n"
                   f"#define ARWEN_F64_MODE_{self.mode.upper()} 1\n"
                   f"#define ARWEN_FTZ {int(self.opts.ftz)}\n"
                   + (f"#define ARWEN_SYNCRED_OFF {self.syncred_off}\n" if self.syncred_off is not None else "")
                   + ("#define ARWEN_SF_INLINE_ALL 1\n" if os.environ.get("ARWEN_METAL_SF_INLINE") == "1" else "")
                   + "#include \"arwen_metal_compat.h\"\n")
        msl = prelude + body + init_kernel
        return Translation(msl, self.kernel_infos, {g.name: g for g in self.globals_mut.values()},
                           self.globals_size, self.mode, self.opts, flags)


def _split_top(text: str) -> list:
    out, cur, depth = [], [], 0
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _ctor_body(text, m):
    """(params close offset, body open offset, body close offset) of a constructor definition."""
    body = next((ch for ch in m.get_children() if ch.kind == K.COMPOUND_STMT), None)
    if body is None:
        return None
    lp = text.find("(", m.location.offset)
    depth, i = 0, lp
    while i < body.extent.start.offset:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            break
        i += 1
    return i, body.extent.start.offset, body.extent.end.offset - 1


def _ctor_as_init(text, m, base_name):
    """``B(args) : a(x), b{y} { body }`` -> ``void __arwen_init_B(args) { a = x; b = {y}; body }``."""
    rp, bo, bc = _ctor_body(text, m)
    lp = text.find("(", m.location.offset)
    prefix = text[m.extent.start.offset:m.location.offset]
    params = text[lp:rp + 1]
    init = text[rp + 1:bo].strip()
    stmts = []
    if init.startswith(":"):
        for it in _split_top(init[1:]):
            mm = re.match(r"\s*([\w:]+)\s*([({])(.*)([)}])\s*$", it, re.S)
            if not mm:
                raise TranslationError(f"{base_name}: base constructor initialiser {it!r}")
            stmts.append(f"{mm.group(1)} = {{{mm.group(3)}}};" if mm.group(2) == "{" else f"{mm.group(1)} = {mm.group(3)};")
    return (f"\n{prefix.replace('explicit', '')}void __arwen_init_{base_name}{params} {{ "
            + " ".join(stmts) + text[bo + 1:bc] + " }\n")


def _derived_ctor_calls_init(text, rw, m, bases):
    """In a derived constructor, ``: Base<T>(args)`` becomes ``__arwen_init_Base(args);``
    at the start of its body."""
    cb = _ctor_body(text, m)
    if cb is None:
        return
    rp, bo, _ = cb
    init = text[rp + 1:bo]
    k = init.find(":")
    if k < 0:
        return
    items = _split_top(init[k + 1:])
    keep, calls = [], []
    for it in items:
        hit = next((name for name, spec in bases
                    if re.match(r"\s*%s\s*(<.*>)?\s*[({]" % re.escape(name), it, re.S)), None)
        if hit:
            args = it[it.index("(") + 1:it.rindex(")")] if "(" in it else it[it.index("{") + 1:it.rindex("}")]
            calls.append(f"this->__arwen_init_{hit}({args});")
        else:
            keep.append(it)
    if not calls:
        return
    rw.replace(rp + 1, bo + 1, (" : " + ", ".join(keep) if keep else "") + " { " + " ".join(calls))


_VIEW_HINT = re.compile(r"template\s*<\s*(?:class|typename)\s+(\w+)[^>]*>\s*struct\s+\w+\s*\{[^{}]*?\b\1\s*\*")


def parametrize_views(text: str, filename: str) -> str:
    """Array views (``template <class T> struct V { T* p; ... }``) that wrap BOTH
    thread-local storage arrays and device workspace: an MSL pointer member has
    one address space, so the pointer type becomes a template parameter,
    ``template <class T, ..., class __ARWEN_AP = device T*>``, and every view
    built over local storage (``V<float> c(c_local, 1)``, ``V<R,1> a = {a_st, ...}``,
    ``V<real> s = V<real>(s_local)`` and views derived from those) is spelled
    with ``thread T*``.  Views over device memory keep the default, so their
    code is unchanged."""
    if not _VIEW_HINT.search(text):
        return text
    tu, errors = parse(text, filename)
    if errors:
        return text
    tu._arwen_text, tu._arwen_filename = text, filename
    nodes, parent = [], {}
    stack = [(tu.cursor, None)]
    while stack:
        c, par = stack.pop()
        if par is not None and (c.location.file is None or c.location.file.name != filename):
            continue
        if par is not None:
            nodes.append(c)
            parent[c.hash] = par
        for ch in reversed(list(c.get_children())):
            stack.append((ch, c))
    views = {}
    for c in nodes:
        if c.kind != K.CLASS_TEMPLATE:
            continue
        tps = [ch for ch in c.get_children() if ch.kind in (K.TEMPLATE_TYPE_PARAMETER, K.TEMPLATE_NON_TYPE_PARAMETER)]
        if not tps or tps[0].kind != K.TEMPLATE_TYPE_PARAMETER:
            continue
        T_ = tps[0].spelling
        fields = [f for f in c.get_children() if f.kind == K.FIELD_DECL]
        ptrs = [f for f in fields if re.fullmatch(r"%s\s*\*" % re.escape(T_), f.type.spelling.strip())]
        if ptrs and not [f for f in fields if "*" in f.type.spelling and f not in ptrs]:
            views[c.spelling] = (c, T_, len(tps))
    if not views:
        return text

    def view_of(t):
        sp = t.spelling.replace("const ", "").strip()
        m = re.match(r"(\w+)\s*<", sp)
        return m.group(1) if m and m.group(1) in views else None

    def strip(e):
        while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len([k for k in e.get_children() if k.kind.is_expression()]) == 1:
            e = [k for k in e.get_children() if k.kind.is_expression()][0]
        return e

    thread_vars = set()

    def is_thread(e, depth=0):
        if e is None or depth > 6:
            return False
        e = strip(e)
        if e.kind == K.DECL_REF_EXPR and e.referenced is not None and e.referenced.kind == K.VAR_DECL:
            r = e.referenced
            if r.hash in thread_vars:
                return True
            p = parent.get(r.hash)
            local = p is not None and p.kind == K.DECL_STMT
            return local and _canon(r.type).kind == T.CONSTANTARRAY
        if e.kind == K.UNARY_OPERATOR and _tokens(e) and _tokens(e)[0].spelling == "&":
            ch = [k for k in e.get_children() if k.kind.is_expression()]
            r = strip(ch[0]).referenced if ch and strip(ch[0]).kind == K.DECL_REF_EXPR else None
            return r is not None and r.kind == K.VAR_DECL and parent.get(r.hash) is not None \
                and parent.get(r.hash).kind == K.DECL_STMT
        kids = [k for k in e.get_children() if k.kind.is_expression()]
        if e.kind in (K.BINARY_OPERATOR, K.CALL_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR, K.INIT_LIST_EXPR,
                      K.CSTYLE_CAST_EXPR, K.UNEXPOSED_EXPR) and kids:
            # a view built or derived from thread storage: its first operand decides
            if e.kind == K.CALL_EXPR and e.referenced is not None and e.referenced.kind == K.CXX_METHOD:
                return is_thread(kids[0], depth + 1)
            return is_thread(kids[0], depth + 1)
        return False

    rw = Rewriter(text)
    retyped = set()
    changed = True
    while changed:
        changed = False
        for v in nodes:
            if v.kind != K.VAR_DECL or v.hash in retyped or view_of(v.type) is None:
                continue
            p = parent.get(v.hash)
            if p is None or p.kind != K.DECL_STMT:
                continue
            init = [k for k in v.get_children() if k.kind.is_expression()]
            if not init or not is_thread(init[-1]):
                continue
            thread_vars.add(v.hash)
            retyped.add(v.hash)
            changed = True
    if not retyped:
        return text

    def close_angle(start, end):
        depth = 0
        for i in range(start, end):
            ch = text[i]
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def thr_arg(tid_start, tid_end):
        """Insert the thread pointer argument into the template-id text[tid_start:tid_end]."""
        lt = text.find("<", tid_start, tid_end)
        gt = close_angle(lt, tid_end) if lt >= 0 else None
        if gt is None:
            raise TranslationError(f"view type spelling not found in {text[tid_start:tid_end]!r}")
        first = _split_top(text[lt + 1:gt])[0]
        rw.insert(gt, f", __ARWEN_THRPTR({first})")

    for v in nodes:
        if v.hash not in retyped:
            continue
        name_tok = next(t for t in _tokens(v) if t.spelling == v.spelling and t.extent.start.offset >= v.location.offset)
        st = parent.get(v.hash)
        decls = [k for k in st.get_children() if k.kind == K.VAR_DECL]
        if len(decls) > 1:
            if any(d.hash not in retyped for d in decls):
                raise TranslationError(f"view {v.spelling}: declared together with views of another storage")
            if decls[0].hash != v.hash:
                continue            # one type for the whole statement, edited at the first declarator
            first = next(t for t in _tokens(decls[0]) if t.spelling == decls[0].spelling)
            thr_arg(st.extent.start.offset, first.extent.start.offset)
        else:
            thr_arg(v.extent.start.offset, name_tok.extent.start.offset)
        # an explicit view construction in the initialiser: V<...>(storage)
        for d in v.walk_preorder():
            if d.kind in (K.CXX_FUNCTIONAL_CAST_EXPR, K.CALL_EXPR, K.UNEXPOSED_EXPR) and d.hash != v.hash:
                toks = _tokens(d)
                if len(toks) > 2 and toks[0].spelling in views and toks[1].spelling == "<" \
                        and toks[0].extent.start.offset > name_tok.extent.start.offset:
                    thr_arg(toks[0].extent.start.offset, d.extent.end.offset)
                    break
    # the view templates: pointer type as a parameter
    for name, (c, T_, ntp) in views.items():
        toks = _tokens(c)
        if toks[0].spelling != "template":
            continue
        gt = close_angle(toks[1].extent.start.offset, c.extent.end.offset)
        rw.insert(gt, f", class __ARWEN_AP = __ARWEN_DEVPTR({T_})")
        bopen = text.find("{", gt)
        body_end = c.extent.end.offset
        body = text[bopen:body_end]
        nb = re.sub(r"\bconst\s+%s\s*\*" % re.escape(T_), "__arwen_cptr_t<__ARWEN_AP> ", body)
        nb = re.sub(r"(?<![\w:])%s\s*\*" % re.escape(T_), "__ARWEN_AP ", nb)
        nb = re.sub(r"const_cast<\s*__ARWEN_AP\s*>", "const_cast<__ARWEN_AP>", nb)
        # self references: same element type -> same pointer type; another element
        # type (a converting constructor) -> a fresh pointer parameter
        def self_ref(m):
            args = _split_top(m.group(1))
            if len(args) != ntp:
                return m.group(0)
            extra = "__ARWEN_AP" if args[0].strip() == T_ else "__ARWEN_AQ"
            return f"{name}<{m.group(1)}, {extra}>"
        nb = re.sub(r"\b%s<([^<>]*(?:<[^<>]*>[^<>]*)*)>" % re.escape(name), self_ref, nb)
        nb = re.sub(r"template\s*<\s*class\s+(\w+)\s*>(?=[^;{]*__ARWEN_AQ)", r"template<class \1, class __ARWEN_AQ>", nb)
        m = re.search(r"template\s*<\s*int\s+(\w+)\s*>\s*__device__\s*operator\s+%s<\s*%s\s*,\s*\1\s*,\s*__ARWEN_AP\s*>" %
                      (re.escape(name), re.escape(T_)), nb)
        if m:
            # rank conversion made explicit at call sites (the callee's view parameter is
            # generic, so the implicit conversion would not happen): keeps the pointer type
            nb = nb[:1] + (f" template<int __S> __device__ {name}<{T_},__S,__ARWEN_AP> __arwen_rank() const "
                           f"{{ return *this; }} ") + nb[1:]
        if nb != body:
            rw.replace(bopen, body_end, nb)
    return rw.apply()


def hoist_block_prototypes(text: str, filename: str) -> str:
    """f2c-style block-scope prototypes (``extern int f_(...), g_(...);`` inside a
    function body) move to file scope, just before the enclosing top-level
    declaration: Metal has no block-scope function declarations, and the
    functions may be defined further down the file."""
    if not re.search(r"\n[ \t]+extern\b", text):
        return text
    tu, errors = parse(text, filename)
    if errors:
        return text
    rw = Rewriter(text)
    moved = False
    stack = [(tu.cursor, None, None)]
    while stack:
        c, parent, top = stack.pop()
        if c.location.file is not None and c.location.file.name != filename:
            continue
        if c.kind == K.DECL_STMT and top is not None:
            kids = list(c.get_children())
            if kids and all(k.kind == K.FUNCTION_DECL for k in kids):
                # one file-scope declaration per prototype (``int f_(...), g_(...);``)
                prefix = text[c.extent.start.offset:kids[0].location.offset]
                decl = "".join(prefix + text[k.location.offset:k.extent.end.offset] + ";\n" for k in kids)
                rw.insert(top.extent.start.offset, decl, order=-1e9)
                rw.replace(c.extent.start.offset, c.extent.end.offset, ";")
                moved = True
                continue
        ntop = top if top is not None else (c if parent is not None and parent.kind == K.TRANSLATION_UNIT else None)
        for ch in reversed(list(c.get_children())):
            stack.append((ch, c, ntop))
    return rw.apply() if moved else text


def flatten_inheritance(text: str, filename: str) -> str:
    """Metal has no derived classes: copy each base's members into the derived
    struct (template parameters substituted, base constructors dropped)."""
    if not re.search(r"\b(struct|class)\s+\w+(\s*<[^>{;]*>)?\s*:\s*(public\s+|private\s+|protected\s+)?\w", text):
        return text
    tu, errors = parse(text, filename)
    if errors:
        return text
    rw = Rewriter(text)
    tu._arwen_text, tu._arwen_filename = text, filename
    changed = False
    for c in tu.cursor.walk_preorder():
        if c.location.file is None or c.location.file.name != filename:
            continue
        if c.kind not in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE):
            continue
        bases = [b for b in c.get_children() if b.kind == K.CXX_BASE_SPECIFIER]
        if not bases:
            continue
        body_open = text.find("{", c.extent.start.offset)
        colon = None
        depth = 0
        for i in range(c.location.offset, body_open):
            ch = text[i]
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            elif ch == ":" and depth == 0 and text[i + 1] != ":" and text[i - 1] != ":":
                colon = i
                break
        if colon is None:
            raise TranslationError(f"{c.spelling}: base clause not found")
        inserted = []
        for b in bases:
            base = b.referenced
            if base is None or base.location.file is None or base.location.file.name != filename:
                raise TranslationError(f"{c.spelling}: base class outside the module")
            if base.kind == K.STRUCT_DECL or base.kind == K.CLASS_DECL:
                tmpl = base.semantic_parent if False else None
            # the template declaration owning the members
            decl = base
            if decl.kind not in (K.CLASS_TEMPLATE, K.STRUCT_DECL, K.CLASS_DECL):
                raise TranslationError(f"{c.spelling}: unsupported base kind {decl.kind}")
            bopen = text.find("{", decl.location.offset)
            bclose = decl.extent.end.offset - 1
            body = text[bopen + 1:bclose]
            ctors = [m for m in decl.get_children() if m.kind == K.CONSTRUCTOR]
            drop = sorted((m.extent.start.offset - bopen - 1, m.extent.end.offset - bopen - 1) for m in ctors)
            for s0, s1 in reversed(drop):
                body = body[:s0] + body[s1:]
            # base constructors become init methods the derived constructors call
            body += "".join(_ctor_as_init(text, m, decl.spelling) for m in ctors if _ctor_body(text, m))
            params = [m.spelling for m in decl.get_children()
                      if m.kind in (K.TEMPLATE_TYPE_PARAMETER, K.TEMPLATE_NON_TYPE_PARAMETER)]
            spec = text[b.extent.start.offset:b.extent.end.offset]
            spec = re.sub(r"^(public|private|protected)\s+", "", spec.strip())
            args = _split_top(spec[spec.find("<") + 1:spec.rfind(">")]) if "<" in spec else []
            if len(args) != len(params):
                raise TranslationError(f"{c.spelling}: base {spec} template arguments do not match")
            for pn, av in zip(params, args):
                body = re.sub(r"\b%s\b" % re.escape(pn), av if re.fullmatch(r"[\w:]+", av) else f"({av})", body)
            fields = [f for f in decl.get_children() if f.kind == K.FIELD_DECL]
            conv = (f"\n__device__ operator {spec}() const {{ {spec} __b; "
                    # array members are copied element by element (not assignable)
                    + " ".join(f"__arwen_acopy(__b.{f.spelling}, this->{f.spelling});"
                               if _canon(f.type).kind == T.CONSTANTARRAY else f"__b.{f.spelling} = this->{f.spelling};"
                               for f in fields) + " return __b; }\n"
                    # binding to ``Base&`` (a reference parameter, generic once translated):
                    # declared for the parser only, empty in MSL (arwen_metal_compat.h)
                    + f"__ARWEN_AS_BASE_REF({spec})\n")
            inserted.append(f"/* members of base {spec} */{body}{conv}")
        rw.replace(colon, body_open + 1, "{" + "".join(inserted))
        for m in c.get_children():
            if m.kind == K.CONSTRUCTOR:
                _derived_ctor_calls_init(text, rw, m, [(b.referenced.spelling,
                                                        text[b.extent.start.offset:b.extent.end.offset]) for b in bases])
        changed = True
    return rw.apply() if changed else text


def metal_flags(opts: NvrtcOptions) -> tuple[str, ...]:
    """Metal compiler flags.  FP contraction is off everywhere: CUDA's explicit
    ``__f*_rn`` intrinsics must stay single-rounded, and plain expressions then
    round like the unfused Fortran/CPU oracles (see docs/metal-backend.md)."""
    if opts.fast_math:
        return ("-ffast-math",)
    return ("-fno-fast-math", "-ffp-contract=off", "-fmetal-math-fp32-functions=precise")


def translate(source: str, options=(), *, f64_mode: str = "sf64", filename: str = "kernel.cu",
              cupy_raw_module: bool = False) -> Translation:
    if f64_mode not in F64_MODES:
        raise ValueError(f"f64_mode must be one of {F64_MODES}")
    opts = NvrtcOptions.parse(options, cupy_raw_module=cupy_raw_module)
    if os.environ.get("ARWEN_METAL_FORCE_FTZ") == "1" and not (
            os.environ.get("ARWEN_METAL_FORCE_FTZ_SKIP") and re.search(os.environ["ARWEN_METAL_FORCE_FTZ_SKIP"], source)):
        # diagnostic only (results are not WRF's): what the FP32 gradual-underflow emulation costs
        opts = dataclasses.replace(opts, ftz=True)
    if opts.unknown:
        raise TranslationError(f"unsupported NVRTC options: {opts.unknown}")
    pp = preprocess(source, opts, filename)
    pp = flatten_inheritance(pp, filename)
    pp = hoist_block_prototypes(pp, filename)
    pp = parametrize_views(pp, filename)
    from . import fastlibm
    pp = fastlibm.apply(pp, opts, source=source, options=options, cupy_raw_module=cupy_raw_module)
    if os.environ.get("ARWEN_METAL_DIAG_FASTLIBM") == "1" and not (
            os.environ.get("ARWEN_METAL_DIAG_FASTLIBM_SKIP") and re.search(os.environ["ARWEN_METAL_DIAG_FASTLIBM_SKIP"], source)):
        pp = _diag_fast_libm(pp)
    mod = _Module(pp, opts, f64_mode, filename)
    mod._literal_done = set()
    mod._cast_done = set()
    mod._select_done = set()
    return mod.run()


_DIAG_LIBM = (("log10", "log10({0})", 1), ("log2", "log2({0})", 1), ("log", "log({0})", 1),
              ("exp2", "exp2({0})", 1), ("exp", "exp({0})", 1), ("pow", "pow({0}, {1})", 2),
              ("atan2", "atan2({0}, {1})", 2), ("atan", "atan({0})", 1), ("asin", "asin({0})", 1),
              ("acos", "acos({0})", 1), ("tanh", "tanh({0})", 1), ("sin", "sin({0})", 1),
              ("cos", "cos({0})", 1), ("tan", "tan({0})", 1), ("cbrt", "cbrt({0})", 1))


def _diag_fast_libm(pp: str) -> str:
    """DIAGNOSTIC ONLY (ARWEN_METAL_DIAG_FASTLIBM=1; results are not WRF's): replace
    the body of every float(float[, float]) function named like a libm function
    whose body computes in double by the FP32 hardware function, to measure how
    much of a kernel's time the glibc-exact float libm costs."""
    hdr = re.compile(r"\b(?:float|real)\s+(\w+)\s*\(\s*(?:const\s+)?(?:float|real)\s+(\w+)\s*"
                     r"(?:,\s*(?:const\s+)?(?:float|real)\s+(\w+)\s*)?\)\s*\{")
    out, pos = [], 0
    for m in hdr.finditer(pp):
        if m.start() < pos:
            continue
        name, x, y = m.group(1), m.group(2), m.group(3)
        depth, j = 1, m.end()
        while depth and j < len(pp):
            depth += {"{": 1, "}": -1}.get(pp[j], 0)
            j += 1
        body = pp[m.end():j]
        low = name.lower()
        expr = None
        for key, tmpl, arity in _DIAG_LIBM:
            if key in low and arity == (2 if y else 1):
                expr = tmpl.format(x, y)
                break
        if expr is None or not re.search(r"\bdouble\b", body):
            continue
        out.append(pp[pos:m.end()])
        out.append(f" return (float){expr}; }}")
        pos = j
    out.append(pp[pos:])
    return "".join(out)


def cache_key(source: str, options, f64_mode: str, cupy_raw_module: bool) -> str:
    from pathlib import Path
    inc = Path(__file__).resolve().parent / "include"
    h = hashlib.sha256()
    for part in (TRANSLATOR_REVISION, f64_mode, json.dumps(list(map(str, options or ()))), str(cupy_raw_module),
                 "sf_inline=" + os.environ.get("ARWEN_METAL_SF_INLINE", "") + os.environ.get("ARWEN_METAL_DIAG_FASTLIBM", "") + os.environ.get("ARWEN_METAL_DIAG_FASTLIBM_SKIP", "") + "force_ftz=" + os.environ.get("ARWEN_METAL_FORCE_FTZ", "") + os.environ.get("ARWEN_METAL_FORCE_FTZ_SKIP", ""), source):
        h.update(part.encode())
        h.update(b"\0")
    from . import fastlibm
    h.update(fastlibm.cache_token().encode())
    for p in sorted(inc.glob("*.h")) + [Path(__file__), Path(__file__).with_name("frontend.py"),
                                        Path(__file__).with_name("fastlibm.py"),
                                        Path(__file__).with_name("cuda_parse.h")]:
        h.update(p.read_bytes())
    return h.hexdigest()
