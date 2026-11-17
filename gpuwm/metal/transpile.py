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
              "__ballot_sync", "__any_sync", "__all_sync"}
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
        if old is not None and old != (end, new):
            raise TranslationError(f"conflicting rewrites at offset {start}: {self.text[start:end]!r}")
        self.repl[start] = (end, new)
        if opaque:
            self.opaque.append((start, end))

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
        return float.fromhex(s) if s.lower().startswith(("0x", "-0x")) else float(s)
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
                continue
            if parent is not None:
                self.parent[c.hash] = parent
                self.nodes.append(c)
                self.fn_of[c.hash] = fn
            nfn = fn
            if c.kind in FN_KINDS or c.kind == K.LAMBDA_EXPR:
                if c.kind != K.LAMBDA_EXPR:
                    nfn = c
                    usr = c.get_usr()
                    self.functions.setdefault(usr, []).append(c)
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
    def analyse(self):
        builtin_users, warp_users, callers = set(), set(), defaultdict(set)
        for c in self.nodes:
            fn = self.fn_of.get(c.hash)
            fusr = fn.get_usr() if fn is not None else None
            if c.kind == K.DECL_REF_EXPR:
                ref = c.referenced
                if c.spelling in BUILTIN_VARS and ref is not None and not self._main(ref):
                    builtin_users.add(fusr)
                elif ref is not None and ref.kind == K.VAR_DECL and self._main(ref) and self._is_mutable_global(ref):
                    builtin_users.add(fusr)
            elif c.kind == K.CALL_EXPR:
                if c.spelling in WARP_FUNCS and c.referenced is not None and not self._main(c.referenced):
                    warp_users.add(fusr)
                cu, _ = self.callee(c)
                if cu is not None and fusr is not None:
                    callers[cu].add(fusr)
            elif c.kind == K.VAR_DECL and self._is_extern_shared(c):
                self.dyn_shared_used = True
                builtin_users.add(fusr)
            elif c.kind == K.VAR_DECL and K.CUDASHARED_ATTR in self._attrs(c) and fn is not None \
                    and not self.is_kernel(fn) and not self._program_scope(c):
                ct = _canon(c.type)
                size, align = ct.get_size(), max(16, ct.get_align())
                self.ssmem_size = (self.ssmem_size + align - 1) // align * align
                self.ssmem_vars[c.hash] = self.ssmem_size
                self.ssmem_size += size
                self.ssmem_users.add(fusr)
                builtin_users.add(fusr)
        self.callers = callers
        need = set(u for u in builtin_users | warp_users if u)
        work = list(need)
        while work:
            u = work.pop()
            for caller in callers.get(u, ()):
                if caller not in need:
                    need.add(caller)
                    work.append(caller)
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
            has_init = any(ch.kind not in (K.CUDADEVICE_ATTR, K.CUDACONSTANT_ATTR, K.TYPE_REF, K.UNEXPOSED_ATTR,
                                           K.NAMESPACE_REF, K.TEMPLATE_REF) and ch.kind.is_expression()
                           for ch in c.get_children())
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
        for c in self.nodes:
            if c.kind != K.FIELD_DECL or _canon(c.type).kind != T.POINTER:
                continue
            toks = _tokens(c)
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None:
                continue
            space = "thread" if c.hash in thread_fields else "device"
            self.field_space[c.hash] = space
            if _canon(c.type.get_pointee()).kind == T.POINTER:
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, self.msl_type(c.type, space=space) + " ")
            elif c.extent.start.offset not in seen_starts:
                # ``const T *a, *b;``: an address-space specifier distributes over the declarators
                seen_starts.add(c.extent.start.offset)
                self.rw.insert(c.extent.start.offset, space + " ")

    def edit_init_lists(self):
        """Element types of brace initialisers (libclang shows their syntactic form,
        without the implicit conversions): walk the declared type alongside."""
        for c in self.nodes:
            if c.kind != K.VAR_DECL:
                continue
            inits = [ch for ch in c.get_children() if ch.kind == K.INIT_LIST_EXPR]
            if not inits:
                continue
            storage = self._program_scope(c)
            self._typed_init(inits[0], c.type, storage)

    def _typed_init(self, node, t, storage):
        ct = _canon(t)
        kids = [k for k in node.get_children() if k.kind.is_expression()]
        if ct.kind in (T.CONSTANTARRAY, T.INCOMPLETEARRAY):
            et = t.get_array_element_type()
            ect = _canon(et)
            aggregate = ect.kind in (T.CONSTANTARRAY, T.RECORD)
            for k in kids:
                if k.kind == K.INIT_LIST_EXPR:
                    self._typed_init(k, et, storage)
                elif aggregate:
                    if _canon(k.type).kind == ect.kind and k.kind != K.INIT_LIST_EXPR:
                        continue
                    raise TranslationError("brace-elided aggregate initialiser")
                else:
                    self._typed_leaf(k, et, storage)
        elif ct.kind == T.RECORD:
            decl = ct.get_declaration()
            fields = [f for f in decl.get_children() if f.kind == K.FIELD_DECL]
            for k, f in zip(kids, fields):
                if k.kind == K.INIT_LIST_EXPR:
                    self._typed_init(k, f.type, storage)
                elif _canon(f.type).kind in (T.CONSTANTARRAY, T.RECORD):
                    if _canon(k.type).kind != _canon(f.type).kind:
                        raise TranslationError("brace-elided aggregate initialiser")
                else:
                    self._typed_leaf(k, f.type, storage)
        elif kids:
            self._typed_leaf(kids[0], t, storage)

    def _typed_leaf(self, k, t, storage):
        to, frm = _arith_kind(t), _arith_kind(k.type)
        if to is None or frm is None or to == frm or "f64" not in (to, frm):
            return
        v = _literal_value(k)
        s0, s1 = k.extent.start.offset, k.extent.end.offset
        if v is not None:
            self.rw.replace(s0, s1, _f64_lit(v, storage) if to == "f64" else _hexf32(v) if to == "f32"
                            else str(int(v)))
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
        """Metal has no goto.  Forward gotos only: to a label right after the
        enclosing loop -> break; to a later label in an enclosing block -> a flag
        that skips the statements in between."""
        gotos = [c for c in self.nodes if c.kind == K.GOTO_STMT]

        def fnkey(c):
            f = self.fn_of.get(c.hash)
            return f.hash if f is not None else None
        labels = {(fnkey(c), _tokens(c)[0].spelling): c for c in self.nodes if c.kind == K.LABEL_STMT}
        flags = {}
        for g in gotos:
            target = next((ch for ch in g.get_children() if ch.kind == K.LABEL_REF), None)
            name = target.spelling if target is not None else _tokens(g)[1].spelling
            L = labels.get((fnkey(g), name))
            if L is None or L.extent.start.offset < g.extent.start.offset:
                raise TranslationError(f"backward or unresolved goto {name}")
            # ancestors of the goto
            chain, p = [], self.parent.get(g.hash)
            while p is not None and p.kind not in FN_KINDS:
                chain.append(p)
                p = self.parent.get(p.hash)
            loop = next((a for a in chain if a.kind in (K.FOR_STMT, K.WHILE_STMT, K.DO_STMT, K.SWITCH_STMT)), None)
            if loop is not None and loop.kind != K.SWITCH_STMT and loop.extent.end.offset <= L.extent.start.offset:
                between = self.text[loop.extent.end.offset:L.extent.start.offset]
                between = re.sub(r"\b\w+\s*:\s*;?", "", between).strip(" ;\n\t")
                if not between:
                    self.rw.replace(g.extent.start.offset, g.extent.end.offset, "break")
                    continue
            Lpar = self.parent.get(L.hash)
            S = next((a for a in [g] + chain if self.parent.get(a.hash) is not None
                      and self.parent.get(a.hash).hash == Lpar.hash), None)
            if S is None or Lpar.kind != K.COMPOUND_STMT:
                raise TranslationError(f"goto {name}: target not in an enclosing block")
            if any(a.kind in (K.FOR_STMT, K.WHILE_STMT, K.DO_STMT, K.SWITCH_STMT) for a in chain[:chain.index(S)] if S in chain):
                raise TranslationError(f"goto {name}: jumps out of a loop to a non-adjacent label")
            sib = [ch for ch in Lpar.get_children()]
            i = next(k for k, ch in enumerate(sib) if ch.hash == S.hash)
            j = next(k for k, ch in enumerate(sib) if ch.hash == L.hash)
            region = sib[i + 1:j]
            if any(ch.kind == K.DECL_STMT for ch in region):
                raise TranslationError(f"goto {name}: declarations between the goto and its label")
            flag = f"__arwen_goto_{name}_{g.extent.start.offset}"
            self.rw.replace(g.extent.start.offset, g.extent.end.offset, f"{flag} = true")
            if (fnkey(g), name) not in flags:
                flags[(fnkey(g), name)] = True
                self.rw.insert(sib[i].extent.start.offset, f"bool {flag} = false; ", order=-1e9)
                if region:
                    self.rw.insert(region[0].extent.start.offset, f"if (!{flag}) {{ ", order=-1e9)
                    endo = region[-1].extent.end.offset
                    semi = endo if self.text[endo - 1] in ";}" else self.text.find(";", endo) + 1
                    self.rw.insert(semi, " }", order=1e9)
            else:
                raise TranslationError(f"several gotos to {name}")
        for L in labels.values():
            toks = _tokens(L)
            if len(toks) >= 2 and toks[1].spelling == ":":
                self.rw.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, "")

    def run(self) -> Translation:
        self.collect()
        self.analyse()
        self.find_reassigned()
        self.eliminate_gotos()
        self.edit_fields()
        self.edit_init_lists()
        self.edit_linkage()
        self.edit_globals()
        self.edit_functions()
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

    def edit_globals(self):
        offset = 0
        for c in self.nodes:
            if c.kind != K.VAR_DECL or not self._program_scope(c):
                continue
            toks = _tokens(c)
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
                if children:
                    init = self.text[children[-1].extent.start.offset:children[-1].extent.end.offset]
                elem, dims = _array_dims(t)
                mty = self.msl_type(elem, storage=True).replace("const ", "")
                gv = GlobalVar(c.spelling, mty, size, align, offset, init)
                gv.dims = dims
                self.globals_mut[c.get_usr()] = gv
                offset += size
                # remove the declaration (through its terminating semicolon)
                end = c.extent.end.offset
                semi = self.text.find(";", end)
                self.rw.replace(c.extent.start.offset, semi + 1, f"/* global {c.spelling} -> globals buffer */")
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
        if not ch:
            return None
        callee_end = ch[0].extent.end.offset
        for t in _tokens(call):
            if t.extent.start.offset >= callee_end and t.spelling == "(":
                return t.extent.end.offset
        return None

    def edit_functions(self):
        for usr, decls in self.functions.items():
            kernel = any(self.is_kernel(d) for d in decls)
            for d in decls:
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
                unpack_ty = fty
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
                    "__arwen_bdim, __arwen_gdim, __arwen_lane, __arwen_g"
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
        return False

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
        if not kids:
            return ""
        d = kids[-1]
        return " = " + self.text[d.extent.start.offset:d.extent.end.offset]

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
            elif ct.kind == T.POINTER and self._dependent(t.get_pointee()) and not self._deducible_elsewhere(params, i):
                # the function template deduces its own parameter from T*: keep it
                # deducible, in device memory
                self.rw.replace(p.extent.start.offset, p.extent.end.offset,
                                f"{self.msl_type(t, space='device')} {pname}{self._param_default(p)}")
            elif ct.kind == T.POINTER:
                tp = f"__arwen_P{i}"
                tparams.append((tp, self.msl_type(t, space="device")))
                self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}{self._param_default(p)}")
            elif ct.kind == T.RECORD and self._record_has_pointers(t):
                # views holding pointers: duck-typed, so a flattened derived view or a
                # view over another address space binds too
                tp = f"__arwen_P{i}"
                tparams.append((tp, None))
                self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{tp} {pname}")
            elif ct.kind in (T.LVALUEREFERENCE, T.RVALUEREFERENCE):
                pt = t.get_pointee()
                if pt.is_const_qualified() and ct.kind == T.LVALUEREFERENCE and not self._record_has_pointers(pt):
                    ty = self.msl_type(pt).replace("const ", "")
                    self.rw.replace(p.extent.start.offset, p.extent.end.offset, f"{ty} {pname}{self._param_default(p)}")
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
                for t in toks:
                    if t.extent.start.offset >= name_tok.extent.start.offset:
                        break
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
            hdr = ", ".join(f"typename {t}" + (f" = {d}" if d else "") for t, d in tparams)
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

    def edit_expressions(self):
        for c in self.nodes:
            k = c.kind
            if k == K.DECL_REF_EXPR:
                ref = c.referenced
                if ref is None:
                    continue
                if c.spelling in BUILTIN_VARS and not self._main(ref):
                    self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                    f"__arwen_ctx.{BUILTIN_VARS[c.spelling]}")
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
            self.rw.replace(ck[1].extent.end.offset, c.extent.end.offset, ")")
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
                        and not self.rw.in_opaque(c.extent.start.offset, c.extent.end.offset):
                    self.rw.replace(toks[0].extent.start.offset, toks[0].extent.end.offset, "__arwen_fneg(")
                    self.rw.insert(c.extent.end.offset, ")", order=c.extent.end.offset - c.extent.start.offset)

    def edit_explicit_cast(self, c):
        ops = [ch for ch in c.get_children() if ch.kind.is_expression()]
        if not ops:
            return
        op = ops[-1]
        s0, s1 = op.extent.start.offset, op.extent.end.offset
        to_c, fr_c = _canon(c.type), _canon(op.type)
        if to_c.kind == T.POINTER and self._is_null(op):
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
            pointee = self.msl_type(c.type.get_pointee(), storage=True)
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
            self.rw.insert(s1, ")", order=1e9)
        self._cast_done.add(op.hash)

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
            self.rw.insert(lp, "__arwen_ctx, ", order=-1)
            return
        usr, ref = self.callee(c)
        if usr is None:
            return
        if usr in self.needs_ctx:
            lp = self._call_lparen(c)
            if lp is None:
                raise TranslationError(f"call to {c.spelling}: argument list not found")
            nargs = len(list(self.call_args(c)))
            self.rw.insert(lp, "__arwen_ctx" + (", " if nargs else ""), order=-1)
        params = self._params(ref)
        for i, a in enumerate(self.call_args(c)):
            if i < len(params) and _canon(params[i].type).kind == T.POINTER and self._is_null(a):
                self.rw.replace(a.extent.start.offset, a.extent.end.offset,
                                f"(({self.msl_type(params[i].type, space='device')})nullptr)", opaque=True)
        refs = self.ref_params.get(usr)
        if refs:
            for i, a in enumerate(self.call_args(c)):
                if refs.get(i) == "ptr":
                    self.rw.insert(a.extent.start.offset, "&(", order=-5)
                    self.rw.insert(a.extent.end.offset, ")", order=5)

    def edit_local_var(self, c):
        if self._program_scope(c):
            return
        attrs = self._attrs(c)
        toks = _tokens(c)
        if K.CUDASHARED_ATTR in attrs:
            fn = self.fn_of.get(c.hash)
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
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, new)
                return
            if fn is None or not self.is_kernel(fn):
                raise TranslationError(f"__shared__ variable {c.spelling} outside a kernel body")
            if self._is_extern_shared(c):
                elem = c.type.get_array_element_type() if _canon(c.type).kind in (T.INCOMPLETEARRAY, T.CONSTANTARRAY) else c.type
                ety = self.msl_type(elem)
                start = c.extent.start.offset
                prev = self.text.rfind("extern", 0, start)
                if prev >= 0 and self.text[prev:start].strip() == "extern":
                    start = prev
                self.rw.replace(start, c.extent.end.offset,
                                f"threadgroup {ety}* {c.spelling} = reinterpret_cast<threadgroup {ety}*>(__arwen_ctx.dsmem)")
                return
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
        if ct.kind == T.CONSTANTARRAY and _canon(_array_dims(c.type)[0]).kind == T.POINTER:
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            elem = _array_dims(c.type)[0]
            if name_tok is not None:
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                self.msl_type(elem, space="device") + " ")
            return
        if ct.kind == T.POINTER:
            init = [ch for ch in c.get_children() if ch.kind.is_expression()]
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None:
                return
            p = self.parent.get(c.hash)
            first_in_stmt = True
            if p is not None and p.kind == K.DECL_STMT:
                decls = [ch for ch in p.get_children() if ch.kind == K.VAR_DECL]
                first_in_stmt = decls[0].hash == c.hash
                if len(decls) > 1:
                    raise TranslationError(f"multiple pointer declarators in one statement ({c.spelling})")
            if not first_in_stmt:
                return
            if init and not self._is_null(init[-1]):
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, "auto ")
            elif init:
                rhs = self._first_assignment(c)
                if rhs is None:
                    # never re-pointed: a null pointer of the declared type, in device memory
                    self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                    self.msl_type(c.type, space="device") + " ")
                else:
                    self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                    f"decltype({self.text[rhs.extent.start.offset:rhs.extent.end.offset]}) ")
            else:
                # address space from the first assignment to this variable
                rhs = self._first_assignment(c)
                if rhs is None:
                    raise TranslationError(f"pointer {c.spelling} declared without initialiser or assignment")
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                f"decltype({self.text[rhs.extent.start.offset:rhs.extent.end.offset]}) ")
        elif ct.kind == T.LVALUEREFERENCE:
            name_tok = next((t for t in toks if t.spelling == c.spelling), None)
            if name_tok is None:
                return
            pt = c.type.get_pointee()
            if pt.is_const_qualified():
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset,
                                self.msl_type(pt).replace("const ", "") + " ")
            else:
                self.rw.replace(c.extent.start.offset, name_tok.extent.start.offset, "decltype(auto) ")

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
            if to == "f64":
                self.rw.replace(s, e, _f64_lit(v, self._in_storage_init(c)))
            elif to == "f32":
                self.rw.replace(s, e, _hexf32(v))
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
        (re.compile(r'\{\s*\.reg\s+\.pred\s+%?%?\w+\s*;\s*setp\.gt\.f32\s+%?%?\w+\s*,\s*%1\s*,\s*%2\s*;\s*selp\.u32\s+%0\s*,\s*1\s*,\s*0\s*,\s*%?%?\w+\s*;\s*\}'), "gt"),
        (re.compile(r'^\s*$'), "empty"),
    ]

    def edit_asm(self, c):
        text = self.text[c.extent.start.offset:c.extent.end.offset]
        m = re.match(r'asm\s*(volatile\s*)?\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)(.*)\)\s*$', text, re.S)
        if not m:
            raise TranslationError(f"unrecognised inline asm: {text[:120]!r}")
        tmpl = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(2)))
        rest = m.group(3)
        sections = [s.strip() for s in self._split_colons(rest)]
        outs = re.findall(r'"=[a-z]"\s*\(([^)]*)\)', sections[0] if sections else "")
        ins = re.findall(r'"[a-z]"\s*\(([^)]*)\)', sections[1] if len(sections) > 1 else "")
        for rx, kind in self._ASM:
            mm = rx.search(tmpl)
            if not mm:
                continue
            if kind == "binop" and len(outs) == 1 and len(ins) == 2:
                fn = {"add": "__fadd_rn", "sub": "__fsub_rn", "mul": "__fmul_rn", "div": "__fdiv_rn"}[mm.group(1)]
                self.rw.replace(c.extent.start.offset, c.extent.end.offset, f"{outs[0]} = {fn}({ins[0]}, {ins[1]})")
                return
            if kind == "gt" and len(outs) == 1 and len(ins) == 2:
                self.rw.replace(c.extent.start.offset, c.extent.end.offset,
                                f"{outs[0]} = __arwen_fgt_u32({ins[0]}, {ins[1]})")
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
                   "#include \"arwen_metal_compat.h\"\n")
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
            drop = sorted((m.extent.start.offset - bopen - 1, m.extent.end.offset - bopen - 1)
                          for m in decl.get_children() if m.kind == K.CONSTRUCTOR)
            for s0, s1 in reversed(drop):
                body = body[:s0] + body[s1:]
            params = [m.spelling for m in decl.get_children()
                      if m.kind in (K.TEMPLATE_TYPE_PARAMETER, K.TEMPLATE_NON_TYPE_PARAMETER)]
            spec = text[b.extent.start.offset:b.extent.end.offset]
            spec = re.sub(r"^(public|private|protected)\s+", "", spec.strip())
            args = _split_top(spec[spec.find("<") + 1:spec.rfind(">")]) if "<" in spec else []
            if len(args) != len(params):
                raise TranslationError(f"{c.spelling}: base {spec} template arguments do not match")
            for pn, av in zip(params, args):
                body = re.sub(r"\b%s\b" % re.escape(pn), av if re.fullmatch(r"[\w:]+", av) else f"({av})", body)
            fields = [f.spelling for f in decl.get_children() if f.kind == K.FIELD_DECL]
            conv = (f"\n__device__ operator {spec}() const {{ {spec} __b; "
                    + " ".join(f"__b.{f} = this->{f};" for f in fields) + " return __b; }\n")
            inserted.append(f"/* members of base {spec} */{body}{conv}")
        rw.replace(colon, body_open + 1, "{" + "".join(inserted))
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
    if opts.unknown:
        raise TranslationError(f"unsupported NVRTC options: {opts.unknown}")
    pp = preprocess(source, opts, filename)
    pp = flatten_inheritance(pp, filename)
    mod = _Module(pp, opts, f64_mode, filename)
    mod._literal_done = set()
    mod._cast_done = set()
    mod._select_done = set()
    return mod.run()


def cache_key(source: str, options, f64_mode: str, cupy_raw_module: bool) -> str:
    from pathlib import Path
    inc = Path(__file__).resolve().parent / "include"
    h = hashlib.sha256()
    for part in (TRANSLATOR_REVISION, f64_mode, json.dumps(list(map(str, options or ()))), str(cupy_raw_module), source):
        h.update(part.encode())
        h.update(b"\0")
    for p in sorted(inc.glob("*.h")) + [Path(__file__), Path(__file__).with_name("frontend.py"),
                                        Path(__file__).with_name("cuda_parse.h")]:
        h.update(p.read_bytes())
    return h.hexdigest()
