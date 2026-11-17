"""Verified fast paths for ArWen's glibc-exact float libm copies.

ArWen's CUDA physics carries per-scheme device copies of glibc's logf/expf
(``mynn_glibc_logf``, ``gfk_log``, ``rlw_exp``...): a table and a short
polynomial evaluated in binary64, rounded once to binary32.  On the Apple GPU
binary64 is integer emulation, which makes these copies the largest cost of
several physics kernels.

For a copy whose fingerprint is in ``fastlibm_receipt.json`` the translator
renames the original to ``<name>__arwen_exact`` and inserts, under the original
name, a wrapper that returns the float-float fast path of arwen_fastlibm.h when
that result is certain and calls the exact original otherwise.  A fingerprint
enters the receipt only after tools/metal/verify_fastlibm.py has compared the
wrapper with the original on all 2^32 binary32 inputs on the GPU with zero
differing words.  The fingerprint covers the function, every table and helper
it reaches, the compile options and the fast-path source, so any change to
either side drops the copy back to the exact path until it is verified again.

Copies seen without a receipt entry are recorded (source and options) under
the cache directory for verify_fastlibm.py to check.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INC = HERE / "include"
RECEIPT = HERE / "fastlibm_receipt.json"
UNSURE_BITS = 0x7FBADBAD

# family -> (fast function, the files its fingerprint covers)
FAMILIES = {
    "logf": ("__arwen_fast_glibc_logf", ("arwen_fastlibm_ff.h", "arwen_fastlibm_logf.h",
                                         "arwen_fastlibm_tables_logf.h")),
    "expf": ("__arwen_fast_glibc_expf", ("arwen_fastlibm_ff.h", "arwen_fastlibm_expf.h",
                                         "arwen_fastlibm_tables_expf.h")),
    "powf": ("__arwen_fast_glibc_powf", ("arwen_fastlibm_ff.h", "arwen_fastlibm_powf.h",
                                         "arwen_fastlibm_tables_logf.h", "arwen_fastlibm_tables_expf.h",
                                         "arwen_fastlibm_tables_powf.h")),
}

_HDR1 = re.compile(r"\b(float|real)\s+(\w+)\s*\(\s*(?:const\s+)?(float|real)\s+(\w+)\s*\)\s*\{")
_HDR2 = re.compile(r"\b(float|real)\s+(\w+)\s*\(\s*(?:const\s+)?(float|real)\s+(\w+)\s*,"
                   r"\s*(?:const\s+)?(float|real)\s+(\w+)\s*\)\s*\{")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
_LITERAL = re.compile(r"^\(*\s*(-?)\s*(0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*[pP][+-]?\d+|\d+\.?\d*(?:[eE][+-]?\d+)?|"
                      r"\.\d+(?:[eE][+-]?\d+)?)([fF]?)\s*\)*$")


def family_of(name: str, arity: int):
    low = name.lower()
    if arity == 2:
        return "powf" if re.search(r"pow(?!_?log|_?exp|_?checkint)", low) else None
    if re.search(r"log(?!10|2|1p)", low) and "pow" not in low:
        return "logf"
    if re.search(r"exp(?!2|m1|10|_?core)", low) and "pow" not in low:
        return "expf"
    return None


def float32_bits_of_literal(text: str):
    """binary32 bits of a C float argument written as a literal (None if it is not one).
    'f' suffix: decimal -> binary32 directly; none: decimal -> binary64 -> binary32 (C's conversion)."""
    from fractions import Fraction
    import struct
    m = _LITERAL.match(text.strip())
    if not m:
        return None
    sign, body, suffix = m.groups()
    if body.lower().startswith("0x"):
        v = Fraction(float.fromhex(body))
    else:
        v = Fraction(body)
        if not suffix:
            v = Fraction(float(v))            # the binary64 literal
    if sign:
        v = -v

    def to_f32(q):                            # round-to-nearest-even, exact
        if q == 0:
            return 0
        s = 1 if q < 0 else 0
        q = abs(q)
        e = q.numerator.bit_length() - q.denominator.bit_length()
        if Fraction(2) ** e > q:
            e -= 1
        e = max(e, -126)
        scale = Fraction(2) ** (23 - e)
        m_ = q * scale
        n = m_.numerator // m_.denominator
        rem = m_ - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
            n += 1
        if n == 1 << 24:
            n >>= 1
            e += 1
        if e > 127:
            return (s << 31) | 0x7F800000
        if n < (1 << 23):                     # subnormal
            return (s << 31) | n
        return (s << 31) | ((e + 127) << 23) | (n - (1 << 23))
    bits = to_f32(v)
    struct.pack("<I", bits)
    return bits


def _toplevel(pp: str):
    """Top-level items of preprocessed C++ text: (defined name or None, start, end)."""
    items, depth, start, i, n = [], 0, 0, 0, len(pp)
    while i < n:
        c = pp[i]
        if c in "\"'":                                  # skip literals
            q = c
            i += 1
            while i < n and pp[i] != q:
                i += 2 if pp[i] == "\\" else 1
        elif c == "/" and pp.startswith("//", i):
            j = pp.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and pp.startswith("/*", i):
            j = pp.find("*/", i + 2)
            i = n if j < 0 else j + 1
        elif c == "#" and (i == 0 or pp[i - 1] == "\n"):
            j = pp.find("\n", i)
            i = n if j < 0 else j
            start = i + 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                k = i + 1
                while k < n and pp[k] in " \t\r\n":
                    k += 1
                if k < n and pp[k] == ";":
                    i = k
                items.append((start, i + 1))
                start = i + 1
        elif c == ";" and depth == 0:
            items.append((start, i + 1))
            start = i + 1
        i += 1
    out = []
    for s, e in items:
        text = pp[s:e]
        head = text.split("{", 1)[0].split("=", 1)[0].strip().rstrip(";")
        m = re.search(r"(\w+)\s*(?:\(|\[|$)", head.strip())
        names = re.findall(r"(\w+)\s*\(", head)
        name = names[-1] if "(" in head and names else (m.group(1) if m else None)
        out.append((name, s, e))
    return out


def _closure(pp: str, items, root: str) -> str:
    defs = {}
    for name, s, e in items:
        if name:
            defs.setdefault(name, []).append(pp[s:e])
    seen, order, todo = set(), [], [root]
    while todo:
        n = todo.pop()
        if n in seen or n not in defs:
            continue
        seen.add(n)
        order.append(n)
        for text in defs[n]:
            todo.extend(t for t in set(_IDENT.findall(text)) if t not in seen and t in defs)
    return "\n".join(re.sub(r"\s+", " ", t).strip() for n in sorted(order) for t in defs[n])


def closure_source(pp: str, root: str) -> str:
    """The definitions ``root`` reaches, in their original order (a standalone translation unit)."""
    items = _toplevel(pp)
    names = set()
    defs = {}
    for name, s, e in items:
        if name:
            defs.setdefault(name, []).append((s, e))
    todo = [root]
    while todo:
        n = todo.pop()
        if n in names or n not in defs:
            continue
        names.add(n)
        for s, e in defs[n]:
            todo.extend(t for t in set(_IDENT.findall(pp[s:e])) if t not in names and t in defs)
    keep = sorted((s, e) for n in names for s, e in defs[n])
    return "\n".join(pp[s:e] for s, e in keep) + "\n"


def _real_is_float(pp: str) -> bool:
    m = re.findall(r"typedef\s+(\w+)\s+real\s*;", pp)
    return bool(m) and all(t == "float" for t in m)


@dataclasses.dataclass
class Candidate:
    name: str
    family: str
    key: str
    start: int            # the definition's top-level item
    end: int
    header_end: int       # index of its opening brace
    ybits: int | None = None   # powf: the literal exponent this entry covers


def _family_digest(family: str) -> str:
    h = hashlib.sha256()
    for f in FAMILIES[family][1]:
        h.update((INC / f).read_bytes())
    return h.hexdigest()


def _key(family, opts, closure, ybits=None) -> str:
    h = hashlib.sha256()
    optrepr = json.dumps(dataclasses.asdict(opts), sort_keys=True, default=list)
    for part in (family, _family_digest(family), optrepr, closure, "" if ybits is None else f"y=0x{ybits:08X}"):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _split_args(text: str, start: int):
    """Arguments of the call whose '(' is at text[start]; (args, index after ')')."""
    depth, i, cur, args = 0, start, [], []
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c == ")":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                return args, i + 1
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    return None, start


def candidates(pp: str, opts) -> list[Candidate]:
    items = _toplevel(pp)
    out = []
    real_float = _real_is_float(pp)
    for hdr, arity in ((_HDR1, 1), (_HDR2, 2)):
        for m in hdr.finditer(pp):
            g = m.groups()
            name, types = g[1], (g[0], g[2]) + ((g[4],) if arity == 2 else ())
            fam = family_of(name, arity)
            if fam is None or ("real" in types and not real_float):
                continue
            item = next(((n, s, e) for n, s, e in items if s <= m.start() < e), None)
            if item is None or item[0] != name:
                continue
            _, s, e = item
            if not re.search(r"\bdouble\b", pp[m.end():e]):
                continue
            closure = _closure(pp, items, name)
            if arity == 1:
                out.append(Candidate(name, fam, _key(fam, opts, closure), s, e, m.end() - 1))
                continue
            # powf: one entry per literal exponent used at a call site
            seen = set()
            for c in re.finditer(r"\b" + re.escape(name) + r"\s*\(", pp):
                if s <= c.start() < e:
                    continue
                args, _end = _split_args(pp, c.end() - 1)
                if not args or len(args) != 2:
                    continue
                yb = float32_bits_of_literal(args[1])
                if yb is None or yb in seen:
                    continue
                seen.add(yb)
                out.append(Candidate(name, fam, _key(fam, opts, closure, yb), s, e, m.end() - 1, yb))
    return out


def _wrapper_name(c: Candidate) -> str:
    return f"{c.name}__arwen_y{c.ybits:08X}"


def substitute(pp: str, cands) -> str:
    """Route verified candidates.

    logf/expf: rename the copy to <name>__arwen_exact, insert the routed wrapper
    under the original name.  powf: add <name>__arwen_y<bits>(x) after the copy
    and rewrite the call sites whose exponent is that literal."""
    unary = [c for c in cands if c.ybits is None]
    powc = [c for c in cands if c.ybits is not None]
    edits = []                                            # (start, end, replacement)
    for c in unary:
        item = pp[c.start:c.end]
        hdr = pp[c.start:c.header_end]
        m = list(re.finditer(r"\b" + re.escape(c.name) + r"\s*\(", hdr))[-1]
        exact_hdr = hdr[:m.start()] + c.name + "__arwen_exact" + hdr[m.start() + len(c.name):]
        arg = re.search(r"\(\s*(?:const\s+)?\w+\s+(\w+)\s*\)", hdr[m.start():]).group(1)
        fast = FAMILIES[c.family][0]
        wrapper = (f"{hdr}{{\n    float __arwen_r = {fast}({arg});\n"
                   f"    if (__float_as_uint(__arwen_r) != 0x{UNSURE_BITS:08X}u) return __arwen_r;\n"
                   f"    return {c.name}__arwen_exact({arg});\n}}\n")
        edits.append((c.start, c.end, exact_hdr + item[len(hdr):] + "\n" + wrapper))
    by_copy = {}
    for c in powc:
        by_copy.setdefault((c.name, c.start, c.end), []).append(c)
    items = _toplevel(pp) if by_copy else []
    routes = {}                                           # name -> {ybits: wrapper name}
    for (name, s, e), cs in by_copy.items():
        hdr = pp[s:cs[0].header_end]
        qual = re.sub(r"\b" + re.escape(name) + r"\s*\(.*$", "", hdr, flags=re.S)   # '__device__ real '
        first = min(st for n, st, en in items if n == name)
        edits.append((first, first, "".join(f"{qual}{_wrapper_name(c)}(float __arwen_x);\n" for c in cs)))
        edits.append((e, e, "".join(
            f"\n{qual}{_wrapper_name(c)}(float __arwen_x) {{\n"
            f"    float __arwen_r = {FAMILIES['powf'][0]}(__arwen_x, __uint_as_float(0x{c.ybits:08X}u));\n"
            f"    if (__float_as_uint(__arwen_r) != 0x{UNSURE_BITS:08X}u) return __arwen_r;\n"
            f"    return {name}(__arwen_x, __uint_as_float(0x{c.ybits:08X}u));\n}}\n" for c in cs)))
        routes[name] = {c.ybits: _wrapper_name(c) for c in cs}
    out, pos = [], 0
    for a, b, text in sorted(edits, key=lambda t: (t[0], t[1])):
        if a < pos:
            continue                                     # nested inside a rewritten call: keep the outer one
        out.append(pp[pos:a])
        out.append(text)
        pos = b
    out.append(pp[pos:])
    text = "".join(out)
    return _rewrite_calls(text, routes) if routes else text


def _rewrite_calls(text: str, routes: dict) -> str:
    """Replace name(x, <literal>) by the routed wrapper(x), innermost calls included."""
    pat = re.compile(r"\b(" + "|".join(map(re.escape, routes)) + r")\s*\(")
    out, pos = [], 0
    for m in pat.finditer(text):
        if m.start() < pos:
            continue
        args, end = _split_args(text, m.end() - 1)
        if not args or len(args) != 2:
            continue
        yb = float32_bits_of_literal(args[1])
        w = routes[m.group(1)].get(yb)
        if w is None:
            continue
        out.append(text[pos:m.start()])
        out.append(f"{w}({_rewrite_calls(args[0], routes)})")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _receipt() -> dict:
    try:
        return json.loads(RECEIPT.read_text())
    except FileNotFoundError:
        return {}


def apply(pp: str, opts, *, source: str, options, cupy_raw_module: bool = False) -> str:
    """The translator hook: route verified copies; record unverified ones."""
    if os.environ.get("ARWEN_METAL_FASTLIBM", "1") == "0":
        return pp
    cands = candidates(pp, opts)
    if not cands:
        return pp
    rec = _receipt().get("verified", {})
    force = os.environ.get("ARWEN_METAL_FASTLIBM_VERIFYING", "")    # verify_fastlibm.py only
    ok = [c for c in cands if rec.get(c.key, {}).get("mismatches") == 0 or c.key == force]
    pending = [c for c in cands if c.key not in rec]
    if pending:
        from .native_build import cache_dir
        d = cache_dir() / "fastlibm_pending"
        d.mkdir(parents=True, exist_ok=True)
        for c in pending:
            p = d / f"{c.key}.json"
            if not p.exists():
                p.write_text(json.dumps({"name": c.name, "family": c.family, "ybits": c.ybits,
                                         "options": list(map(str, options or ())),
                                         "cupy_raw_module": bool(cupy_raw_module), "source": source}))
    return substitute(pp, ok) if ok else pp


def cache_token() -> str:
    """Part of the module cache key: the receipt and the fast-path sources."""
    h = hashlib.sha256()
    h.update(os.environ.get("ARWEN_METAL_FASTLIBM", "1").encode())
    h.update(os.environ.get("ARWEN_METAL_FASTLIBM_VERIFYING", "").encode())
    h.update(RECEIPT.read_bytes() if RECEIPT.exists() else b"")
    for fam in FAMILIES:
        h.update(_family_digest(fam).encode())
    return h.hexdigest()
