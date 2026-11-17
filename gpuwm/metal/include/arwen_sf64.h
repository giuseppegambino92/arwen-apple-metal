// arwen_sf64.h -- IEEE-754 binary64 arithmetic on 64-bit integers.
//
// Apple GPUs have no FP64 hardware.  CUDA ``double`` code in ArWen is written
// for bit-exact IEEE binary64 (``__dmul_rn``, ``__double_as_longlong``, the
// correctly rounded math of f64cr.cuh), so the reference FP64 mode emulates
// binary64 exactly: round-to-nearest-even, gradual underflow (binary64 has no
// flush-to-zero on CUDA either), +-0, infinities and NaN.
//
// The rounding/packing structure follows Berkeley SoftFloat 3e
// (J. R. Hauser; BSD-3-Clause, see licenses/), restricted to
// round-to-nearest-even with no exception flags.  Division and square root
// are exact digit-recurrence; fma uses an exact 128-bit sum.
//
// NaN results are the canonical quiet NaN 0x7FF8000000000000 when an
// operation is invalid, otherwise the first NaN operand, quieted.  (CUDA
// leaves NaN payloads unspecified.)
//
// This file is valid MSL and valid host C++17 (tests/metal/test_sf64_host.py
// checks it exhaustively against the host FPU).
#pragma once

#ifdef __METAL_VERSION__
#define SF_INLINE inline __attribute__((always_inline))
// the arithmetic entry points stay out of line: inlining a full soft-float
// operation at every double/subnormal-capable float operation of a large
// kernel made Metal's backend compile run for many minutes
#define SF_OUTLINE __attribute__((noinline))
#define SF_CONSTEXPR constexpr
typedef ulong sf_u64;
typedef long sf_i64;
typedef uint sf_u32;
SF_INLINE int sf_clz64(sf_u64 a) { return (int)metal::clz(a); }
SF_INLINE sf_u64 sf_mulhi64(sf_u64 a, sf_u64 b) { return metal::mulhi(a, b); }
SF_INLINE sf_u32 sf_f32_bits(float f) { return as_type<sf_u32>(f); }
SF_INLINE float sf_bits_f32(sf_u32 u) { return as_type<float>(u); }
#define SF_THREAD thread
#else
#include <cstdint>
#include <cstring>
#define SF_INLINE inline
#define SF_OUTLINE inline
#define SF_CONSTEXPR constexpr
typedef uint64_t sf_u64;
typedef int64_t sf_i64;
typedef uint32_t sf_u32;
SF_INLINE int sf_clz64(sf_u64 a) { return a ? __builtin_clzll(a) : 64; }
SF_INLINE sf_u64 sf_mulhi64(sf_u64 a, sf_u64 b) { return (sf_u64)(((unsigned __int128)a * b) >> 64); }
SF_INLINE sf_u32 sf_f32_bits(float f) { sf_u32 u; memcpy(&u, &f, 4); return u; }
SF_INLINE float sf_bits_f32(sf_u32 u) { float f; memcpy(&f, &u, 4); return f; }
#define SF_THREAD
#endif

#define SF_DEFAULT_NAN 0x7FF8000000000000UL
#define SF_SIGN(u) ((bool)((u) >> 63))
#define SF_EXP(u) ((int)(((u) >> 52) & 0x7FF))
#define SF_FRAC(u) ((u) & 0x000FFFFFFFFFFFFFUL)
#define SF_PACK(s, e, sig) ((((sf_u64)(s)) << 63) + (((sf_u64)(e)) << 52) + (sig))
#define SF_ISNAN(u) ((~(u) & 0x7FF0000000000000UL) == 0 && SF_FRAC(u))

struct sf_u128 { sf_u64 hi, lo; };

SF_INLINE sf_u64 sf_shift_right_jam64(sf_u64 a, int dist) {
    return (dist < 63) ? ((a >> dist) | (sf_u64)((a << ((-dist) & 63)) != 0)) : (sf_u64)(a != 0);
}

SF_INLINE sf_u64 sf_propagate_nan(sf_u64 a, sf_u64 b) {
    if (SF_ISNAN(a)) return a | 0x0008000000000000UL;
    return b | 0x0008000000000000UL;
}

// SoftFloat 3e softfloat_roundPackToF64, round-to-nearest-even.
// ``sig`` carries the significand with its leading 1 at bit 62; ``exp`` is the
// biased exponent minus one (the leading bit is added by SF_PACK).
// ``odd``: round to odd instead (keep the truncated significand, set its last
// bit when anything was discarded).  A binary64 result rounded to odd and then
// rounded once to binary32 is the correctly rounded binary32 result (53 >= 24+2),
// which is how the FP32 subnormal emulation computes exact float operations.
SF_INLINE sf_u64 sf_round_pack(bool sign, int exp, sf_u64 sig, bool odd = false) {
    sf_u64 round_bits = sig & 0x3FF;
    if ((sf_u32)exp >= 0x7FDu) {
        if (exp < 0) {
            sig = sf_shift_right_jam64(sig, -exp);
            exp = 0;
            round_bits = sig & 0x3FF;
        } else if (exp > 0x7FD || sig + 0x200 >= 0x8000000000000000UL) {
            return SF_PACK(sign, 0x7FF, 0);
        }
    }
    if (odd) {
        sig = (sig >> 10) | (sf_u64)(round_bits != 0);
    } else {
        sig = (sig + 0x200) >> 10;
        sig &= ~(sf_u64)((round_bits ^ 0x200) == 0);
    }
    if (!sig) exp = 0;
    return SF_PACK(sign, exp, sig);
}

SF_INLINE sf_u64 sf_norm_round_pack(bool sign, int exp, sf_u64 sig, bool odd = false) {
    int shift = sf_clz64(sig) - 1;
    exp -= shift;
    if (shift >= 10 && (sf_u32)exp < 0x7FDu)
        return SF_PACK(sign, sig ? exp : 0, sig << (shift - 10));
    return sf_round_pack(sign, exp, sig << shift, odd);
}

SF_INLINE void sf_norm_subnormal(sf_u64 frac, SF_THREAD int& exp, SF_THREAD sf_u64& sig) {
    int shift = sf_clz64(frac) - 11;
    exp = 1 - shift;
    sig = frac << shift;
}

SF_INLINE sf_u64 sf_add_mags(sf_u64 a, sf_u64 b, bool sz, bool odd = false) {
    int ea = SF_EXP(a), eb = SF_EXP(b);
    sf_u64 sa = SF_FRAC(a), sb = SF_FRAC(b);
    int d = ea - eb, ez;
    sf_u64 sig;
    if (!d) {
        if (!ea) return a + sb;
        if (ea == 0x7FF) return (sa | sb) ? sf_propagate_nan(a, b) : a;
        ez = ea;
        sig = (0x0020000000000000UL + sa + sb) << 9;
    } else {
        sa <<= 9;
        sb <<= 9;
        if (d < 0) {
            if (eb == 0x7FF) return sb ? sf_propagate_nan(a, b) : SF_PACK(sz, 0x7FF, 0);
            ez = eb;
            sa = ea ? sa + 0x2000000000000000UL : sa << 1;
            sa = sf_shift_right_jam64(sa, -d);
        } else {
            if (ea == 0x7FF) return sa ? sf_propagate_nan(a, b) : a;
            ez = ea;
            sb = eb ? sb + 0x2000000000000000UL : sb << 1;
            sb = sf_shift_right_jam64(sb, d);
        }
        sig = 0x2000000000000000UL + sa + sb;
        if (sig < 0x4000000000000000UL) { --ez; sig <<= 1; }
    }
    return sf_round_pack(sz, ez, sig, odd);
}

SF_INLINE sf_u64 sf_sub_mags(sf_u64 a, sf_u64 b, bool sz, bool odd = false) {
    int ea = SF_EXP(a), eb = SF_EXP(b);
    sf_u64 sa = SF_FRAC(a), sb = SF_FRAC(b);
    int d = ea - eb, ez;
    if (!d) {
        if (ea == 0x7FF) return (sa | sb) ? sf_propagate_nan(a, b) : SF_DEFAULT_NAN;
        sf_i64 diff = (sf_i64)(sa - sb);
        if (!diff) return SF_PACK(false, 0, 0);
        if (ea) --ea;
        if (diff < 0) { sz = !sz; diff = -diff; }
        int shift = sf_clz64((sf_u64)diff) - 11;
        ez = ea - shift;
        if (ez < 0) { shift = ea; ez = 0; }
        return SF_PACK(sz, ez, ((sf_u64)diff) << shift);
    }
    sa <<= 10;
    sb <<= 10;
    sf_u64 sig;
    if (d < 0) {
        sz = !sz;
        if (eb == 0x7FF) return sb ? sf_propagate_nan(a, b) : SF_PACK(sz, 0x7FF, 0);
        sa += ea ? 0x4000000000000000UL : sa;
        sa = sf_shift_right_jam64(sa, -d);
        sb |= 0x4000000000000000UL;
        ez = eb;
        sig = sb - sa;
    } else {
        if (ea == 0x7FF) return sa ? sf_propagate_nan(a, b) : a;
        sb += eb ? 0x4000000000000000UL : sb;
        sb = sf_shift_right_jam64(sb, d);
        sa |= 0x4000000000000000UL;
        ez = ea;
        sig = sa - sb;
    }
    return sf_norm_round_pack(sz, ez - 1, sig, odd);
}

SF_OUTLINE sf_u64 sf_add(sf_u64 a, sf_u64 b, bool odd = false) {
    bool sa = SF_SIGN(a);
    return (sa == SF_SIGN(b)) ? sf_add_mags(a, b, sa, odd) : sf_sub_mags(a, b, sa, odd);
}

SF_OUTLINE sf_u64 sf_sub(sf_u64 a, sf_u64 b, bool odd = false) {
    bool sa = SF_SIGN(a);
    return (sa == SF_SIGN(b)) ? sf_sub_mags(a, b, sa, odd) : sf_add_mags(a, b, sa, odd);
}

SF_OUTLINE sf_u64 sf_mul(sf_u64 a, sf_u64 b) {
    bool sz = SF_SIGN(a) ^ SF_SIGN(b);
    int ea = SF_EXP(a), eb = SF_EXP(b);
    sf_u64 sa = SF_FRAC(a), sb = SF_FRAC(b);
    if (ea == 0x7FF) {
        if (sa || (eb == 0x7FF && sb)) return sf_propagate_nan(a, b);
        return (eb | sb) ? SF_PACK(sz, 0x7FF, 0) : SF_DEFAULT_NAN;
    }
    if (eb == 0x7FF) {
        if (sb) return sf_propagate_nan(a, b);
        return (ea | sa) ? SF_PACK(sz, 0x7FF, 0) : SF_DEFAULT_NAN;
    }
    if (!ea) { if (!sa) return SF_PACK(sz, 0, 0); sf_norm_subnormal(sa, ea, sa); }
    if (!eb) { if (!sb) return SF_PACK(sz, 0, 0); sf_norm_subnormal(sb, eb, sb); }
    int ez = ea + eb - 0x3FF;
    sa = (sa | 0x0010000000000000UL) << 10;
    sb = (sb | 0x0010000000000000UL) << 11;
    sf_u64 hi = sf_mulhi64(sa, sb), lo = sa * sb;
    sf_u64 sig = hi | (sf_u64)(lo != 0);
    if (sig < 0x4000000000000000UL) { --ez; sig <<= 1; }
    return sf_round_pack(sz, ez, sig);
}

SF_OUTLINE sf_u64 sf_div(sf_u64 a, sf_u64 b, bool odd = false) {
    bool sz = SF_SIGN(a) ^ SF_SIGN(b);
    int ea = SF_EXP(a), eb = SF_EXP(b);
    sf_u64 sa = SF_FRAC(a), sb = SF_FRAC(b);
    if (ea == 0x7FF) {
        if (sa) return sf_propagate_nan(a, b);
        if (eb == 0x7FF) return sb ? sf_propagate_nan(a, b) : SF_DEFAULT_NAN;
        return SF_PACK(sz, 0x7FF, 0);
    }
    if (eb == 0x7FF) return sb ? sf_propagate_nan(a, b) : SF_PACK(sz, 0, 0);
    if (!eb) {
        if (!sb) return (ea | sa) ? SF_PACK(sz, 0x7FF, 0) : SF_DEFAULT_NAN;
        sf_norm_subnormal(sb, eb, sb);
    }
    if (!ea) { if (!sa) return SF_PACK(sz, 0, 0); sf_norm_subnormal(sa, ea, sa); }
    int ez = ea - eb + 0x3FE;
    sa |= 0x0010000000000000UL;
    sb |= 0x0010000000000000UL;
    if (sa < sb) { --ez; sa <<= 1; }
    // restoring division: q = floor(sa * 2^62 / sb), quotient in [2^62, 2^63)
    sf_u64 rem = sa, q = 0;
    for (int i = 0; i < 63; ++i) {
        q <<= 1;
        if (rem >= sb) { rem -= sb; q |= 1; }
        rem <<= 1;
    }
    q |= (sf_u64)(rem != 0);
    return sf_round_pack(sz, ez, q, odd);
}

SF_OUTLINE sf_u64 sf_sqrt(sf_u64 a, bool odd = false) {
    bool s = SF_SIGN(a);
    int e = SF_EXP(a);
    sf_u64 f = SF_FRAC(a);
    if (e == 0x7FF) {
        if (f) return sf_propagate_nan(a, 0);
        return s ? SF_DEFAULT_NAN : a;
    }
    if (s) return (e | f) ? SF_DEFAULT_NAN : a;
    if (!e) { if (!f) return a; sf_norm_subnormal(f, e, f); }
    int unb = e - 0x3FF;
    int ez = (unb >> 1) + 0x3FE;   // arithmetic shift: floor(unb / 2)
    sf_u64 sig = f | 0x0010000000000000UL;
    if (unb & 1) sig <<= 1;
    // r = floor(sqrt(sig << 56)), 55 significant bits; radicand bits 0..109
    sf_u64 nhi = sig >> 8, nlo = sig << 56;
    sf_u64 r = 0, rem = 0;
    for (int i = 54; i >= 0; --i) {
        int sh = 2 * i;
        sf_u64 pair = (sh >= 64) ? ((nhi >> (sh - 64)) & 3) : (((nlo >> sh) | (sh ? (nhi << (64 - sh)) : 0)) & 3);
        rem = (rem << 2) | pair;
        sf_u64 trial = (r << 2) | 1;
        if (rem >= trial) { rem -= trial; r = (r << 1) | 1; } else { r <<= 1; }
    }
    sf_u64 z = (r << 8) | (sf_u64)(rem != 0);
    return sf_round_pack(false, ez, z, odd);
}

// ---- 128-bit helpers for fma
SF_INLINE sf_u128 sf_mul64x64(sf_u64 a, sf_u64 b) { sf_u128 r; r.hi = sf_mulhi64(a, b); r.lo = a * b; return r; }
SF_INLINE int sf_clz128(sf_u128 a) { return a.hi ? sf_clz64(a.hi) : 64 + sf_clz64(a.lo); }
SF_INLINE sf_u128 sf_shl128(sf_u128 a, int n) {
    sf_u128 r;
    if (n == 0) return a;
    if (n >= 128) { r.hi = 0; r.lo = 0; return r; }
    if (n >= 64) { r.hi = a.lo << (n - 64); r.lo = 0; return r; }
    r.hi = (a.hi << n) | (a.lo >> (64 - n));
    r.lo = a.lo << n;
    return r;
}
SF_INLINE sf_u128 sf_shr_jam128(sf_u128 a, int n) {
    sf_u128 r;
    if (n <= 0) return a;
    if (n >= 128) { r.hi = 0; r.lo = (sf_u64)((a.hi | a.lo) != 0); return r; }
    if (n >= 64) {
        sf_u64 lost = a.lo | ((n > 64) ? (a.hi << (128 - n)) : 0);
        r.lo = (n == 64 ? a.hi : (a.hi >> (n - 64))) | (sf_u64)(lost != 0);
        r.hi = 0;
        return r;
    }
    sf_u64 lost = a.lo << (64 - n);
    r.lo = (a.lo >> n) | (a.hi << (64 - n)) | (sf_u64)(lost != 0);
    r.hi = a.hi >> n;
    return r;
}
SF_INLINE bool sf_lt128(sf_u128 a, sf_u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
SF_INLINE sf_u128 sf_add128(sf_u128 a, sf_u128 b) { sf_u128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (sf_u64)(r.lo < a.lo); return r; }
SF_INLINE sf_u128 sf_sub128(sf_u128 a, sf_u128 b) { sf_u128 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (sf_u64)(a.lo < b.lo); return r; }

// a*b + c with a single rounding.
SF_OUTLINE sf_u64 sf_fma(sf_u64 a, sf_u64 b, sf_u64 c, bool odd = false) {
    bool sa = SF_SIGN(a), sb = SF_SIGN(b), sc = SF_SIGN(c), sp = sa ^ sb;
    int ea = SF_EXP(a), eb = SF_EXP(b), ec = SF_EXP(c);
    sf_u64 fa = SF_FRAC(a), fb = SF_FRAC(b), fc = SF_FRAC(c);
    if (SF_ISNAN(a) || SF_ISNAN(b)) return sf_propagate_nan(SF_ISNAN(a) ? a : b, 0);
    bool a_inf = ea == 0x7FF, b_inf = eb == 0x7FF;
    bool a_zero = !ea && !fa, b_zero = !eb && !fb;
    if (a_inf || b_inf) {
        if ((a_inf && b_zero) || (b_inf && a_zero)) return SF_DEFAULT_NAN;
        if (SF_ISNAN(c)) return sf_propagate_nan(c, 0);
        if (ec == 0x7FF && sc != sp) return SF_DEFAULT_NAN;
        return SF_PACK(sp, 0x7FF, 0);
    }
    if (SF_ISNAN(c)) return sf_propagate_nan(c, 0);
    if (ec == 0x7FF) return c;
    if (a_zero || b_zero) {
        bool c_zero = !ec && !fc;
        if (c_zero) return SF_PACK(sp && sc, 0, 0);
        return c;
    }
    // value = m * 2^E with m an integer significand
    int Ea, Eb, Ec;
    sf_u64 ma, mb, mc;
    if (ea) { ma = fa | 0x0010000000000000UL; Ea = ea - 1075; } else { int e; sf_norm_subnormal(fa, e, ma); Ea = e - 1075; }
    if (eb) { mb = fb | 0x0010000000000000UL; Eb = eb - 1075; } else { int e; sf_norm_subnormal(fb, e, mb); Eb = e - 1075; }
    sf_u128 P = sf_mul64x64(ma, mb);        // 105..106 bits
    int Pe = Ea + Eb;
    bool c_zero = !ec && !fc;
    if (c_zero) {
        // exact product, rounded once
        int p = 127 - sf_clz128(P);
        sf_u64 sig;
        if (p > 62) { sf_u128 t = sf_shr_jam128(P, p - 62); sig = t.lo; } else { sig = P.lo << (62 - p); }
        return sf_round_pack(sp, Pe + p + 1022, sig, odd);
    }
    if (ec) { mc = fc | 0x0010000000000000UL; Ec = ec - 1075; } else { mc = fc; Ec = 1 - 1075; }
    sf_u128 C; C.hi = 0; C.lo = mc;
    int pmsb = 127 - sf_clz128(P), cmsb = 63 - sf_clz64(mc);
    int top_p = Pe + pmsb, top_c = Ec + cmsb;
    int top = top_p > top_c ? top_p : top_c;
    // frame: the larger operand's leading bit at bit 125; LSB weight 2^(top - 125)
    int lsb = top - 125;
    sf_u128 X = (Pe >= lsb) ? sf_shl128(P, Pe - lsb) : sf_shr_jam128(P, lsb - Pe);
    sf_u128 Y = (Ec >= lsb) ? sf_shl128(C, Ec - lsb) : sf_shr_jam128(C, lsb - Ec);
    sf_u128 S;
    bool sz;
    if (sp == sc) { S = sf_add128(X, Y); sz = sp; }
    else if (sf_lt128(X, Y)) { S = sf_sub128(Y, X); sz = sc; }
    else { S = sf_sub128(X, Y); sz = sp; }
    if (!S.hi && !S.lo) return SF_PACK(false, 0, 0);
    int p = 127 - sf_clz128(S);
    sf_u64 sig;
    if (p > 62) { sf_u128 t = sf_shr_jam128(S, p - 62); sig = t.lo; } else { sig = S.lo << (62 - p); }
    return sf_round_pack(sz, lsb + p + 1022, sig, odd);
}

// ---- comparisons (IEEE: NaN unordered, +0 == -0)
SF_INLINE bool sf_eq(sf_u64 a, sf_u64 b) {
    if (SF_ISNAN(a) || SF_ISNAN(b)) return false;
    return a == b || ((a | b) << 1) == 0;
}
SF_INLINE bool sf_lt(sf_u64 a, sf_u64 b) {
    if (SF_ISNAN(a) || SF_ISNAN(b)) return false;
    bool sa = SF_SIGN(a), sb = SF_SIGN(b);
    if (sa != sb) return sa && ((a | b) << 1) != 0;
    return (a != b) && (sa ^ (a < b));
}
SF_INLINE bool sf_le(sf_u64 a, sf_u64 b) {
    if (SF_ISNAN(a) || SF_ISNAN(b)) return false;
    bool sa = SF_SIGN(a), sb = SF_SIGN(b);
    if (sa != sb) return sa || ((a | b) << 1) == 0;
    return (a == b) || (sa ^ (a < b));
}

// ---- conversions
SF_OUTLINE sf_u64 sf_from_f32(float f) {
    sf_u32 u = sf_f32_bits(f);
    bool s = u >> 31;
    int e = (int)((u >> 23) & 0xFF);
    sf_u64 m = u & 0x7FFFFF;
    if (e == 0xFF) {
        if (m) return SF_PACK(s, 0x7FF, (m << 29) | 0x0008000000000000UL);
        return SF_PACK(s, 0x7FF, 0);
    }
    if (!e) {
        if (!m) return SF_PACK(s, 0, 0);
        int shift = sf_clz64(m) - 40;   // put leading bit at bit 23
        m <<= shift;
        e = 1 - shift;
    }
    return SF_PACK(s, e + 0x380, (m & 0x7FFFFF) << 29);
}

// SoftFloat roundPackToF32 (round-to-nearest-even); sig leading bit at bit 30
SF_INLINE float sf_round_pack_f32(bool s, int e, sf_u32 sig) {
    sf_u32 rb = sig & 0x7F;
    if ((sf_u32)e >= 0xFDu) {
        if (e < 0) {
            int d = -e;
            sig = (d < 31) ? ((sig >> d) | (sf_u32)((sig << ((-d) & 31)) != 0)) : (sf_u32)(sig != 0);
            e = 0;
            rb = sig & 0x7F;
        } else if (e > 0xFD || sig + 0x40 >= 0x80000000u) {
            return sf_bits_f32(((sf_u32)s << 31) | 0x7F800000u);
        }
    }
    sig = (sig + 0x40) >> 7;
    sig &= ~(sf_u32)((rb ^ 0x40) == 0);
    if (!sig) e = 0;
    return sf_bits_f32(((sf_u32)s << 31) + ((sf_u32)e << 23) + sig);
}

SF_OUTLINE float sf_to_f32(sf_u64 a) {
    bool s = SF_SIGN(a);
    int e = SF_EXP(a);
    sf_u64 f = SF_FRAC(a);
    if (e == 0x7FF) {
        if (f) return sf_bits_f32(((sf_u32)s << 31) | 0x7FC00000u | (sf_u32)(f >> 29));
        return sf_bits_f32(((sf_u32)s << 31) | 0x7F800000u);
    }
    sf_u32 sig = (sf_u32)sf_shift_right_jam64(f, 22);   // 30 bits
    if (!(e | sig)) return sf_bits_f32((sf_u32)s << 31);
    return sf_round_pack_f32(s, e - 0x381, sig | 0x40000000u);
}

SF_INLINE sf_u64 sf_from_u64(sf_u64 a) {
    if (!a) return 0;
    if (a & 0x8000000000000000UL)
        return sf_round_pack(false, 0x43D, sf_shift_right_jam64(a, 1));
    return sf_norm_round_pack(false, 0x43C, a);
}
SF_INLINE sf_u64 sf_from_i64(sf_i64 a) {
    bool s = a < 0;
    sf_u64 m = s ? (sf_u64)0 - (sf_u64)a : (sf_u64)a;
    sf_u64 r = sf_from_u64(m);
    return s ? (r | 0x8000000000000000UL) : r;
}

// Truncation toward zero with CUDA's saturating semantics (cvt.rzi): NaN -> 0.
SF_INLINE sf_i64 sf_to_i64_rz(sf_u64 a) {
    bool s = SF_SIGN(a);
    int e = SF_EXP(a);
    sf_u64 f = SF_FRAC(a);
    if (e == 0x7FF && f) return 0;
    if (e < 0x3FF) return 0;
    if (e >= 0x3FF + 63) return s ? (sf_i64)0x8000000000000000UL : (sf_i64)0x7FFFFFFFFFFFFFFFUL;
    sf_u64 m = f | 0x0010000000000000UL;
    int sh = e - 0x433;
    sf_u64 v = sh >= 0 ? (m << sh) : (m >> (-sh));
    return s ? -(sf_i64)v : (sf_i64)v;
}
SF_INLINE sf_u64 sf_to_u64_rz(sf_u64 a) {
    bool s = SF_SIGN(a);
    int e = SF_EXP(a);
    sf_u64 f = SF_FRAC(a);
    if (e == 0x7FF && f) return 0;
    if (s || e < 0x3FF) return 0;
    if (e >= 0x3FF + 64) return 0xFFFFFFFFFFFFFFFFUL;
    sf_u64 m = f | 0x0010000000000000UL;
    int sh = e - 0x433;
    return sh >= 0 ? (m << sh) : (m >> (-sh));
}
SF_INLINE int sf_to_i32_rz(sf_u64 a) {
    sf_i64 v = sf_to_i64_rz(a);
    if (v > 2147483647) return 2147483647;
    if (v < -2147483647 - 1) return -2147483647 - 1;
    return (int)v;
}
SF_INLINE sf_u32 sf_to_u32_rz(sf_u64 a) {
    sf_u64 v = sf_to_u64_rz(a);
    return v > 0xFFFFFFFFUL ? 0xFFFFFFFFu : (sf_u32)v;
}

// ---- exact integral rounding (results are representable, no rounding error)
SF_INLINE sf_u64 sf_trunc(sf_u64 a) {
    int e = SF_EXP(a);
    if (e >= 0x433) return (e == 0x7FF && SF_FRAC(a)) ? (a | 0x0008000000000000UL) : a;
    if (e < 0x3FF) return a & 0x8000000000000000UL;
    sf_u64 mask = (1UL << (0x433 - e)) - 1;
    return a & ~mask;
}
SF_INLINE sf_u64 sf_floor(sf_u64 a) {
    int e = SF_EXP(a);
    if (e >= 0x433) return (e == 0x7FF && SF_FRAC(a)) ? (a | 0x0008000000000000UL) : a;
    bool s = SF_SIGN(a);
    if (e < 0x3FF) {
        if (!s || !((a << 1))) return a & 0x8000000000000000UL ? a : (a & 0x8000000000000000UL);
        return 0xBFF0000000000000UL;     // -1
    }
    sf_u64 mask = (1UL << (0x433 - e)) - 1;
    if (!(a & mask)) return a;
    sf_u64 t = a & ~mask;
    return s ? sf_sub(t, 0x3FF0000000000000UL) : t;
}
SF_INLINE sf_u64 sf_ceil(sf_u64 a) {
    int e = SF_EXP(a);
    if (e >= 0x433) return (e == 0x7FF && SF_FRAC(a)) ? (a | 0x0008000000000000UL) : a;
    bool s = SF_SIGN(a);
    if (e < 0x3FF) {
        if (s || !((a << 1))) return s ? 0x8000000000000000UL : a;
        return 0x3FF0000000000000UL;
    }
    sf_u64 mask = (1UL << (0x433 - e)) - 1;
    if (!(a & mask)) return a;
    sf_u64 t = a & ~mask;
    return s ? t : sf_add(t, 0x3FF0000000000000UL);
}
// round half away from zero (C round)
SF_INLINE sf_u64 sf_round(sf_u64 a) {
    int e = SF_EXP(a);
    if (e >= 0x433) return (e == 0x7FF && SF_FRAC(a)) ? (a | 0x0008000000000000UL) : a;
    bool s = SF_SIGN(a);
    if (e < 0x3FE) return a & 0x8000000000000000UL;
    if (e == 0x3FE) return s ? 0xBFF0000000000000UL : 0x3FF0000000000000UL;
    sf_u64 halfbit = 1UL << (0x432 - e);
    sf_u64 mask = (1UL << (0x433 - e)) - 1;
    sf_u64 t = (a + halfbit) & ~mask;   // carry into the exponent is exact
    return t;
}
// round half to even (rint / nearbyint in the default rounding mode)
SF_INLINE sf_u64 sf_rint(sf_u64 a) {
    int e = SF_EXP(a);
    if (e >= 0x433) return (e == 0x7FF && SF_FRAC(a)) ? (a | 0x0008000000000000UL) : a;
    bool s = SF_SIGN(a);
    if (e < 0x3FE) return a & 0x8000000000000000UL;
    // add and subtract 2^52 with the sign of a: exact RNE to an integer
    sf_u64 big = s ? 0xC330000000000000UL : 0x4330000000000000UL;
    sf_u64 r = sf_sub(sf_add(a, big), big);
    return (r & 0x7FFFFFFFFFFFFFFFUL) ? r : (r | (a & 0x8000000000000000UL));
}


// ---- exact binary32 arithmetic with gradual underflow (for CUDA --ftz=false code)
// Apple GPUs flush FP32 subnormal operands and results; these compute the
// correctly rounded IEEE binary32 result through binary64 rounded to odd.
SF_OUTLINE float sf32_add(float a, float b) { return sf_to_f32(sf_add(sf_from_f32(a), sf_from_f32(b), true)); }
SF_OUTLINE float sf32_sub(float a, float b) { return sf_to_f32(sf_sub(sf_from_f32(a), sf_from_f32(b), true)); }
SF_OUTLINE float sf32_mul(float a, float b) { return sf_to_f32(sf_mul(sf_from_f32(a), sf_from_f32(b))); }   // exact in binary64
SF_OUTLINE float sf32_div(float a, float b) { return sf_to_f32(sf_div(sf_from_f32(a), sf_from_f32(b), true)); }
SF_OUTLINE float sf32_sqrt(float a) { return sf_to_f32(sf_sqrt(sf_from_f32(a), true)); }
SF_OUTLINE float sf32_fma(float a, float b, float c) { return sf_to_f32(sf_fma(sf_from_f32(a), sf_from_f32(b), sf_from_f32(c), true)); }
