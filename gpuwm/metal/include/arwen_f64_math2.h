// arwen_f64_math2.h -- further binary64 functions on __f64 and their binary32 forms.
//
// expm1/log1p use Kahan's correction (u = exp(x): (u-1) x / log(u)), which keeps
// them within a few ulp given exp/log of arwen_f64_math.h; tanh, sinh, asinh,
// acosh, atanh are built on them; asin/acos on atan2; erf on its all-positive
// series 2/sqrt(pi) e^-x^2 sum 2^n x^(2n+1)/(2n+1)!! (no cancellation), erfc on
// the continued fraction (modified Lentz) for x >= 0.75 with e^-x^2 corrected
// by the exact error of x*x; tgamma on Stirling's series after shifting to z >= 16
// with reflection.  Measured against the host libm in tests/metal/f64_math2_host_check.cpp
// (declared tolerances there); CUDA's libdevice is a different few-ulp implementation.
#pragma once

AF_INLINE __f64 expm1(__f64 x) {
    if (isnan(x)) return x;
    if (x > AF_C(0x40862E42FEFA39EF)) return AF_INF;
    if (x < AF_C(0xC044000000000000)) return -AF_ONE;
    if (fabs(x) < AF_C(0x3C90000000000000)) return x;
    __f64 u = exp(x);
    if (u == AF_ONE) return x;
    __f64 um1 = u - AF_ONE;
    if (um1 == -AF_ONE) return -AF_ONE;
    return um1 * (x / log(u));                       // x / log(u) ~ 1: no intermediate overflow
}
AF_INLINE __f64 log1p(__f64 x) {
    if (isnan(x)) return x;
    if (x == -AF_ONE) return -AF_INF;
    if (x < -AF_ONE) return AF_NAN;
    if (isinf(x)) return x;
    __f64 u = AF_ONE + x;
    if (u == AF_ONE) return x;
    return log(u) * x / (u - AF_ONE);
}
AF_INLINE __f64 tanh(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a > AF_C(0x4036000000000000)) return copysign(AF_ONE, x);
    if (a < AF_C(0x3E30000000000000)) return x;
    __f64 e = expm1(a + a);
    return copysign(e / (e + AF_TWO), x);
}
AF_INLINE __f64 sinh(__f64 x) {
    if (isnan(x) || isinf(x)) return x;
    __f64 a = fabs(x);
    if (a < AF_C(0x3E30000000000000)) return x;
    if (a < AF_C(0x4036000000000000)) { __f64 e = expm1(a); return copysign(AF_HALF * (e + e / (e + AF_ONE)), x); }
    if (a < AF_C(0x40862E42FEFA39EF)) return copysign(AF_HALF * exp(a), x);
    __f64 w = exp(AF_HALF * a);
    return copysign((AF_HALF * w) * w, x);
}
AF_INLINE __f64 cosh(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a < AF_C(0x4036000000000000)) { __f64 e = exp(a); return AF_HALF * (e + AF_ONE / e); }
    if (a < AF_C(0x40862E42FEFA39EF)) return AF_HALF * exp(a);
    __f64 w = exp(AF_HALF * a);
    return (AF_HALF * w) * w;
}
AF_INLINE __f64 asin(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a > AF_ONE) return AF_NAN;
    if (a < AF_C(0x3E30000000000000)) return x;
    return atan2(x, sqrt((AF_ONE - a) * (AF_ONE + a)));
}
AF_INLINE __f64 acos(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a > AF_ONE) return AF_NAN;
    return atan2(sqrt((AF_ONE - a) * (AF_ONE + a)), x);
}
AF_INLINE __f64 asinh(__f64 x) {
    if (isnan(x) || isinf(x)) return x;
    __f64 a = fabs(x), r;
    if (a < AF_C(0x3E30000000000000)) return x;
    if (a > AF_C(0x41B0000000000000)) r = log(a) + AF_C(0x3FE62E42FEFA39EF);
    else if (a > AF_TWO) r = log(AF_TWO * a + AF_ONE / (sqrt(a * a + AF_ONE) + a));
    else r = log1p(a + a * a / (AF_ONE + sqrt(AF_ONE + a * a)));
    return copysign(r, x);
}
AF_INLINE __f64 acosh(__f64 x) {
    if (isnan(x)) return x;
    if (x < AF_ONE) return AF_NAN;
    if (isinf(x)) return x;
    if (x > AF_C(0x41B0000000000000)) return log(x) + AF_C(0x3FE62E42FEFA39EF);
    if (x > AF_TWO) return log(AF_TWO * x - AF_ONE / (x + sqrt(x * x - AF_ONE)));
    __f64 t = x - AF_ONE;
    return log1p(t + sqrt(AF_TWO * t + t * t));
}
AF_INLINE __f64 atanh(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a > AF_ONE) return AF_NAN;
    if (a == AF_ONE) return copysign(AF_INF, x);
    if (a < AF_C(0x3E30000000000000)) return x;
    __f64 r = (a < AF_HALF) ? AF_HALF * log1p(a + a + (a + a) * a / (AF_ONE - a))
                            : AF_HALF * log1p((a + a) / (AF_ONE - a));
    return copysign(r, x);
}
// e^(-x^2) with the rounding error of x*x carried: e^-(s+d) = e^-s (1 - d)
AF_INLINE __f64 af_exp_mx2(__f64 x) {
    __f64 s = x * x;
    __f64 d = __fma_rn(x, x, -s);
    return exp(-s) * (AF_ONE - d);
}
AF_INLINE __f64 af_erf_series(__f64 a) {            // 0 <= a < 2.5
    __f64 t = a, sum = a, x2 = a * a;
    for (int n = 1; n < 200; ++n) {
        t = t * (x2 + x2) / af_i(2 * n + 1);
        sum = sum + t;
        if (t < sum * AF_C(0x3C70000000000000)) break;
    }
    return AF_C(0x3FF20DD750429B6D) * af_exp_mx2(a) * sum;
}
AF_INLINE __f64 af_erfc_cf(__f64 a) {               // a >= 0.75: Lentz on erfc = e^-a^2/sqrt(pi) * 1/(a + (1/2)/(a + 1/(a + (3/2)/(a + ...))))
    const __f64 tiny = AF_C(0x01A56E1FC2F8F359);
    __f64 f = a, C = a, D = AF_ZERO;
    for (int n = 1; n < 5000; ++n) {
        __f64 an = af_i(n) * AF_HALF;
        D = a + an * D; if (fabs(D) < tiny) D = tiny;
        C = a + an / C; if (fabs(C) < tiny) C = tiny;
        D = AF_ONE / D;
        __f64 delta = C * D;
        f = f * delta;
        if (fabs(delta - AF_ONE) < AF_C(0x3C80000000000000)) break;
    }
    return AF_C(0x3FE20DD750429B6D) * af_exp_mx2(a) / f;
}
AF_INLINE __f64 erf(__f64 x) {
    if (isnan(x)) return x;
    __f64 a = fabs(x);
    if (a < AF_C(0x3E30000000000000)) return x * AF_C(0x3FF20DD750429B6D);
    if (a >= AF_C(0x4018000000000000)) return copysign(AF_ONE, x);
    __f64 r = (a < AF_C(0x4004000000000000)) ? af_erf_series(a) : AF_ONE - af_erfc_cf(a);
    return copysign(r, x);
}
AF_INLINE __f64 erfc(__f64 x) {
    if (isnan(x)) return x;
    if (x < AF_ZERO) {
        __f64 a = -x;
        if (a >= AF_C(0x4018000000000000)) return AF_TWO;
        return AF_ONE + ((a < AF_C(0x4004000000000000)) ? af_erf_series(a) : AF_ONE - af_erfc_cf(a));
    }
    if (x > AF_C(0x403B4CCCCCCCCCCD)) return AF_ZERO;
    if (x < AF_C(0x3FE8000000000000)) return AF_ONE - af_erf_series(x);
    return af_erfc_cf(x);
}
// Gamma(z), z >= 16: sqrt(2 pi) z^(z-1/2) e^-z e^S(z), S the Stirling series to z^-15
AF_INLINE __f64 af_gamma_stirling(__f64 z) {
    const __f64 c1 = AF_C(0x3FB5555555555555), c2 = AF_C(0xBF66C16C16C16C17), c3 = AF_C(0x3F4A01A01A01A01A),
                c4 = AF_C(0xBF43813813813814), c5 = AF_C(0x3F4B951E2B18FF23), c6 = AF_C(0xBF5F6AB0D9993C7D),
                c7 = AF_C(0x3F7A41A41A41A41A), c8 = AF_C(0xBF9E4286CB0F5398);
    __f64 iz = AF_ONE / z, iz2 = iz * iz;
    __f64 ser = iz * (c1 + iz2 * (c2 + iz2 * (c3 + iz2 * (c4 + iz2 * (c5 + iz2 * (c6 + iz2 * (c7 + iz2 * c8)))))));
    __f64 h = pow(z, AF_HALF * (z - AF_HALF));      // z^(z-1/2) in two halves: no early overflow
    return AF_C(0x40040D931FF62705) * h * (h * exp(-z)) * exp(ser);
}
// x > 0: shift to z = x + n >= 16 and divide by x (x+1) ... (x+n-1)
// The shifted arguments x + k are carried exactly as zh + zl (two-sum): the product
// gets the first-order relative correction sum zl/zh, and Gamma(zh + zl) =
// Gamma(zh) (1 + psi(zh) zl), psi(z) ~ log z - 1/(2z) - 1/(12 z^2).
AF_INLINE __f64 af_gamma_pos(__f64 x, __f64 xl = __f64{0}) {
    __f64 prod = AF_ONE, zh = x, zl = xl, rel = AF_ZERO;
    while (zh < AF_C(0x4030000000000000)) {
        prod = prod * zh;
        rel = rel + zl / zh;
        __f64 sum = zh + AF_ONE, bb = sum - zh;
        __f64 err = (zh - (sum - bb)) + (AF_ONE - bb);
        zh = sum;
        zl = zl + err;
    }
    __f64 psi = log(zh) - AF_HALF / zh - AF_ONE / (AF_C(0x4028000000000000) * zh * zh);
    return af_gamma_stirling(zh) * (AF_ONE + psi * zl) / (prod * (AF_ONE + rel));
}
AF_INLINE __f64 tgamma(__f64 x) {
    if (isnan(x)) return x;
    if (x == AF_ZERO) return copysign(AF_INF, x);
    if (isinf(x)) return x > AF_ZERO ? x : AF_NAN;
    if (x < AF_ZERO && floor(x) == x) return AF_NAN;
    if (x > AF_C(0x406573FAE561F647)) return AF_INF;
    if (x == floor(x) && x <= AF_C(0x4037000000000000)) {            // exact products while they fit
        __f64 r = AF_ONE;
        for (__f64 k = AF_TWO; k < x; k = k + AF_ONE) r = r * k;
        return r;
    }
    if (x > AF_ZERO) return af_gamma_pos(x);
    if (x < AF_C(0xC067200000000000)) return copysign(AF_ZERO, sin(AF_C(0x400921FB54442D18) * x));
    // reflection: Gamma(x) = pi / (sin(pi x) Gamma(1 - x)); r = x - rint(x) is exact,
    // sin(pi x) = (-1)^rint(x) sin(pi r); 1 - x carried exactly as h + l
    const __f64 pi = AF_C(0x400921FB54442D18);
    __f64 n = rint(x), r = x - n;
    __f64 s = sin(pi * r);
    if (__arwen_cvt<long>(n) & 1) s = -s;
    __f64 h = AF_ONE - x, bb = h - AF_ONE;
    __f64 l = (AF_ONE - (h - bb)) + (-x - bb);
    return pi / (s * af_gamma_pos(h, l));
}
AF_INLINE __f64 rsqrt(__f64 x) { return AF_ONE / sqrt(x); }
AF_INLINE __f64 scalbn(__f64 x, int n) { return ldexp(x, n); }
AF_INLINE __f64 nextafter(__f64 x, __f64 y) {
    if (isnan(x) || isnan(y)) return x + y;
    if (x == y) return y;
    if (x == AF_ZERO) return copysign(__f64{(sf_u64)1}, y);
    bool up = (y > x) == (x > AF_ZERO);
    return __f64{up ? x.v + 1 : x.v - 1};
}
AF_INLINE __f64 remainder(__f64 x, __f64 y) {      // fdlibm e_remainder.c
    if (isnan(x) || isnan(y) || isinf(x) || y == AF_ZERO) return AF_NAN;
    if (isinf(y)) return x;
    __f64 p = fabs(y);
    bool neg = signbit(x);
    __f64 r = (p < AF_C(0x7FD0000000000000)) ? fmod(x, p + p) : x;   // exact, |r| < 2p
    r = fabs(r);
    if (p < AF_C(0x0020000000000000)) {               // tiny p: compare with r + r
        if (r + r > p) { r = r - p; if (r + r >= p) r = r - p; }
    } else {
        __f64 h = AF_HALF * p;
        if (r > h) { r = r - p; if (r >= h) r = r - p; }
    }
    return neg ? -r : r;
}

#ifdef __METAL_VERSION__
// binary32 forms, evaluated in binary64 and rounded once
inline float erff(float x) { return __double2float_rn(erf(__arwen_cvt<__f64>(x))); }
inline float erfcf(float x) { return __double2float_rn(erfc(__arwen_cvt<__f64>(x))); }
inline float tgammaf(float x) { return __double2float_rn(tgamma(__arwen_cvt<__f64>(x))); }
inline float lgammaf(float x) { return __double2float_rn(lgamma(__arwen_cvt<__f64>(x))); }
inline float erf(float x) { return erff(x); }
inline float erfc(float x) { return erfcf(x); }
inline float tgamma(float x) { return tgammaf(x); }
inline float lgamma(float x) { return lgammaf(x); }
inline float remainderf(float x, float y) { return __double2float_rn(remainder(__arwen_cvt<__f64>(x), __arwen_cvt<__f64>(y))); }
inline float remainder(float x, float y) { return remainderf(x, y); }
inline float nextafterf(float x, float y) {
    if (metal::isnan(x) || metal::isnan(y)) return x + y;
    if (x == y) return y;
    if (x == 0.0f) return as_type<float>((as_type<uint>(y) & 0x80000000u) | 1u);
    bool up = (y > x) == (x > 0.0f);
    return as_type<float>(up ? as_type<uint>(x) + 1u : as_type<uint>(x) - 1u);
}
inline float scalbnf(float x, int n) { return metal::ldexp(x, n); }
inline long llround(float x) { return (long)metal::round(x); }

// directed conversions from binary64
inline float __double2float_rz(__f64 a) {
    float r = sf_to_f32(a.v);
    if (metal::isfinite(r) && sf_lt(sf_from_f32(metal::fabs(r)), (fabs(a)).v)) return r;        // |r| < |a|: already toward zero
    if (metal::isinf(r) && isfinite(a)) return metal::copysign(as_type<float>(0x7f7fffffu), signbit(a) ? -1.0f : 1.0f);
    if (metal::isfinite(r) && sf_lt((fabs(a)).v, sf_from_f32(metal::fabs(r))))
        return as_type<float>(as_type<uint>(r) - 1u);  // |r| > |a|: one step toward zero
    return r;
}
inline float __double2float_ru(__f64 a) {
    float r = sf_to_f32(a.v);
    if (metal::isfinite(r) && sf_lt(sf_from_f32(r), a.v)) return nextafterf(r, INFINITY);
    return r;
}
inline float __double2float_rd(__f64 a) {
    float r = sf_to_f32(a.v);
    if (metal::isfinite(r) && sf_lt(a.v, sf_from_f32(r))) return nextafterf(r, -INFINITY);
    return r;
}
inline int __double2int_ru(__f64 a) { return sf_to_i32_rz(sf_ceil(a.v)); }
inline int __double2int_rd(__f64 a) { return sf_to_i32_rz(sf_floor(a.v)); }
#endif  // __METAL_VERSION__
