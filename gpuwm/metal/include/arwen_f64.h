// arwen_f64.h -- the ``double`` of CUDA kernels on Metal.
//
// The translator spells every CUDA ``double`` as ``__f64`` (values) or
// ``__f64m`` (objects in device/constant memory, always IEEE binary64 bits so
// a float64 array means the same thing to the host and the GPU), and turns
// every conversion involving double -- implicit or written -- into
// ``__arwen_cvt<To>(x)``.  ``__f64`` is therefore a plain aggregate with no
// constructors or member functions: it can live in any address space,
// including threadgroup arrays and ``constant`` tables initialised from
// __F64_LIT bit patterns.
//
// ARWEN_F64_MODE_SF64  binary64 emulated exactly on 64-bit integers
//                      (arwen_sf64.h; bit-identical to CUDA's _rn intrinsics)
// ARWEN_F64_MODE_DF64  float-float arithmetic (~48-bit significand, float range)
// ARWEN_F64_MODE_FP32  demoted to float
#pragma once
#include "arwen_sf64.h"

#if defined(ARWEN_F64_MODE_SF64)

struct __f64 { ulong v; };
typedef __f64 __f64m;
#define __F64_LIT(bits, hi, lo) (__f64{bits})
#define __F64M_LIT(bits, hi, lo) (__f64{bits})
inline __f64 __f64_from_bits(ulong b) { return __f64{b}; }
inline ulong __f64_bits(__f64 x) { return x.v; }

// ---- conversions (every one the translator emits goes through here)
template <typename To, typename From> struct __arwen_cvt_impl { static To f(From x) { return static_cast<To>(x); } };
template <typename To, typename From> inline To __arwen_cvt(From x) { return __arwen_cvt_impl<To, From>::f(x); }
#define ARWEN_CVT(To, From, expr) \
    template <> struct __arwen_cvt_impl<To, From> { static To f(From x) { return expr; } };
ARWEN_CVT(__f64, __f64, x)
ARWEN_CVT(__f64, float, __f64{sf_from_f32(x)})
#ifdef __METAL_VERSION__
ARWEN_CVT(__f64, half, __f64{sf_from_f32((float)x)})
ARWEN_CVT(half, __f64, (half)sf_to_f32(x.v))
#endif
ARWEN_CVT(__f64, int, __f64{sf_from_i64((long)x)})
ARWEN_CVT(__f64, uint, __f64{sf_from_u64((ulong)x)})
ARWEN_CVT(__f64, short, __f64{sf_from_i64((long)x)})
ARWEN_CVT(__f64, ushort, __f64{sf_from_u64((ulong)x)})
ARWEN_CVT(__f64, char, __f64{sf_from_i64((long)x)})
ARWEN_CVT(__f64, uchar, __f64{sf_from_u64((ulong)x)})
ARWEN_CVT(__f64, long, __f64{sf_from_i64(x)})
ARWEN_CVT(__f64, ulong, __f64{sf_from_u64(x)})
ARWEN_CVT(__f64, bool, __f64{x ? 0x3FF0000000000000UL : 0UL})
ARWEN_CVT(float, __f64, sf_to_f32(x.v))
ARWEN_CVT(int, __f64, sf_to_i32_rz(x.v))
ARWEN_CVT(uint, __f64, sf_to_u32_rz(x.v))
ARWEN_CVT(short, __f64, (short)sf_to_i32_rz(x.v))
ARWEN_CVT(ushort, __f64, (ushort)sf_to_u32_rz(x.v))
ARWEN_CVT(char, __f64, (char)sf_to_i32_rz(x.v))
ARWEN_CVT(uchar, __f64, (uchar)sf_to_u32_rz(x.v))
ARWEN_CVT(long, __f64, sf_to_i64_rz(x.v))
ARWEN_CVT(ulong, __f64, sf_to_u64_rz(x.v))
ARWEN_CVT(bool, __f64, (x.v << 1) != 0)

// ---- arithmetic
inline __f64 operator+(__f64 a, __f64 b) { return __f64{sf_add(a.v, b.v)}; }
inline __f64 operator-(__f64 a, __f64 b) { return __f64{sf_sub(a.v, b.v)}; }
inline __f64 operator*(__f64 a, __f64 b) { return __f64{sf_mul(a.v, b.v)}; }
__attribute__((noinline)) inline __f64 operator/(__f64 a, __f64 b) { return __f64{sf_div(a.v, b.v)}; }
inline __f64 operator-(__f64 a) { return __f64{a.v ^ 0x8000000000000000UL}; }
inline __f64 operator+(__f64 a) { return a; }
inline bool operator==(__f64 a, __f64 b) { return sf_eq(a.v, b.v); }
inline bool operator!=(__f64 a, __f64 b) { return !sf_eq(a.v, b.v); }
inline bool operator<(__f64 a, __f64 b) { return sf_lt(a.v, b.v); }
inline bool operator<=(__f64 a, __f64 b) { return sf_le(a.v, b.v); }
inline bool operator>(__f64 a, __f64 b) { return sf_lt(b.v, a.v); }
inline bool operator>=(__f64 a, __f64 b) { return sf_le(b.v, a.v); }
// mixed operands (template code whose types are only known at instantiation):
// the usual arithmetic conversions promote the other operand to double
#define ARWEN_F64_MIXED(T)                                                                                      \
    inline __f64 operator+(__f64 a, T b) { return a + __arwen_cvt<__f64>(b); }                                  \
    inline __f64 operator-(__f64 a, T b) { return a - __arwen_cvt<__f64>(b); }                                  \
    inline __f64 operator*(__f64 a, T b) { return a * __arwen_cvt<__f64>(b); }                                  \
    inline __f64 operator/(__f64 a, T b) { return a / __arwen_cvt<__f64>(b); }                                  \
    inline __f64 operator+(T a, __f64 b) { return __arwen_cvt<__f64>(a) + b; }                                  \
    inline __f64 operator-(T a, __f64 b) { return __arwen_cvt<__f64>(a) - b; }                                  \
    inline __f64 operator*(T a, __f64 b) { return __arwen_cvt<__f64>(a) * b; }                                  \
    inline __f64 operator/(T a, __f64 b) { return __arwen_cvt<__f64>(a) / b; }                                  \
    inline bool operator<(__f64 a, T b) { return a < __arwen_cvt<__f64>(b); }                                   \
    inline bool operator>(__f64 a, T b) { return a > __arwen_cvt<__f64>(b); }                                   \
    inline bool operator<=(__f64 a, T b) { return a <= __arwen_cvt<__f64>(b); }                                 \
    inline bool operator>=(__f64 a, T b) { return a >= __arwen_cvt<__f64>(b); }                                 \
    inline bool operator==(__f64 a, T b) { return a == __arwen_cvt<__f64>(b); }                                 \
    inline bool operator!=(__f64 a, T b) { return a != __arwen_cvt<__f64>(b); }                                 \
    inline bool operator<(T a, __f64 b) { return __arwen_cvt<__f64>(a) < b; }                                   \
    inline bool operator>(T a, __f64 b) { return __arwen_cvt<__f64>(a) > b; }                                   \
    inline bool operator<=(T a, __f64 b) { return __arwen_cvt<__f64>(a) <= b; }                                 \
    inline bool operator>=(T a, __f64 b) { return __arwen_cvt<__f64>(a) >= b; }
ARWEN_F64_MIXED(float)
ARWEN_F64_MIXED(int)
ARWEN_F64_MIXED(uint)
ARWEN_F64_MIXED(long)
ARWEN_F64_MIXED(ulong)
#define ARWEN_F64_COMPOUND(AS)                                                                   \
    inline AS __f64& operator+=(AS __f64& a, __f64 b) { a.v = sf_add(a.v, b.v); return a; }   \
    inline AS __f64& operator-=(AS __f64& a, __f64 b) { a.v = sf_sub(a.v, b.v); return a; }   \
    inline AS __f64& operator*=(AS __f64& a, __f64 b) { a.v = sf_mul(a.v, b.v); return a; }   \
    inline AS __f64& operator/=(AS __f64& a, __f64 b) { a.v = sf_div(a.v, b.v); return a; }   \
    inline AS __f64& operator++(AS __f64& a) { a.v = sf_add(a.v, 0x3FF0000000000000UL); return a; } \
    inline AS __f64& operator--(AS __f64& a) { a.v = sf_sub(a.v, 0x3FF0000000000000UL); return a; } \
    inline __f64 operator++(AS __f64& a, int) { __f64 o = a; a.v = sf_add(a.v, 0x3FF0000000000000UL); return o; } \
    inline __f64 operator--(AS __f64& a, int) { __f64 o = a; a.v = sf_sub(a.v, 0x3FF0000000000000UL); return o; }
#ifdef __METAL_VERSION__
ARWEN_F64_COMPOUND(thread)
ARWEN_F64_COMPOUND(device)
ARWEN_F64_COMPOUND(threadgroup)
#else
ARWEN_F64_COMPOUND()
#endif
// float op= double: C++ computes in double and converts back
#define ARWEN_F32_COMPOUND_F64(AS)                                                                                     \
    inline AS float& operator+=(AS float& a, __f64 b) { a = sf_to_f32(sf_add(sf_from_f32(a), b.v)); return a; }      \
    inline AS float& operator-=(AS float& a, __f64 b) { a = sf_to_f32(sf_sub(sf_from_f32(a), b.v)); return a; }      \
    inline AS float& operator*=(AS float& a, __f64 b) { a = sf_to_f32(sf_mul(sf_from_f32(a), b.v)); return a; }      \
    inline AS float& operator/=(AS float& a, __f64 b) { a = sf_to_f32(sf_div(sf_from_f32(a), b.v)); return a; }
#ifdef __METAL_VERSION__
ARWEN_F32_COMPOUND_F64(thread)
ARWEN_F32_COMPOUND_F64(device)
ARWEN_F32_COMPOUND_F64(threadgroup)
#else
ARWEN_F32_COMPOUND_F64()
#endif

// ---- CUDA double intrinsics
inline __f64 __dadd_rn(__f64 a, __f64 b) { return a + b; }
inline __f64 __dsub_rn(__f64 a, __f64 b) { return a - b; }
inline __f64 __dmul_rn(__f64 a, __f64 b) { return a * b; }
inline __f64 __ddiv_rn(__f64 a, __f64 b) { return a / b; }
__attribute__((noinline)) inline __f64 __dsqrt_rn(__f64 a) { return __f64{sf_sqrt(a.v)}; }
__attribute__((noinline)) inline __f64 __fma_rn(__f64 a, __f64 b, __f64 c) { return __f64{sf_fma(a.v, b.v, c.v)}; }
inline __f64 __drcp_rn(__f64 a) { return __f64{sf_div(0x3FF0000000000000UL, a.v)}; }
inline float __double2float_rn(__f64 a) { return sf_to_f32(a.v); }
inline int __double2int_rz(__f64 a) { return sf_to_i32_rz(a.v); }
inline int __double2int_rn(__f64 a) { return sf_to_i32_rz(sf_rint(a.v)); }
inline long __double2ll_rz(__f64 a) { return sf_to_i64_rz(a.v); }
inline long __double2ll_rn(__f64 a) { return sf_to_i64_rz(sf_rint(a.v)); }
inline __f64 __int2double_rn(int a) { return __f64{sf_from_i64(a)}; }
inline __f64 __ll2double_rn(long a) { return __f64{sf_from_i64(a)}; }
inline long __double_as_longlong(__f64 a) { return (long)a.v; }
inline __f64 __longlong_as_double(long a) { return __f64{(ulong)a}; }
inline int __double2hiint(__f64 a) { return (int)(a.v >> 32); }
inline int __double2loint(__f64 a) { return (int)(uint)a.v; }
inline __f64 __hiloint2double(int hi, int lo) { return __f64{((ulong)(uint)hi << 32) | (ulong)(uint)lo}; }

// ---- exact math (no rounding error, or a single IEEE rounding)
#ifdef __METAL_VERSION__
inline __f64 nan(constant char*) { return __f64{SF_DEFAULT_NAN}; }
#endif
inline __f64 fabs(__f64 a) { return __f64{a.v & 0x7FFFFFFFFFFFFFFFUL}; }
inline __f64 copysign(__f64 a, __f64 b) { return __f64{(a.v & 0x7FFFFFFFFFFFFFFFUL) | (b.v & 0x8000000000000000UL)}; }
inline bool isnan(__f64 a) { return SF_ISNAN(a.v); }
inline bool isinf(__f64 a) { return (a.v << 1) == 0xFFE0000000000000UL; }
inline bool isfinite(__f64 a) { return ((a.v >> 52) & 0x7FF) != 0x7FF; }
inline bool signbit(__f64 a) { return (a.v >> 63) != 0; }
// CUDA fmax/fmin (IEEE maxNum/minNum; -0 < +0)
inline __f64 fmax(__f64 a, __f64 b) {
    if (SF_ISNAN(a.v)) return SF_ISNAN(b.v) ? __f64{a.v | 0x0008000000000000UL} : b;
    if (SF_ISNAN(b.v)) return a;
    if (((a.v | b.v) << 1) == 0) return __f64{a.v & b.v};
    return sf_lt(a.v, b.v) ? b : a;
}
inline __f64 fmin(__f64 a, __f64 b) {
    if (SF_ISNAN(a.v)) return SF_ISNAN(b.v) ? __f64{a.v | 0x0008000000000000UL} : b;
    if (SF_ISNAN(b.v)) return a;
    if (((a.v | b.v) << 1) == 0) return __f64{a.v | b.v};
    return sf_lt(b.v, a.v) ? b : a;
}
inline __f64 max(__f64 a, __f64 b) { return fmax(a, b); }
inline __f64 min(__f64 a, __f64 b) { return fmin(a, b); }
inline __f64 max(float a, __f64 b) { return fmax(__f64{sf_from_f32(a)}, b); }
inline __f64 max(__f64 a, float b) { return fmax(a, __f64{sf_from_f32(b)}); }
inline __f64 min(float a, __f64 b) { return fmin(__f64{sf_from_f32(a)}, b); }
inline __f64 min(__f64 a, float b) { return fmin(a, __f64{sf_from_f32(b)}); }
inline __f64 fdim(__f64 a, __f64 b) {
    if (SF_ISNAN(a.v) || SF_ISNAN(b.v)) return a + b;
    return sf_lt(b.v, a.v) ? a - b : __f64{0};
}
inline __f64 floor(__f64 a) { return __f64{sf_floor(a.v)}; }
inline __f64 ceil(__f64 a) { return __f64{sf_ceil(a.v)}; }
inline __f64 trunc(__f64 a) { return __f64{sf_trunc(a.v)}; }
inline __f64 round(__f64 a) { return __f64{sf_round(a.v)}; }
inline __f64 rint(__f64 a) { return __f64{sf_rint(a.v)}; }
inline __f64 nearbyint(__f64 a) { return __f64{sf_rint(a.v)}; }
inline long lrint(__f64 a) { return sf_to_i64_rz(sf_rint(a.v)); }
inline long llrint(__f64 a) { return sf_to_i64_rz(sf_rint(a.v)); }
inline long lround(__f64 a) { return sf_to_i64_rz(sf_round(a.v)); }
inline long llround(__f64 a) { return sf_to_i64_rz(sf_round(a.v)); }
inline __f64 sqrt(__f64 a) { return __dsqrt_rn(a); }
inline __f64 fma(__f64 a, __f64 b, __f64 c) { return __fma_rn(a, b, c); }
// ldexp: exact scaling with one rounding (subnormal results)
inline __f64 ldexp(__f64 a, int e) {
    int ea = SF_EXP(a.v);
    if (ea == 0x7FF || !(a.v << 1)) return a;
    if (e > 2100) e = 2100;
    if (e < -2100) e = -2100;
    __f64 r = a;
    // scale in steps that keep every intermediate exact or singly rounded
    while (e > 1000) { r.v = sf_mul(r.v, 0x7E70000000000000UL); e -= 1000; }       // 2^1000
    while (e < -1000) { r.v = sf_mul(r.v, 0x0170000000000000UL); e += 1000; }      // 2^-1000
    ulong scale = (ulong)(e + 1023) << 52;
    if (e + 1023 <= 0) {
        // 2^e is subnormal here: multiply by 2^(e+600) then 2^-600
        r.v = sf_mul(r.v, (ulong)(e + 600 + 1023) << 52);
        return __f64{sf_mul(r.v, 0x1A70000000000000UL)};
    }
    return __f64{sf_mul(r.v, scale)};
}
template <typename P> inline __f64 frexp(__f64 a, P e) {
    int ea = SF_EXP(a.v);
    if (ea == 0x7FF || !(a.v << 1)) { *e = 0; return a; }
    ulong v = a.v;
    int adj = 0;
    if (!ea) { v = sf_mul(v, 0x4350000000000000UL); ea = SF_EXP(v); adj = -54; }   // * 2^54
    *e = ea - 1022 + adj;
    return __f64{(v & 0x800FFFFFFFFFFFFFUL) | 0x3FE0000000000000UL};
}
template <typename P> inline __f64 modf(__f64 a, P ip) {
    __f64 t = trunc(a);
    *ip = t;
    if (isinf(a)) return copysign(__f64{0}, a);
    return copysign(a - t, a);
}
inline __f64 fmod(__f64 x, __f64 y) {
    // exact: remainder of x by |y| computed by long division on significands
    ulong ux = x.v, uy = y.v & 0x7FFFFFFFFFFFFFFFUL;
    ulong sx = ux & 0x8000000000000000UL;
    ux &= 0x7FFFFFFFFFFFFFFFUL;
    if (SF_ISNAN(x.v) || SF_ISNAN(y.v) || SF_EXP(ux) == 0x7FF || uy == 0) return __f64{SF_DEFAULT_NAN};
    if (ux < uy) return x;
    int ex = SF_EXP(ux), ey = SF_EXP(uy);
    ulong mx = SF_FRAC(ux), my = SF_FRAC(uy);
    if (!ex) { int sh = sf_clz64(mx) - 11; mx <<= sh; ex = 1 - sh; } else mx |= 0x0010000000000000UL;
    if (!ey) { int sh = sf_clz64(my) - 11; my <<= sh; ey = 1 - sh; } else my |= 0x0010000000000000UL;
    for (int n = ex - ey; n > 0; --n) {
        if (mx >= my) mx -= my;
        mx <<= 1;
    }
    if (mx >= my) mx -= my;
    if (!mx) return __f64{sx};
    int sh = sf_clz64(mx) - 11;
    mx <<= sh;
    int e = ey - sh;
    if (e <= 0) { mx >>= (1 - e); e = 0; }
    else mx &= 0x000FFFFFFFFFFFFFUL;
    return __f64{sx | ((ulong)e << 52) | mx};
}

#elif defined(ARWEN_F64_MODE_DF64) || defined(ARWEN_F64_MODE_FP32)
#error "df64/fp32 FP64 modes are not implemented yet; translate with f64_mode='sf64'"
#else
#error "arwen_f64.h: define ARWEN_F64_MODE_SF64, ARWEN_F64_MODE_DF64 or ARWEN_F64_MODE_FP32"
#endif

#include "arwen_f64_math.h"
