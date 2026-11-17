// arwen_fastlibm_ff.h -- float-float arithmetic and the rounding test of the libm fast paths.
#pragma once

#define ARWEN_FASTLIBM_UNSURE_BITS 0x7FBADBADu

struct __arwen_ff { float h, l; };

inline __arwen_ff __arwen_ff_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return __arwen_ff{s, (a - (s - bb)) + (b - bb)};
}
inline __arwen_ff __arwen_ff_fast_two_sum(float a, float b) {   // |a| >= |b| or a == 0
    float s = a + b;
    return __arwen_ff{s, b - (s - a)};
}
inline __arwen_ff __arwen_ff_add(__arwen_ff a, __arwen_ff b) {
    __arwen_ff s = __arwen_ff_two_sum(a.h, b.h);
    __arwen_ff t = __arwen_ff_two_sum(a.l, b.l);
    s.l += t.h;
    s = __arwen_ff_fast_two_sum(s.h, s.l);
    s.l += t.l;
    return __arwen_ff_fast_two_sum(s.h, s.l);
}
inline __arwen_ff __arwen_ff_mul(__arwen_ff a, __arwen_ff b) {
    float p = a.h * b.h;
    float e = metal::fma(a.h, b.h, -p);
    e = metal::fma(a.h, b.l, e);
    e = metal::fma(a.l, b.h, e);
    return __arwen_ff_fast_two_sum(p, e);
}
inline __arwen_ff __arwen_ff_of(float h, float l) { return __arwen_ff{h, l}; }

// The binary32 rounding of any value within E of h + l, if it is certainly h:
// the interval [h + l - E, h + l + E] lies strictly between the rounding
// boundaries around h, h + ulp/2 above and h - ulp/2 below -- or h - ulp/4
// below when h is a power of two (the grid below is twice as fine).
inline float __arwen_ff_round_sure(__arwen_ff y, float E) {
    uint hb = as_type<uint>(y.h);
    uint e = hb & 0x7f800000u;
    // tiny (keeps the half-ulp computation normal), inf or nan
    if (e <= (26u << 23) || e == 0x7f800000u)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    float half_ulp = as_type<float>(e - (24u << 23));
    float below = ((hb & 0x007fffffu) == 0u) ? 0.5f * half_ulp : half_ulp;
    if (y.l + E >= half_ulp || E - y.l >= below)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    return y.h;
}

