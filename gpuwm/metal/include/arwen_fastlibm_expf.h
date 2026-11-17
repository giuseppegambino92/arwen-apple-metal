// arwen_fastlibm_expf.h -- fast path of the glibc expf copies (see arwen_fastlibm.h).
#pragma once
#include "arwen_fastlibm_ff.h"
#include "arwen_fastlibm_tables_expf.h"

// glibc expf: k = round(x 32/ln2), r = x 32/ln2 - k,
// y = 2^(k/32) (1 + C2 r + (C1 + C0 r) r^2), 2^(k/32) = T[k & 31] 2^(k >> 5).
inline float __arwen_fast_glibc_expf(float x) {
    uint ix = as_type<uint>(x);
    uint ax = ix & 0x7fffffffu;
    // |x| >= 80 (result near overflow / underflow), inf, nan, subnormal: unsure
    if (ax >= 0x42a00000u || (ax != 0u && ax < 0x00800000u))
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    // t = x * InvLn2N as float-float (exact product with the high part)
    float th = x * ARWEN_EXPF_INVLN2N_H;
    float tl = metal::fma(x, ARWEN_EXPF_INVLN2N_H, -th);
    tl = metal::fma(x, ARWEN_EXPF_INVLN2N_L, tl);
    float kf = metal::rint(th);
    float f = (th - kf) + tl;                      // th - kf exact (|th| < 2^13)
    // the copies round x * InvLn2N (fma or rounded product) to the nearest
    // integer: undecided within 2^-30 of a half
    float af = metal::fabs(f);
    if (metal::fabs(af - 0.5f) < 0x1p-30f)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    if (f > 0.5f) kf += 1.0f;
    if (f < -0.5f) kf -= 1.0f;
    __arwen_ff r = __arwen_ff_two_sum(th - kf, tl);
    int k = (int)kf;
    int j = k & 31;
    // P(r) = 1 + C2 r + (C1 + C0 r) r^2
    __arwen_ff c1 = __arwen_ff_add(__arwen_ff_of(ARWEN_EXPF_C1_H, ARWEN_EXPF_C1_L),
                                   __arwen_ff_mul(__arwen_ff_of(ARWEN_EXPF_C0_H, ARWEN_EXPF_C0_L), r));
    __arwen_ff pr = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_EXPF_C2_H, ARWEN_EXPF_C2_L), r),
                                   __arwen_ff_mul(c1, __arwen_ff_mul(r, r)));
    pr = __arwen_ff_add(__arwen_ff_of(1.0f, 0.0f), pr);
    __arwen_ff y = __arwen_ff_mul(pr, __arwen_ff_of(ARWEN_EXP2F_T_H[j], ARWEN_EXP2F_T_L[j]));
    // decide on the unscaled value in [~0.98, ~2.03]; the scale 2^(k >> 5) is exact
    // for the normal results |x| < 80 gives
    float E = 0x1p-36f * metal::fabs(y.h);
    float m = __arwen_ff_round_sure(y, E);
    if (as_type<uint>(m) == ARWEN_FASTLIBM_UNSURE_BITS) return m;
    return m * as_type<float>((uint)(((k >> 5) + 127) << 23));
}
