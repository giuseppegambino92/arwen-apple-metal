// arwen_fastlibm_logf.h -- fast path of the glibc logf copies (see arwen_fastlibm.h).
#pragma once
#include "arwen_fastlibm_ff.h"
#include "arwen_fastlibm_tables_logf.h"

// glibc logf: y = (k ln2 + logc) + r + r^2 (A1 r + A2 + A0 r^2), r = z invc - 1.
inline float __arwen_fast_glibc_logf(float x) {
    uint ix = as_type<uint>(x);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || ix == 0x3f800000u)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);       // 0, subnormal, negative, inf, nan, 1
    uint tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = ((int)tmp) >> 23;
    float z = as_type<float>(ix - (tmp & 0xff800000u));
    // r = z * invc - 1: z * invc_h exactly as p + pe, p - 1 exact (Sterbenz)
    float p = z * ARWEN_LOGF_INVC_H[i];
    float pe = metal::fma(z, ARWEN_LOGF_INVC_H[i], -p);
    pe = metal::fma(z, ARWEN_LOGF_INVC_L[i], pe);
    __arwen_ff r = __arwen_ff_two_sum(p - 1.0f, pe);
    // y0 = k ln2 + logc (k * ARWEN_LN2_1, k * ARWEN_LN2_2 exact for |k| < 256)
    float kf = (float)k;
    __arwen_ff y0 = __arwen_ff_two_sum(kf * ARWEN_LN2_1, ARWEN_LOGF_LOGC_H[i]);
    __arwen_ff y1 = __arwen_ff_two_sum(kf * ARWEN_LN2_2, ARWEN_LOGF_LOGC_L[i]);
    y1.l = metal::fma(kf, ARWEN_LN2_3, y1.l);
    y0 = __arwen_ff_add(y0, y1);
    __arwen_ff r2 = __arwen_ff_mul(r, r);
    __arwen_ff q = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_LOGF_A1_H, ARWEN_LOGF_A1_L), r),
                                  __arwen_ff_of(ARWEN_LOGF_A2_H, ARWEN_LOGF_A2_L));
    q = __arwen_ff_add(q, __arwen_ff_mul(__arwen_ff_of(ARWEN_LOGF_A0_H, ARWEN_LOGF_A0_L), r2));
    q = __arwen_ff_mul(q, r2);
    __arwen_ff y = __arwen_ff_add(__arwen_ff_add(y0, r), q);
    // budget: float-float evaluation and binary64 rounding of the copies,
    // relative to the summands (cancellation), plus z*invc rounded to binary64
    // in the mul+sub copies (2^-53 absolute)
    float E = 0x1p-36f * (metal::fabs(y0.h) + metal::fabs(r.h)) + 0x1p-50f;
    return __arwen_ff_round_sure(y, E);
}

