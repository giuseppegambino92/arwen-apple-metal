// arwen_fastlibm_powf.h -- fast path of the glibc powf copies (see arwen_fastlibm.h).
//
// glibc powf for x > 0: log2(x) = (log2 c + k) + A4 r + A2 r^2 ... (table and
// degree-5 polynomial, r = z invc - 1), ylogx = y log2(x), then
// 2^ylogx = T[k & 31] 2^(k >> 5) (1 + C2 r + (C0 r + C1) r^2), k = round(32 ylogx),
// r = ylogx - k/32.  Negative or special x, special y, and |ylogx| >= 120 are
// left to the exact original.  gpuwm.metal.fastlibm routes only call sites
// whose exponent is a literal, each (copy, exponent) pair verified on all 2^32 x.
#pragma once
#include "arwen_fastlibm_ff.h"
#include "arwen_fastlibm_tables_logf.h"
#include "arwen_fastlibm_tables_expf.h"
#include "arwen_fastlibm_tables_powf.h"

inline float __arwen_fast_glibc_powf(float x, float y) {
    uint ix = as_type<uint>(x);
    uint ay = as_type<uint>(y) & 0x7fffffffu;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || ix == 0x3f800000u ||
        ay - 0x00800000u >= 0x7f800000u - 0x00800000u)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    uint tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    uint top = tmp & 0xff800000u;
    float z = as_type<float>(ix - top);
    float kf = (float)(((int)top) >> 23);
    // r = z invc - 1 (as logf)
    float p = z * ARWEN_LOGF_INVC_H[i];
    float pe = metal::fma(z, ARWEN_LOGF_INVC_H[i], -p);
    pe = metal::fma(z, ARWEN_LOGF_INVC_L[i], pe);
    __arwen_ff r = __arwen_ff_two_sum(p - 1.0f, pe);
    __arwen_ff y0 = __arwen_ff_add(__arwen_ff_of(kf, 0.0f), __arwen_ff_of(ARWEN_POWF_LOG2C_H[i], ARWEN_POWF_LOG2C_L[i]));
    __arwen_ff r2 = __arwen_ff_mul(r, r);
    __arwen_ff r4 = __arwen_ff_mul(r2, r2);
    __arwen_ff a = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_POWF_A0_H, ARWEN_POWF_A0_L), r),
                                  __arwen_ff_of(ARWEN_POWF_A1_H, ARWEN_POWF_A1_L));
    __arwen_ff b = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_POWF_A2_H, ARWEN_POWF_A2_L), r),
                                  __arwen_ff_of(ARWEN_POWF_A3_H, ARWEN_POWF_A3_L));
    __arwen_ff q = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_POWF_A4_H, ARWEN_POWF_A4_L), r), y0);
    q = __arwen_ff_add(__arwen_ff_mul(b, r2), q);
    __arwen_ff logx = __arwen_ff_add(__arwen_ff_mul(a, r4), q);
    __arwen_ff ylogx = __arwen_ff_mul(__arwen_ff_of(y, 0.0f), logx);
    if (!(metal::fabs(ylogx.h) < 120.0f))
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    // k = round(32 ylogx): the copies round a binary64 32*ylogx; undecided near a half
    float th = ylogx.h * 32.0f, tl = ylogx.l * 32.0f;
    float k2 = metal::rint(th);
    float f = (th - k2) + tl;
    if (metal::fabs(metal::fabs(f) - 0.5f) < 0x1p-26f)
        return as_type<float>(ARWEN_FASTLIBM_UNSURE_BITS);
    if (f > 0.5f) k2 += 1.0f;
    if (f < -0.5f) k2 -= 1.0f;
    __arwen_ff rr = __arwen_ff_two_sum(th - k2, tl);
    rr.h *= 0x1p-5f;
    rr.l *= 0x1p-5f;
    int kk = (int)k2;
    int j = kk & 31;
    __arwen_ff zc = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_EXP2F_C0_H, ARWEN_EXP2F_C0_L), rr),
                                   __arwen_ff_of(ARWEN_EXP2F_C1_H, ARWEN_EXP2F_C1_L));
    __arwen_ff pr = __arwen_ff_add(__arwen_ff_mul(__arwen_ff_of(ARWEN_EXP2F_C2_H, ARWEN_EXP2F_C2_L), rr),
                                   __arwen_ff_mul(zc, __arwen_ff_mul(rr, rr)));
    pr = __arwen_ff_add(__arwen_ff_of(1.0f, 0.0f), pr);
    __arwen_ff res = __arwen_ff_mul(pr, __arwen_ff_of(ARWEN_EXP2F_T_H[j], ARWEN_EXP2F_T_L[j]));
    // budget: binary64 roundings of the copies and the float-float evaluation,
    // the log2 error scaled by |y| (relative to the summands: cancellation near
    // x = c) and amplified into 2^ylogx
    float E = metal::fabs(res.h) * (0x1p-38f + 0x1p-40f * (metal::fabs(ylogx.h) +
                                    metal::fabs(y) * (metal::fabs(y0.h) + 2.0f * metal::fabs(r.h))));
    float m = __arwen_ff_round_sure(res, E);
    if (as_type<uint>(m) == ARWEN_FASTLIBM_UNSURE_BITS) return m;
    return m * as_type<float>((uint)(((kk >> 5) + 127) << 23));
}
