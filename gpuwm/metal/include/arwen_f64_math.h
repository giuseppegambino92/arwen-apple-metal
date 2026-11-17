// arwen_f64_math.h -- double-precision elementary functions on __f64.
//
// Algorithms: Sun fdlibm 5.3 (e_exp.c, e_log.c, e_log10.c, e_pow.c, s_atan.c,
// e_atan2.c, k_sin.c, k_cos.c, e_rem_pio2.c medium path), transcribed onto the
// __f64 operators.  Copyright (C) 1993 by Sun Microsystems, Inc. All rights
// reserved.  Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// Accuracy is fdlibm's (< 1 ulp); tests/metal/f64_math_host_check.cpp measures
// it against the host libm.  CUDA's libdevice is a different < 1-2 ulp
// implementation, so results can differ from the CUDA run in the last bit.
// Trigonometric argument reduction uses fdlibm's three-part Cody-Waite path,
// exact for |x| < 2^20 * pi/2 (1.6e6); larger arguments lose accuracy.
#pragma once

#ifdef __METAL_VERSION__
#define AF_INLINE inline
#else
#define AF_INLINE inline
#endif

#define AF_C(bits) (__f64{(sf_u64)(bits##UL)})
AF_INLINE int af_hi(__f64 x) { return (int)(x.v >> 32); }
AF_INLINE sf_u32 af_lo(__f64 x) { return (sf_u32)x.v; }
AF_INLINE __f64 af_words(int hi, sf_u32 lo) { return __f64{((sf_u64)(sf_u32)hi << 32) | (sf_u64)lo}; }
AF_INLINE __f64 af_set_lo(__f64 x, sf_u32 lo) { return __f64{(x.v & 0xFFFFFFFF00000000UL) | (sf_u64)lo}; }
AF_INLINE __f64 af_set_hi(__f64 x, int hi) { return __f64{((sf_u64)(sf_u32)hi << 32) | (x.v & 0xFFFFFFFFUL)}; }
AF_INLINE __f64 af_i(int k) { return __f64{sf_from_i64(k)}; }
AF_INLINE __f64 af_scalbn(__f64 x, int n) { return ldexp(x, n); }

#define AF_ONE AF_C(0x3FF0000000000000)
#define AF_TWO AF_C(0x4000000000000000)
#define AF_HALF AF_C(0x3FE0000000000000)
#define AF_ZERO AF_C(0x0000000000000000)
#define AF_HUGE AF_C(0x7E37E43C8800759C)
#define AF_TINY AF_C(0x01A56E1FC2F8F359)
#define AF_NAN __f64{SF_DEFAULT_NAN}
#define AF_INF AF_C(0x7FF0000000000000)

// ---------------------------------------------------------------- exp (e_exp.c)
AF_INLINE __f64 exp(__f64 x) {
    const __f64 ln2HI = AF_C(0x3FE62E42FEE00000), ln2LO = AF_C(0x3DEA39EF35793C76);
    const __f64 invln2 = AF_C(0x3FF71547652B82FE);
    const __f64 P1 = AF_C(0x3FC555555555553E), P2 = AF_C(0xBF66C16C16BEBD93), P3 = AF_C(0x3F11566AAF25DE2C),
                P4 = AF_C(0xBEBBBD41C5D26BF1), P5 = AF_C(0x3E66376972BEA4D0);
    const __f64 o_threshold = AF_C(0x40862E42FEFA39EF), u_threshold = AF_C(0xC0874910D52D3051);
    __f64 hi = AF_ZERO, lo = AF_ZERO, c, t;
    int k = 0;
    int hx = af_hi(x);
    int xsb = (hx >> 31) & 1;
    hx &= 0x7fffffff;
    if (hx >= 0x40862E42) {
        if (hx >= 0x7ff00000) {
            if (((hx & 0xfffff) | af_lo(x)) != 0) return x + x;
            return xsb == 0 ? x : AF_ZERO;
        }
        if (x > o_threshold) return AF_INF;
        if (x < u_threshold) return AF_ZERO;
    }
    if (hx > 0x3fd62e42) {
        if (hx < 0x3FF0A2B2) {
            hi = xsb ? x + ln2HI : x - ln2HI;
            lo = xsb ? -ln2LO : ln2LO;
            k = 1 - xsb - xsb;
        } else {
            k = __arwen_cvt<int>(invln2 * x + (xsb ? -AF_HALF : AF_HALF));
            t = af_i(k);
            hi = x - t * ln2HI;
            lo = t * ln2LO;
        }
        x = hi - lo;
    } else if (hx < 0x3e300000) {
        return AF_ONE + x;
    } else {
        k = 0;
    }
    t = x * x;
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return AF_ONE - ((x * c) / (c - AF_TWO) - x);
    __f64 y = AF_ONE - ((lo - (x * c) / (AF_TWO - c)) - hi);
    if (k >= -1021) return af_set_hi(y, af_hi(y) + (k << 20));
    y = af_set_hi(y, af_hi(y) + ((k + 1000) << 20));
    return y * AF_C(0x0170000000000000);   // 2^-1000
}

// ---------------------------------------------------------------- log (e_log.c)
AF_INLINE __f64 log(__f64 x) {
    const __f64 ln2_hi = AF_C(0x3FE62E42FEE00000), ln2_lo = AF_C(0x3DEA39EF35793C76), two54 = AF_C(0x4350000000000000);
    const __f64 Lg1 = AF_C(0x3FE5555555555593), Lg2 = AF_C(0x3FD999999997FA04), Lg3 = AF_C(0x3FD2492494229359),
                Lg4 = AF_C(0x3FCC71C51D8E78AF), Lg5 = AF_C(0x3FC7466496CB03DE), Lg6 = AF_C(0x3FC39A09D078C69F),
                Lg7 = AF_C(0x3FC2F112DF3E5244);
    int hx = af_hi(x);
    sf_u32 lx = af_lo(x);
    int k = 0;
    if (hx < 0x00100000) {
        if (((hx & 0x7fffffff) | lx) == 0) return -AF_INF;
        if (hx < 0) return AF_NAN;
        k -= 54;
        x = x * two54;
        hx = af_hi(x);
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    int i = (hx + 0x95f64) & 0x100000;
    x = af_set_hi(x, hx | (i ^ 0x3ff00000));
    k += (i >> 20);
    __f64 f = x - AF_ONE;
    __f64 dk;
    if ((0x000fffff & (2 + hx)) < 3) {
        if (f == AF_ZERO) {
            if (k == 0) return AF_ZERO;
            dk = af_i(k);
            return dk * ln2_hi + dk * ln2_lo;
        }
        __f64 R = f * f * (AF_HALF - AF_C(0x3FD5555555555555) * f);
        if (k == 0) return f - R;
        dk = af_i(k);
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    __f64 s = f / (AF_TWO + f);
    dk = af_i(k);
    __f64 z = s * s;
    i = hx - 0x6147a;
    __f64 w = z * z;
    int j = 0x6b851 - hx;
    __f64 t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    __f64 t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    i |= j;
    __f64 R = t2 + t1;
    if (i > 0) {
        __f64 hfsq = AF_HALF * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

// ---------------------------------------------------------------- log10 (e_log10.c)
AF_INLINE __f64 log10(__f64 x) {
    const __f64 two54 = AF_C(0x4350000000000000), ivln10 = AF_C(0x3FDBCB7B1526E50E),
                log10_2hi = AF_C(0x3FD34413509F6000), log10_2lo = AF_C(0x3D59FEF311F12B36);
    int hx = af_hi(x);
    sf_u32 lx = af_lo(x);
    int k = 0;
    if (hx < 0x00100000) {
        if (((hx & 0x7fffffff) | lx) == 0) return -AF_INF;
        if (hx < 0) return AF_NAN;
        k -= 54;
        x = x * two54;
        hx = af_hi(x);
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023;
    int i = (int)(((sf_u32)k & 0x80000000u) >> 31);
    hx = (hx & 0x000fffff) | ((0x3ff - i) << 20);
    __f64 y = af_i(k + i);
    x = af_set_hi(x, hx);
    __f64 z = y * log10_2lo + ivln10 * log(x);
    return z + y * log10_2hi;
}
AF_INLINE __f64 log2(__f64 x) {
    // log(x) / ln 2 split: k + log(m)/ln2 keeps integer powers of two exact
    int hx = af_hi(x);
    if (hx < 0x00100000 || hx >= 0x7ff00000) return log(x) * AF_C(0x3FF71547652B82FE);
    int k = (hx >> 20) - 1023;
    __f64 m = af_set_hi(x, (hx & 0x000fffff) | 0x3ff00000);
    if (m == AF_ONE) return af_i(k);
    return af_i(k) + log(m) * AF_C(0x3FF71547652B82FE);
}

// ---------------------------------------------------------------- pow (e_pow.c)
AF_INLINE __f64 pow(__f64 x, __f64 y) {
    const __f64 bp[2] = {AF_ONE, AF_C(0x3FF8000000000000)};
    const __f64 dp_h[2] = {AF_ZERO, AF_C(0x3FE2B80340000000)};
    const __f64 dp_l[2] = {AF_ZERO, AF_C(0x3E4CFDEB43CFD006)};
    const __f64 two53 = AF_C(0x4340000000000000);
    const __f64 L1 = AF_C(0x3FE3333333333303), L2 = AF_C(0x3FDB6DB6DB6FABFF), L3 = AF_C(0x3FD55555518F264D),
                L4 = AF_C(0x3FD17460A91D4101), L5 = AF_C(0x3FCD864A93C9DB65), L6 = AF_C(0x3FCA7E284A454EEF);
    const __f64 P1 = AF_C(0x3FC555555555553E), P2 = AF_C(0xBF66C16C16BEBD93), P3 = AF_C(0x3F11566AAF25DE2C),
                P4 = AF_C(0xBEBBBD41C5D26BF1), P5 = AF_C(0x3E66376972BEA4D0);
    const __f64 lg2 = AF_C(0x3FE62E42FEFA39EF), lg2_h = AF_C(0x3FE62E4300000000), lg2_l = AF_C(0xBE205C610CA86C39);
    const __f64 ovt = AF_C(0x3C971547652B82FE);
    const __f64 cp = AF_C(0x3FEEC709DC3A03FD), cp_h = AF_C(0x3FEEC709E0000000), cp_l = AF_C(0xBE3E2FE0145B01F5);
    const __f64 ivln2 = AF_C(0x3FF71547652B82FE), ivln2_h = AF_C(0x3FF7154760000000), ivln2_l = AF_C(0x3E54AE0BF85DDF44);
    __f64 z, ax, z_h, z_l, p_h, p_l, y1, t1, t2, r, s, t, u, v, w;
    int i, j, k, yisint, n;
    int hx = af_hi(x), hy = af_hi(y);
    sf_u32 lx = af_lo(x), ly = af_lo(y);
    int ix = hx & 0x7fffffff, iy = hy & 0x7fffffff;
    if ((iy | (int)ly) == 0) return AF_ONE;
    if (hx == 0x3ff00000 && lx == 0) return AF_ONE;
    if (ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) || iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0))
        return (x + AF_ZERO) + (y + AF_ZERO);
    yisint = 0;
    if (hx < 0) {
        if (iy >= 0x43400000) yisint = 2;
        else if (iy >= 0x3ff00000) {
            k = (iy >> 20) - 0x3ff;
            if (k > 20) {
                sf_u32 jj = ly >> (52 - k);
                if ((jj << (52 - k)) == ly) yisint = 2 - (int)(jj & 1);
            } else if (ly == 0) {
                j = iy >> (20 - k);
                if ((j << (20 - k)) == iy) yisint = 2 - (j & 1);
            }
        }
    }
    if (ly == 0) {
        if (iy == 0x7ff00000) {
            if (((ix - 0x3ff00000) | (int)lx) == 0) return AF_ONE;
            else if (ix >= 0x3ff00000) return (hy >= 0) ? y : AF_ZERO;
            else return (hy < 0) ? -y : AF_ZERO;
        }
        if (iy == 0x3ff00000) return (hy < 0) ? AF_ONE / x : x;
        if (hy == 0x40000000) return x * x;
        if (hy == 0x3fe00000 && hx >= 0) return sqrt(x);
    }
    ax = fabs(x);
    if (lx == 0) {
        if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
            z = ax;
            if (hy < 0) z = AF_ONE / z;
            if (hx < 0) {
                if (((ix - 0x3ff00000) | yisint) == 0) z = AF_NAN;
                else if (yisint == 1) z = -z;
            }
            return z;
        }
    }
    n = (int)(((sf_u32)hx >> 31)) - 1;
    if ((n | yisint) == 0) return AF_NAN;
    s = AF_ONE;
    if ((n | (yisint - 1)) == 0) s = -AF_ONE;
    if (iy > 0x41e00000) {
        if (iy > 0x43f00000) {
            if (ix <= 0x3fefffff) return (hy < 0) ? AF_HUGE * AF_HUGE : AF_TINY * AF_TINY;
            if (ix >= 0x3ff00000) return (hy > 0) ? AF_HUGE * AF_HUGE : AF_TINY * AF_TINY;
        }
        if (ix < 0x3fefffff) return (hy < 0) ? s * AF_HUGE * AF_HUGE : s * AF_TINY * AF_TINY;
        if (ix > 0x3ff00000) return (hy > 0) ? s * AF_HUGE * AF_HUGE : s * AF_TINY * AF_TINY;
        t = ax - AF_ONE;
        w = (t * t) * (AF_HALF - t * (AF_C(0x3FD5555555555555) - t * AF_C(0x3FD0000000000000)));
        u = ivln2_h * t;
        v = t * ivln2_l - w * ivln2;
        t1 = u + v;
        t1 = af_set_lo(t1, 0);
        t2 = v - (t1 - u);
    } else {
        __f64 ss, s2, s_h, s_l, t_h, t_l;
        n = 0;
        if (ix < 0x00100000) { ax = ax * two53; n -= 53; ix = af_hi(ax); }
        n += ((ix) >> 20) - 0x3ff;
        j = ix & 0x000fffff;
        ix = j | 0x3ff00000;
        if (j <= 0x3988E) k = 0;
        else if (j < 0xBB67A) k = 1;
        else { k = 0; n += 1; ix -= 0x00100000; }
        ax = af_set_hi(ax, ix);
        u = ax - bp[k];
        v = AF_ONE / (ax + bp[k]);
        ss = u * v;
        s_h = af_set_lo(ss, 0);
        t_h = af_words(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18), 0);
        t_l = ax - (t_h - bp[k]);
        s_l = v * ((u - s_h * t_h) - s_h * t_l);
        s2 = ss * ss;
        r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        t_h = AF_C(0x4008000000000000) + s2 + r;
        t_h = af_set_lo(t_h, 0);
        t_l = r - ((t_h - AF_C(0x4008000000000000)) - s2);
        u = s_h * t_h;
        v = s_l * t_h + t_l * ss;
        p_h = u + v;
        p_h = af_set_lo(p_h, 0);
        p_l = v - (p_h - u);
        z_h = cp_h * p_h;
        z_l = cp_l * p_h + p_l * cp + dp_l[k];
        t = af_i(n);
        t1 = (((z_h + z_l) + dp_h[k]) + t);
        t1 = af_set_lo(t1, 0);
        t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
    }
    y1 = af_set_lo(y, 0);
    p_l = (y - y1) * t1 + y * t2;
    p_h = y1 * t1;
    z = p_l + p_h;
    j = af_hi(z);
    i = (int)af_lo(z);
    if (j >= 0x40900000) {
        if (((j - 0x40900000) | i) != 0) return s * AF_HUGE * AF_HUGE;
        if (p_l + ovt > z - p_h) return s * AF_HUGE * AF_HUGE;
    } else if ((j & 0x7fffffff) >= 0x4090cc00) {
        if (((j - (int)0xc090cc00) | i) != 0) return s * AF_TINY * AF_TINY;
        if (p_l <= z - p_h) return s * AF_TINY * AF_TINY;
    }
    i = j & 0x7fffffff;
    k = (i >> 20) - 0x3ff;
    n = 0;
    if (i > 0x3fe00000) {
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        t = af_words(n & ~(0x000fffff >> k), 0);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if (j < 0) n = -n;
        p_h -= t;
    }
    t = p_l + p_h;
    t = af_set_lo(t, 0);
    u = t * lg2_h;
    v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
    z = u + v;
    w = v - (z - u);
    t = z * z;
    t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    r = (z * t1) / (t1 - AF_TWO) - (w + z * w);
    z = AF_ONE - (r - z);
    j = af_hi(z);
    j += (n << 20);
    if ((j >> 20) <= 0) z = af_scalbn(z, n);
    else z = af_set_hi(z, j);
    return s * z;
}
AF_INLINE __f64 pow(__f64 x, int n) { return pow(x, af_i(n)); }
AF_INLINE __f64 exp2(__f64 x) { return pow(AF_TWO, x); }
AF_INLINE __f64 exp10(__f64 x) { return pow(AF_C(0x4024000000000000), x); }

// ---------------------------------------------------------------- hypot
// sqrt(x^2 + y^2) from an exact double-double sum of squares (fma), scaled
// away from overflow/underflow; < 1 ulp.
AF_INLINE __f64 hypot(__f64 x, __f64 y) {
    __f64 a = fabs(x), b = fabs(y);
    if (isinf(a) || isinf(b)) return AF_INF;
    if (isnan(a) || isnan(b)) return a + b;
    if (a < b) { __f64 t = a; a = b; b = t; }
    if (b == AF_ZERO) return a;
    int ea = SF_EXP(a.v), eb = SF_EXP(b.v);
    if (ea - eb > 60) return a + b;      // b^2 below half an ulp of a^2
    int sc = 0;
    if (ea > 0x5F0) sc = -600;           // keep a^2 far from overflow
    else if (ea < 0x210) sc = 600;       // and b^2 far from underflow
    a = ldexp(a, sc);
    b = ldexp(b, sc);
    __f64 a2 = a * a, a2l = fma(a, a, -a2);
    __f64 b2 = b * b, b2l = fma(b, b, -b2);
    __f64 sh = a2 + b2, sl = (a2 - sh) + b2 + a2l + b2l;
    __f64 r = sqrt(sh);
    // one Newton correction on the double-double radicand
    __f64 rr = fma(-r, r, sh) + sl;
    r = r + rr / (AF_TWO * r);
    return ldexp(r, -sc);
}

// ---------------------------------------------------------------- atan / atan2 (s_atan.c, e_atan2.c)
AF_INLINE __f64 atan(__f64 x) {
    const __f64 atanhi[4] = {AF_C(0x3FDDAC670561BB4F), AF_C(0x3FE921FB54442D18), AF_C(0x3FEF730BD281F69B), AF_C(0x3FF921FB54442D18)};
    const __f64 atanlo[4] = {AF_C(0x3C7A2B7F222F65E2), AF_C(0x3C81A62633145C07), AF_C(0x3C7007887AF0CBBD), AF_C(0x3C91A62633145C07)};
    const __f64 aT[11] = {AF_C(0x3FD555555555550D), AF_C(0xBFC999999998EBC4), AF_C(0x3FC24924920083FF), AF_C(0xBFBC71C6FE231671),
                          AF_C(0x3FB745CDC54C206E), AF_C(0xBFB3B0F2AF749A6D), AF_C(0x3FB10D66A0D03D51), AF_C(0xBFADDE2D52DEFD9A),
                          AF_C(0x3FA97B4B24760DEB), AF_C(0xBFA2B4442C6A6C2F), AF_C(0x3F90AD3AE322DA11)};
    int hx = af_hi(x), ix = hx & 0x7fffffff, id;
    if (ix >= 0x44100000) {
        if (ix > 0x7ff00000 || (ix == 0x7ff00000 && af_lo(x) != 0)) return x + x;
        return hx > 0 ? atanhi[3] + atanlo[3] : -atanhi[3] - atanlo[3];
    }
    if (ix < 0x3fdc0000) {
        if (ix < 0x3e200000) return x;
        id = -1;
    } else {
        x = fabs(x);
        if (ix < 0x3ff30000) {
            if (ix < 0x3fe60000) { id = 0; x = (AF_TWO * x - AF_ONE) / (AF_TWO + x); }
            else { id = 1; x = (x - AF_ONE) / (x + AF_ONE); }
        } else {
            if (ix < 0x40038000) { id = 2; x = (x - AF_C(0x3FF8000000000000)) / (AF_ONE + AF_C(0x3FF8000000000000) * x); }
            else { id = 3; x = -AF_ONE / x; }
        }
    }
    __f64 z = x * x, w = z * z;
    __f64 s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    __f64 s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0) return x - x * (s1 + s2);
    z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
    return (hx < 0) ? -z : z;
}
AF_INLINE __f64 atan2(__f64 y, __f64 x) {
    const __f64 pi_o_4 = AF_C(0x3FE921FB54442D18), pi_o_2 = AF_C(0x3FF921FB54442D18), pi = AF_C(0x400921FB54442D18),
                pi_lo = AF_C(0x3CA1A62633145C07);
    int hx = af_hi(x), ix = hx & 0x7fffffff, hy = af_hi(y), iy = hy & 0x7fffffff;
    sf_u32 lx = af_lo(x), ly = af_lo(y);
    if ((ix | ((lx | (0u - lx)) >> 31)) > 0x7ff00000 || (iy | ((ly | (0u - ly)) >> 31)) > 0x7ff00000) return x + y;
    if (((hx - 0x3ff00000) | (int)lx) == 0) return atan(y);
    int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if ((iy | (int)ly) == 0) {
        switch (m) {
        case 0: case 1: return y;
        case 2: return pi + AF_TINY;
        default: return -pi - AF_TINY;
        }
    }
    if ((ix | (int)lx) == 0) return (hy < 0) ? -pi_o_2 - AF_TINY : pi_o_2 + AF_TINY;
    if (ix == 0x7ff00000) {
        if (iy == 0x7ff00000) {
            switch (m) {
            case 0: return pi_o_4 + AF_TINY;
            case 1: return -pi_o_4 - AF_TINY;
            case 2: return AF_C(0x4002D97C7F3321D2) + AF_TINY;
            default: return AF_C(0xC002D97C7F3321D2) - AF_TINY;
            }
        } else {
            switch (m) {
            case 0: return AF_ZERO;
            case 1: return -AF_ZERO;
            case 2: return pi + AF_TINY;
            default: return -pi - AF_TINY;
            }
        }
    }
    if (iy == 0x7ff00000) return (hy < 0) ? -pi_o_2 - AF_TINY : pi_o_2 + AF_TINY;
    int k = (iy - ix) >> 20;
    __f64 z;
    if (k > 60) z = pi_o_2 + AF_HALF * pi_lo;
    else if (hx < 0 && k < -60) z = AF_ZERO;
    else z = atan(fabs(y / x));
    switch (m) {
    case 0: return z;
    case 1: return -z;
    case 2: return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
    }
}

// ---------------------------------------------------------------- sin / cos (k_sin.c, k_cos.c, e_rem_pio2.c)
AF_INLINE __f64 af_ksin(__f64 x, __f64 y, int iy) {
    const __f64 S1 = AF_C(0xBFC5555555555549), S2 = AF_C(0x3F8111111110F8A6), S3 = AF_C(0xBF2A01A019C161D5),
                S4 = AF_C(0x3EC71DE357B1FE7D), S5 = AF_C(0xBE5AE5E68A2B9CEB), S6 = AF_C(0x3DE5D93A5ACFD57C);
    int ix = af_hi(x) & 0x7fffffff;
    if (ix < 0x3e400000 && __arwen_cvt<int>(x) == 0) return x;
    __f64 z = x * x, v = z * x;
    __f64 r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if (iy == 0) return x + v * (S1 + z * r);
    return x - ((z * (AF_HALF * y - v * r) - y) - v * S1);
}
AF_INLINE __f64 af_kcos(__f64 x, __f64 y) {
    const __f64 C1 = AF_C(0x3FA555555555554C), C2 = AF_C(0xBF56C16C16C15177), C3 = AF_C(0x3EFA01A019CB1590),
                C4 = AF_C(0xBE927E4F809C52AD), C5 = AF_C(0x3E21EE9EBDB4B1C4), C6 = AF_C(0xBDA8FAE9BE8838D4);
    int ix = af_hi(x) & 0x7fffffff;
    if (ix < 0x3e400000 && __arwen_cvt<int>(x) == 0) return AF_ONE;
    __f64 z = x * x;
    __f64 r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if (ix < 0x3FD33333) return AF_ONE - (AF_HALF * z - (z * r - x * y));
    __f64 qx = (ix > 0x3fe90000) ? AF_C(0x3FD2000000000000) : af_words(ix - 0x00200000, 0);
    __f64 hz = AF_HALF * z - qx, a = AF_ONE - qx;
    return a - (hz - (z * r - x * y));
}
// x = n*pi/2 + (y0 + y1); returns n
AF_INLINE int af_rem_pio2(__f64 x, SF_THREAD __f64& y0, SF_THREAD __f64& y1) {
    const __f64 invpio2 = AF_C(0x3FE45F306DC9C883), pio2_1 = AF_C(0x3FF921FB54400000), pio2_1t = AF_C(0x3DD0B4611A626331),
                pio2_2 = AF_C(0x3DD0B4611A600000), pio2_2t = AF_C(0x3BA3198A2E037073), pio2_3 = AF_C(0x3BA3198A2E000000),
                pio2_3t = AF_C(0x397B839A252049C1);
    int hx = af_hi(x), ix = hx & 0x7fffffff;
    __f64 t = fabs(x);
    int n = __arwen_cvt<int>(t * invpio2 + AF_HALF);
    __f64 fn = af_i(n);
    __f64 r = t - fn * pio2_1, w = fn * pio2_1t;
    int j = ix >> 20;
    y0 = r - w;
    int i = j - ((af_hi(y0) >> 20) & 0x7ff);
    if (i > 16) {
        t = r;
        w = fn * pio2_2;
        r = t - w;
        w = fn * pio2_2t - ((t - r) - w);
        y0 = r - w;
        i = j - ((af_hi(y0) >> 20) & 0x7ff);
        if (i > 49) {
            t = r;
            w = fn * pio2_3;
            r = t - w;
            w = fn * pio2_3t - ((t - r) - w);
            y0 = r - w;
        }
    }
    y1 = (r - y0) - w;
    if (hx < 0) { y0 = -y0; y1 = -y1; return -n; }
    return n;
}
AF_INLINE __f64 sin(__f64 x) {
    int ix = af_hi(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return af_ksin(x, AF_ZERO, 0);
    if (ix >= 0x7ff00000) return AF_NAN;
    __f64 y0, y1;
    int n = af_rem_pio2(x, y0, y1);
    switch (n & 3) {
    case 0: return af_ksin(y0, y1, 1);
    case 1: return af_kcos(y0, y1);
    case 2: return -af_ksin(y0, y1, 1);
    default: return -af_kcos(y0, y1);
    }
}
AF_INLINE __f64 cos(__f64 x) {
    int ix = af_hi(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return af_kcos(x, AF_ZERO);
    if (ix >= 0x7ff00000) return AF_NAN;
    __f64 y0, y1;
    int n = af_rem_pio2(x, y0, y1);
    switch (n & 3) {
    case 0: return af_kcos(y0, y1);
    case 1: return -af_ksin(y0, y1, 1);
    case 2: return -af_kcos(y0, y1);
    default: return af_ksin(y0, y1, 1);
    }
}
AF_INLINE __f64 tan(__f64 x) { return sin(x) / cos(x); }

// ---------------------------------------------------------------- cbrt
// float estimate, exponent handled exactly in multiples of 3, two Newton steps
// in binary64 (error < 1 ulp + rounding of the last step).
AF_INLINE __f64 cbrt(__f64 x) {
    if (x == AF_ZERO || !isfinite(x)) return x;
    bool neg = signbit(x);
    __f64 a = fabs(x);
    int e;
    __f64 m = frexp(a, &e);            // a = m * 2^e, m in [0.5, 1)
    int r3 = ((e % 3) + 3) % 3;
    m = ldexp(m, r3);                  // m in [0.5, 4)
    int k = (e - r3) / 3;
    float mf = sf_to_f32(m.v);
#ifdef __METAL_VERSION__
    float y0 = metal::precise::pow(mf, 1.0f / 3.0f);
#else
    float y0 = std::cbrt(mf);
#endif
    __f64 y = __f64{sf_from_f32(y0)};
    const __f64 three = AF_C(0x4008000000000000);
    y = y - (y * y * y - m) / (three * y * y);
    y = y - (y * y * y - m) / (three * y * y);
    y = ldexp(y, k);
    return neg ? -y : y;
}

// ---------------------------------------------------------------- lgamma
// x > 0: shift to z = x + n >= 16 and use Stirling's series to z^-15 (series
// error < 1e-19), lgamma(x) = stirling(z) - log(x (x+1) ... (x+n-1)).
// x < 0: reflection log|pi / (x sin(pi x))| - lgamma(-x).  Absolute error
// ~1e-15 (relative accuracy is lost near the zeros at 1 and 2).
AF_INLINE __f64 af_lgamma_pos(__f64 x) {
    __f64 prod = AF_ONE, z = x;
    while (z < AF_C(0x4030000000000000)) { prod = prod * z; z = z + AF_ONE; }   // 16
    const __f64 hl2pi = AF_C(0x3FED67F1C864BEB4);                                 // 0.5*log(2*pi)
    __f64 iz = AF_ONE / z, iz2 = iz * iz;
    // Bernoulli terms B2k / (2k (2k-1)) for k = 1..8
    const __f64 c1 = AF_C(0x3FB5555555555555), c2 = AF_C(0xBF66C16C16C16C17), c3 = AF_C(0x3F4A01A01A01A01A),
                c4 = AF_C(0xBF43813813813814), c5 = AF_C(0x3F4B951E2B18FF23), c6 = AF_C(0xBF5F6AB0D9993C7D),
                c7 = AF_C(0x3F7A41A41A41A41A), c8 = AF_C(0xBF9E4286CB0F5398);
    __f64 ser = iz * (c1 + iz2 * (c2 + iz2 * (c3 + iz2 * (c4 + iz2 * (c5 + iz2 * (c6 + iz2 * (c7 + iz2 * c8)))))));
    __f64 st = (z - AF_HALF) * log(z) - z + hl2pi + ser;
    return st - log(prod);
}
AF_INLINE __f64 lgamma(__f64 x) {
    if (isnan(x)) return x;
    if (isinf(x)) return AF_INF;
    if (x == AF_ZERO) return AF_INF;
    if (x > AF_ZERO) return af_lgamma_pos(x);
    if (floor(x) == x) return AF_INF;                  // poles at non-positive integers
    const __f64 pi = AF_C(0x400921FB54442D18);
    __f64 f = x - floor(x);                             // exact, in [0, 1)
    if (f > AF_HALF) f = f - AF_ONE;                    // exact; keeps pi*f away from pi
    __f64 s = sin(pi * f);                              // |sin(pi x)|, small argument -> small relative error
    __f64 r = log(pi / fabs(x * s));
    return r - af_lgamma_pos(-x);
}
