"""Goto elimination (gpuwm/metal/transpile.py) is exact: the same C functions,
compiled natively by clang (real gotos) and translated to Metal (gotos
eliminated), give identical results on 200,000 pseudo-random inputs each.

Shapes covered: a Fortran-style iteration (backward goto) with forward exits,
several gotos to one label, a forward goto out of two nested loops, a goto
out of a switch, a label inside a loop body (the flag must be cleared on every
pass), two consecutive backward-goto iterations, a goto skipping statements in
the middle of a block, a goto into a sibling branch of an if (NoahMP runoff); and SSiB's real NEWTON routine (consecutive labels,
interleaved backward and forward gotos).  Declared tolerance: 0 differing words."""
import ctypes
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401,E402
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402

FUNCS = r'''
DEV int iter_newton(float a, int n0) {           /* backward goto + forward exits (KF style) */
    int it = 0; float x = a, f;
    x = x * 0.5f + 1.0f;
L90:
    f = x * x - a;
    if (FABS(f) < 1e-3f) goto L50;
    it = it + 1;
    if (it > 8 + (n0 & 3)) goto L50;
    x = x - f / (2.0f * x);
    goto L90;
L50:
    if (x > 3.0f) { it += 100; goto L96; }
    it += 1000;
    if (it > 1005) goto L96;
    it += 7;
L96:
    return it * 3 + (int)(x * 16.0f);
}
DEV int nested_exit(int n, int m, int key) {     /* forward goto out of two loops */
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            acc += i * 7 + j;
            if (((acc ^ key) & 63) == 5) goto out;
        }
        acc += 1;
        if (acc > key + 400) goto out;
    }
    acc -= 3;
out:
    return acc;
}
DEV int switch_exit(int a, int b) {              /* goto out of a switch inside a loop */
    int r = 0;
    for (int k = 0; k < 12; ++k) {
        switch ((a + k) & 3) {
        case 0: r += 1; break;
        case 1: r += b; if (r > 50) goto done; break;
        case 2: r -= 2; break;
        default: r *= 2; if (r > 900) goto done;
        }
        r += k;
    }
    r += 10000;
done:
    return r;
}
DEV int label_in_loop(int a) {                   /* label inside the loop body: flag cleared per pass */
    int s = 0;
    for (int i = 0; i < 20; ++i) {
        if (((a + i) % 5) == 0) goto skip;
        s += i;
        s ^= a;
skip:
        s += 1;
    }
    return s;
}
DEV int two_iterations(int a) {                  /* two consecutive backward-goto loops */
    int i = 0, j = 0;
L10:
    i += 3;
    if (i < (a & 31)) goto L10;
L20:
    j += i;
    i -= 1;
    if (i > 0) goto L20;
    return j;
}
DEV float ss_sign(float a, float b) { return copysignf(fabsf(a), b); }
/* the real SSiB NEWTON (gpuwm/core/kernels/ssib.cu): Fortran spaghetti, consecutive
   labels L160: L150:, backward gotos L2/L3 and forward exits */
DEV void ss_newton(float &a1, float &y, float finc, int &nox,
                          int nonpos, int &iwolk, int l, float *zinc,
                          float *a2, float *y1, float *iter)
{
    // IWALK and NEX are REAL(3) locals; only element L is ever read, and it
    // is set from the arguments on entry.  ITER (the caller's ITEX) is REAL.
    const float cons = 1.0f;
    float iwalk[4] = {0.f}, nex[4] = {0.f};
    float ertol = 0.f, a = 0.f;
    ertol = 0.05f * finc;
    iwalk[l] = (float)iwolk;
    nex[l] = (float)nox;
    if (iter[l] >= 490.f) goto L160;
    if (ertol < 0.00000001f) ertol = 0.000001f;
    if (fabsf(y) <= ertol) goto L150;
    if ((fabsf(y - y1[l])) <= 0.01f * ertol && iwalk[l] == 0.f) goto L8;
    if (fabsf(y1[l]) > ertol) goto L1;
    a2[l] = a1;
    a1 = a1 - y;
    nex[l] = 0.f;
    y1[l] = y;
    iter[l] = 1.f;
    if (iwalk[l] == 3.f) goto L101;
    iwalk[l] = 0.f;
    goto L101;
L1:
    iter[l] = iter[l] + 1.f;
    if (iter[l] == 10.f) iwalk[l] = 1.f;
    if (iwalk[l] != 0.f) goto L2;
    if (fabsf(y) > ertol) goto L3;
    nex[l] = 1.f;
    goto L150;
L3:
    a = a1 - y * (a1 - a2[l]) / (y - y1[l]);
    if (fabsf(a - a1) > (10.0f * finc))
        a = a1 + 10.0f * finc * ss_sign(cons, (a - a1));
    a2[l] = a1;
    a1 = a;
    y1[l] = y;
    goto L101;
L2:
    if (iwalk[l] == 2.f) goto L4;
    if (iwalk[l] == 3.f) goto L6;
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L3;
    zinc[l] = (a1 - a2[l]) / 4.0f;
    a1 = a2[l] + zinc[l];
    iwalk[l] = 2.f;
    nex[l] = 0.f;
    goto L101;
L4:
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L5;
    zinc[l] = -zinc[l] / 4.0f;
    a2[l] = a1;
    a1 = a1 + zinc[l];
    nex[l] = 0.f;
    y1[l] = y;
    goto L101;
L5:
    a2[l] = a1;
    a1 = a1 + zinc[l];
    y1[l] = y;
    nex[l] = 0.f;
    goto L101;
L6:
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L7;
    iwalk[l] = 1.f;
    goto L2;
L7:
    a2[l] = a1;
    a1 = a1 + finc;
    y1[l] = y;
    nex[l] = 0.f;
    goto L101;
L8:
    a1 = a1 + finc * 2.0f;
    nex[l] = 0.f;
    goto L101;
L160:
L150:
    nex[l] = 1.f;
    zinc[l] = 0.0f;
    iter[l] = 0.f;
    iwalk[l] = 0.f;
    y1[l] = 0.0f;
    y = 0.0f;
    a2[l] = 0.0f;
L101:
    if (nonpos == 1 && a1 < 0.0f) a1 = a2[l] / 2.0f;
    nox = (int)nex[l];
    iwolk = (int)iwalk[l];
}
DEV int into_branch(int a, int b) {              /* goto into a sibling branch (NoahMP runoff) */
    int r = a & 255, it;
    if (a & 1) {
        r += 3;
        goto L2001;
    } else {
        if (a & 2) {
            for (it = 0; it < 9; ++it) {
                r = r * 3 + it;
                if (((r ^ b) & 7) == 1 || it == 8) {
                    goto L1003;
                }
            }
        } else if (a & 4) {
            r -= b & 63;
            if (r < 0) goto L1003;
            r += 1;
        } else {
            for (it = 0; it < 5; ++it) {
                r += it * b;
                if ((r & 15) == 3) break;
            }
            L1003: ;
            if (r > 100) r = 100;
            r = r * 2 + 1;
            goto L2001;
        }
        r += 1000;
    }
L2001:
    return r;
}
DEV int middle_skip(int a, int b) {              /* several forward gotos, statements in between */
    int r = a;
    if (a > b) goto L1;
    r += 5;
    if (r & 1) goto L1;
    r *= 3;
    if (r > 400) goto L2;
L1:
    r -= b;
    if (r < 0) goto L2;
    r += 11;
L2:
    return r;
}
'''

DRIVER = r'''
extern "C" __global__ void run(const int* ia, const float* fa, int* out, float* fo, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int a = ia[t], b = ia[n + t];
    { float a1 = fa[t], y = fa[t] - 3.0f + (float)(b & 7), finc = 0.25f + 0.01f * (float)(a & 15);
      int nox = a & 1, iwolk = (a >> 1) & 3, l = (a >> 3) & 3;
      float zinc[4] = {0.1f, 0.2f, 0.3f, 0.4f}, a2[4] = {fa[t] * 0.5f, 1.0f, 2.0f, 3.0f};
      float y1[4] = {y * 0.9f, y + 0.01f, 0.0f, y}, iter[4] = {(float)(b & 15), (float)(b % 495), 9.0f, 489.0f};
      ss_newton(a1, y, finc, nox, (b >> 4) & 1, iwolk, l, zinc, a2, y1, iter);
      fo[8 * t + 0] = a1; fo[8 * t + 1] = y; fo[8 * t + 2] = zinc[l]; fo[8 * t + 3] = a2[l];
      fo[8 * t + 4] = y1[l]; fo[8 * t + 5] = iter[l]; fo[8 * t + 6] = (float)nox; fo[8 * t + 7] = (float)iwolk; }
    out[7 * t + 0] = iter_newton(fa[t], a);
    out[7 * t + 1] = nested_exit(a & 15, b & 15, a ^ b);
    out[7 * t + 2] = switch_exit(a, b & 31);
    out[7 * t + 3] = label_in_loop(a);
    out[7 * t + 4] = two_iterations(a);
    out[7 * t + 5] = middle_skip(a & 511, b & 511);
    out[7 * t + 6] = into_branch(a, b);
}
'''

HOST = r'''
#include <math.h>
#define DEV static
#define FABS fabsf
#define __forceinline__
''' + FUNCS + r'''
extern "C" void run_host(const int* ia, const float* fa, int* out, float* fo, int n) {
    for (int t = 0; t < n; ++t) {
        int a = ia[t], b = ia[n + t];
        { float a1 = fa[t], y = fa[t] - 3.0f + (float)(b & 7), finc = 0.25f + 0.01f * (float)(a & 15);
      int nox = a & 1, iwolk = (a >> 1) & 3, l = (a >> 3) & 3;
      float zinc[4] = {0.1f, 0.2f, 0.3f, 0.4f}, a2[4] = {fa[t] * 0.5f, 1.0f, 2.0f, 3.0f};
      float y1[4] = {y * 0.9f, y + 0.01f, 0.0f, y}, iter[4] = {(float)(b & 15), (float)(b % 495), 9.0f, 489.0f};
      ss_newton(a1, y, finc, nox, (b >> 4) & 1, iwolk, l, zinc, a2, y1, iter);
      fo[8 * t + 0] = a1; fo[8 * t + 1] = y; fo[8 * t + 2] = zinc[l]; fo[8 * t + 3] = a2[l];
      fo[8 * t + 4] = y1[l]; fo[8 * t + 5] = iter[l]; fo[8 * t + 6] = (float)nox; fo[8 * t + 7] = (float)iwolk; }
        out[7 * t + 0] = iter_newton(fa[t], a);
        out[7 * t + 1] = nested_exit(a & 15, b & 15, a ^ b);
        out[7 * t + 2] = switch_exit(a, b & 31);
        out[7 * t + 3] = label_in_loop(a);
        out[7 * t + 4] = two_iterations(a);
        out[7 * t + 5] = middle_skip(a & 511, b & 511);
        out[7 * t + 6] = into_branch(a, b);
    out[7 * t + 6] = into_branch(a, b);
    }
}
'''


def main():
    n = 200_000
    rng = np.random.default_rng(5)
    ia = rng.integers(0, 1 << 20, 2 * n).astype(np.int32)
    fa = rng.uniform(0.05, 50.0, n).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "g.cpp")
        so = os.path.join(d, "g.so")
        open(c, "w").write(HOST)
        subprocess.run(["clang++", "-O1", "-ffp-contract=off", "-shared", "-fPIC", c, "-o", so], check=True)
        lib = ctypes.CDLL(so)
        ref = np.empty(7 * n, np.int32)
        fref = np.zeros(8 * n, np.float32)
        lib.run_host(ia.ctypes.data_as(ctypes.c_void_p), fa.ctypes.data_as(ctypes.c_void_p),
                     ref.ctypes.data_as(ctypes.c_void_p), fref.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
    src = "#define DEV __device__\n#define FABS fabsf\n" + FUNCS + DRIVER
    k = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false", "--ftz=false")).get_function("run")
    out = cp.empty(7 * n, cp.int32)
    fo = cp.zeros(8 * n, cp.float32)
    k(((n + 255) // 256,), (256,), (cp.asarray(ia), cp.asarray(fa), out, fo, np.int32(n)))
    got = cp.asnumpy(out)
    fgot = cp.asnumpy(fo)
    bad = 0
    for j, name in enumerate(("iter_newton", "nested_exit", "switch_exit", "label_in_loop", "two_iterations",
                              "middle_skip", "into_branch")):
        d = int(np.count_nonzero(got[j::7] != ref[j::7]))
        bad += d
        print(f"{name:16s} differing words {d} of {n}")
    d = int(np.count_nonzero(fgot.view(np.uint32) != fref.view(np.uint32)))
    bad += d
    print(f"{'ssib ss_newton':16s} differing words {d} of {8 * n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
