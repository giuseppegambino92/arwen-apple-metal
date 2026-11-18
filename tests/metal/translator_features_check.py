"""Translator constructs added for full-corpus coverage are exact: the same C++
functions, compiled natively by clang on the host and translated to Metal, give
bit-identical results on 100,000 pseudo-random inputs each.

Covered: conditional operator with a type-dependent branch (float vs double and
int vs float), dependent ``double(e)`` / ``T(e)`` constructs and copy-
initialisation (``T r = 1``, ``double q = p[i]`` with p a ``R*``), generic
lambdas with reference parameters bound to thread and device objects, C brace
elision (``float m[2][3] = {1, ..., 6}``, array of records), overloads told
apart by pointee type, pointer tables over local arrays, multi-declarator
references, block-scope prototypes, an RAII destructor that needs the launch
context, program-scope ``__shared__`` arrays, a backward ``goto`` that restarts
a loop nest from inside it, a derived struct passed to a ``Base&`` parameter,
and an assignment to ``(c ? local : device[i])``.  Plus PTX ``setp.ne.f32``
(ordered: false on NaN), checked against its PTX ISA definition.
Declared tolerance: 0 differing words."""
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401,E402
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402

FUNCS = r'''
template <class A, class B> DEV auto tmax(A a, B b) -> decltype(a + b) { return a > b ? a : b; }
template <class A, class B> DEV auto tdiv(A a, B b) -> decltype(a + b) { using T = decltype(a + b); return T(a) / T(b); }
template <class T> DEV T tpow(T a, int n) { T r = 1; for (int k = 0; k < n; ++k) r = r * a; return r; }
template <class R> DEV double tsum(const R* p, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) { double q = p[i]; s += q * double(p[i]); }
    return s;
}
DEV int pick(const float* p) { return 1; }
DEV int pick(const double* p) { return 2; }
DEV float elided(float x) {
    float m[2][3] = {1.5f, 2.5f, x, 4.0f, x * x, 6.0f};
    struct P { int a; float b[2]; };
    P arr[2] = {1, 2.5f, x, 4, 5.5f, 6.5f};
    return m[0][2] + m[1][1] * 0.5f + arr[1].b[0] - arr[0].b[1] + (float)(arr[0].a + arr[1].a);
}
DEV float lam(float x, float* dev) {
    float a = x, b = 2.0f * x;
    auto bump = [&](int j, auto& u, auto& w) { u += (float)j; w = w * u; };
    bump(3, a, b);
    bump(1, dev[0], dev[1]);
    return a + b;
}
DEV float table(float x) {
    float p[2] = {x, x + 1.0f}, q[2] = {2.0f * x, 3.0f};
    const float* v[2] = {p, q};
    float s = 0.0f;
    for (int k = 0; k < 2; ++k) s += v[k][0] * v[k][1];
    return s;
}
DEV float refs(float* a) { float &x = a[0], &y = a[1]; x += y; y *= 2.0f; return x - y; }
DEV int proto_user(int a) {
    extern DEV int proto_target(int);
    return proto_target(a) + 1;
}
DEV int proto_target(int a) { return a * 3; }
DEV int restart(int a) {                      /* goto back to a label before a loop nest */
    int n = 3 + (a & 3), acc = 0, merged = 0;
again:
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) {
            acc += i * k + 1;
            if (((acc ^ a) & 31) == 7 && n > 2 && merged < 3) { n -= 1; ++merged; goto again; }
        }
        acc ^= i;
    }
    return acc * 8 + merged;
}
struct Base { float tr; float tg; };
struct Derived : Base { float tv; };
DEV void load_base(const float* sp, Base& c) { c.tr = sp[0] * 2.0f; c.tg = sp[1]; }
DEV float load_derived(const float* sp, Derived& c) { load_base(sp, c); c.tv = sp[2]; return c.tr + c.tg * c.tv; }
'''

DRIVER = r'''
struct Ws { int top; };
__shared__ int arena_top[256];
struct Mark {
    int saved;
    __device__ Mark() : saved(arena_top[threadIdx.x]) {}
    __device__ ~Mark() { arena_top[threadIdx.x] = saved; }
};
__device__ int raii(int k) {
    Mark m;
    arena_top[threadIdx.x] += k;
    if (k > 5) return arena_top[threadIdx.x] * 10;
    arena_top[threadIdx.x] += 1;
    return arena_top[threadIdx.x];
}
extern "C" __global__ void run(const float* fa, const double* da, const int* ia, float* fout, double* dout,
                               int* iout, float* dev, float* sel, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float x = fa[t]; double d = da[t]; int i = ia[t];
    dout[4 * t + 0] = tmax(x, d);
    fout[8 * t + 0] = tmax(i, x);
    dout[4 * t + 1] = tdiv(x, d);
    dout[4 * t + 2] = tpow(d, i & 7);
    float loc[3] = {x, x * 0.5f, x + 2.0f};
    dout[4 * t + 3] = tsum(loc, 3);
    fout[8 * t + 1] = elided(x);
    dev[2 * t] = x; dev[2 * t + 1] = 1.5f;
    fout[8 * t + 2] = lam(x, dev + 2 * t);
    fout[8 * t + 3] = dev[2 * t] + dev[2 * t + 1];
    fout[8 * t + 4] = table(x);
    float ab[2] = {x, 0.25f};
    fout[8 * t + 5] = refs(ab) + ab[0] * ab[1];
    double dd = d;
    iout[4 * t + 0] = pick(&x) * 10 + pick(&dd);
    iout[4 * t + 1] = proto_user(i);
    arena_top[threadIdx.x] = i & 15;
    int r = raii(i & 7);
    iout[4 * t + 2] = r * 100 + arena_top[threadIdx.x];
    unsigned ne;
    float nb = (i & 1) ? __int_as_float(0x7fc00000) : x;
    float x1 = x + 1.0f;
    asm("{ .reg .pred p; setp.ne.f32 p,%1,%2; selp.u32 %0,1,0,p; }" : "=r"(ne) : "f"(nb), "f"(x1));
    iout[4 * t + 3] = (int)ne;
    fout[8 * t + 6] = (float)restart(i);
    float sp3[3] = {x, 0.5f, x + 3.0f};
    Derived dv;
    fout[8 * t + 7] = load_derived(sp3, dv) + dv.tv;
    float lv = 0.0f;
    for (int k = 0; k < 2; ++k) ((i + k) & 1 ? lv : sel[2 * t + k]) = x * (float)(k + 1);
    sel[2 * t] += lv;
}
'''

HOST = r'''
#include <math.h>
#define DEV
''' + FUNCS + r'''
static int arena_top_h;
static int raii_h(int k) {
    int saved = arena_top_h; int ret;
    arena_top_h += k;
    if (k > 5) { ret = arena_top_h * 10; arena_top_h = saved; return ret; }
    arena_top_h += 1; ret = arena_top_h; arena_top_h = saved; return ret;
}
extern "C" void run_host(const float* fa, const double* da, const int* ia, float* fout, double* dout,
                         int* iout, float* dev, float* sel, int n) {
    for (int t = 0; t < n; ++t) {
        float x = fa[t]; double d = da[t]; int i = ia[t];
        dout[4 * t + 0] = tmax(x, d);
        fout[8 * t + 0] = tmax(i, x);
        dout[4 * t + 1] = tdiv(x, d);
        dout[4 * t + 2] = tpow(d, i & 7);
        float loc[3] = {x, x * 0.5f, x + 2.0f};
        dout[4 * t + 3] = tsum(loc, 3);
        fout[8 * t + 1] = elided(x);
        dev[2 * t] = x; dev[2 * t + 1] = 1.5f;
        fout[8 * t + 2] = lam(x, dev + 2 * t);
        fout[8 * t + 3] = dev[2 * t] + dev[2 * t + 1];
        fout[8 * t + 4] = table(x);
        float ab[2] = {x, 0.25f};
        fout[8 * t + 5] = refs(ab) + ab[0] * ab[1];
        double dd = d;
        iout[4 * t + 0] = pick(&x) * 10 + pick(&dd);
        iout[4 * t + 1] = proto_user(i);
        arena_top_h = i & 15;
        int r = raii_h(i & 7);
        iout[4 * t + 2] = r * 100 + arena_top_h;
        float nb = (i & 1) ? NAN : x;
        iout[4 * t + 3] = (!isnan(nb) && !isnan(x + 1.0f) && nb != x + 1.0f) ? 1 : 0;   /* PTX ordered ne */
        fout[8 * t + 6] = (float)restart(i);
        float sp3[3] = {x, 0.5f, x + 3.0f};
        Derived dv;
        fout[8 * t + 7] = load_derived(sp3, dv) + dv.tv;
        float lv = 0.0f;
        for (int k = 0; k < 2; ++k) ((i + k) & 1 ? lv : sel[2 * t + k]) = x * (float)(k + 1);
        sel[2 * t] += lv;
    }
}
'''


DEP_SHARED = r'''
template <int K> __device__ void col(const float* in, float* out) {
    __shared__ float a[K];
    __shared__ float b[K + 1];
    __shared__ int c;
    int t = threadIdx.x;
    if (t < K) a[t] = in[blockIdx.x * K + t];
    if (t <= K) b[t] = 100.0f * (float)t;
    if (t == 0) c = K;
    __syncthreads();
    if (t < K) out[blockIdx.x * 2 * K + t] = a[K - 1 - t] + b[t + 1] + (float)c;
    if (t < K) out[blockIdx.x * 2 * K + K + t] = a[t] * 2.0f;
}
extern "C" __global__ void k32(const float* in, float* out) { col<32>(in, out); }
extern "C" __global__ void k64(const float* in, float* out) { col<64>(in, out); }
'''


def dependent_shared():
    """``__shared__ T a[K]`` in ``template <int K>``: laid out per instantiation."""
    m = cp.RawModule(code=DEP_SHARED, options=("-std=c++17", "--ftz=false"))
    bad = 0
    for k, name in ((32, "k32"), (64, "k64")):
        nb = 7
        x = np.random.default_rng(k).uniform(-5, 5, nb * k).astype(np.float32)
        o = cp.zeros(nb * 2 * k, cp.float32)
        m.get_function(name)((nb,), (k + 1,), (cp.asarray(x), o))
        got = cp.asnumpy(o).reshape(nb, 2 * k)
        xr = x.reshape(nb, k)
        exp = np.concatenate([xr[:, ::-1] + 100.0 * np.arange(1, k + 1, dtype=np.float32) + k, xr * 2],
                             axis=1).astype(np.float32)
        d = int(np.count_nonzero(got.view(np.uint32) != exp.view(np.uint32)))
        bad += d
        print(f"{'dependent __shared__ <' + str(k) + '>':26s} differing words {d} of {got.size}")
    return bad


def main():
    n = 100_000
    rng = np.random.default_rng(11)
    fa = rng.uniform(-40.0, 40.0, n).astype(np.float32)
    fa[::97] = np.float32(1e-41)                     # subnormal inputs too
    da = rng.uniform(-1e3, 1e3, n)
    ia = rng.integers(0, 1 << 20, n).astype(np.int32)
    ref = dict(f=np.zeros(8 * n, np.float32), d=np.zeros(4 * n), i=np.zeros(4 * n, np.int32),
               dev=np.zeros(2 * n, np.float32), sel=np.zeros(2 * n, np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        c, so = os.path.join(tmp, "f.cpp"), os.path.join(tmp, "f.so")
        open(c, "w").write(HOST)
        subprocess.run(["clang++", "-std=c++17", "-O1", "-ffp-contract=off", "-shared", "-fPIC", c, "-o", so],
                       check=True)
        lib = ctypes.CDLL(so)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        lib.run_host(p(fa), p(da), p(ia), p(ref["f"]), p(ref["d"]), p(ref["i"]), p(ref["dev"]), p(ref["sel"]),
                     ctypes.c_int(n))
    src = "#define DEV __device__\n" + FUNCS + DRIVER
    # IEEE semantics as on the host (subnormals kept): --ftz=false explicitly
    k = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false", "--ftz=false")).get_function("run")
    fo, do, io, dv, sl = (cp.zeros(8 * n, cp.float32), cp.zeros(4 * n, cp.float64), cp.zeros(4 * n, cp.int32),
                          cp.zeros(2 * n, cp.float32), cp.zeros(2 * n, cp.float32))
    k(((n + 255) // 256,), (256,), (cp.asarray(fa), cp.asarray(da), cp.asarray(ia), fo, do, io, dv, sl, np.int32(n)))
    got = dict(f=cp.asnumpy(fo), d=cp.asnumpy(do), i=cp.asnumpy(io), dev=cp.asnumpy(dv), sel=cp.asnumpy(sl))
    names = {"f": ["tmax(int,float)", "brace elision", "generic lambda", "lambda on device", "pointer table",
                   "multi-decl refs", "goto restart from loops", "derived -> Base& param"],
             "d": ["tmax(float,double)", "T(a)/T(b)", "T r = 1", "double q = p[i] (R*)"],
             "i": ["overload by pointee", "block-scope prototype", "RAII dtor + __shared__", "setp.ne ordered"]}
    bad = 0
    for key, cols in names.items():
        w = len(cols)
        for j, name in enumerate(cols):
            if name is None:
                continue
            a, b = got[key][j::w], ref[key][j::w]
            d = int(np.count_nonzero(a.view(f"u{a.itemsize}") != b.view(f"u{b.itemsize}")))
            bad += d
            print(f"{name:26s} differing words {d} of {n}")
    d = int(np.count_nonzero(got["dev"].view(np.uint32) != ref["dev"].view(np.uint32)))
    bad += d
    print(f"{'device memory via lambda':26s} differing words {d} of {2 * n}")
    d = int(np.count_nonzero(got["sel"].view(np.uint32) != ref["sel"].view(np.uint32)))
    bad += d
    print(f"{'(c ? local : dev[i]) = v':26s} differing words {d} of {2 * n}")
    bad += dependent_shared()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
