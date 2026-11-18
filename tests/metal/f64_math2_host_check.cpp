// ULP error of arwen_f64_math2.h (on the sf64 __f64) against the host libm.
// Declared tolerance (CUDA libdevice documents 1-2 ulp for these; erf/erfc/tgamma up to a few):
//   expm1 log1p tanh sinh cosh asinh acosh atanh asin acos <= 4 ulp; erf <= 16, erfc <= 64, tgamma <= 16 ulp;
//   remainder nextafter exact (0 ulp).
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
typedef uint64_t ulong; typedef uint32_t uint; typedef unsigned short ushort; typedef unsigned char uchar;
#define ARWEN_F64_MODE_SF64 1
namespace af {
#include "../../gpuwm/metal/include/arwen_f64.h"
}
using af::__f64;
static double D(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t B(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static int64_t ord(uint64_t u) { return (u >> 63) ? -(int64_t)(u & 0x7FFFFFFFFFFFFFFFULL) : (int64_t)u; }
static long ulps(uint64_t a, double r) {
    if (std::isnan(r)) return std::isnan(D(a)) ? 0 : 1L << 40;
    if (std::isinf(r)) return B(r) == a ? 0 : 1L << 40;
    int64_t d = ord(a) - ord(B(r)); return d < 0 ? -d : d;
}
int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0, 1);
    const char* names[] = {"expm1", "log1p", "tanh", "sinh", "cosh", "asinh", "acosh", "atanh", "asin", "acos",
                           "erf", "erfc", "tgamma", "remainder", "nextafter"};
    const long tol[] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 16, 64, 16, 0, 0};
    long maxu[15] = {0}, bad[15] = {0};
    auto rec = [&](int k, uint64_t got, double ref) { long u = ulps(got, ref); if (u) bad[k]++; if (u > maxu[k]) maxu[k] = u; };
    auto sgn = [&]() { return (rng() & 1) ? 1.0 : -1.0; };
    for (long i = 0; i < n; ++i) {
        double s = sgn() * std::ldexp(0.5 + U(rng), (int)(rng() % 70) - 60);           // small and moderate
        double m = sgn() * 40 * U(rng);
        rec(0, af::expm1(__f64{B(s)}).v, std::expm1(s)); rec(0, af::expm1(__f64{B(m)}).v, std::expm1(m));
        double lp = (rng() & 1) ? std::fabs(s) : -0.999 * U(rng);
        rec(1, af::log1p(__f64{B(lp)}).v, std::log1p(lp));
        rec(2, af::tanh(__f64{B(s)}).v, std::tanh(s)); rec(2, af::tanh(__f64{B(m)}).v, std::tanh(m));
        rec(3, af::sinh(__f64{B(s)}).v, std::sinh(s)); rec(3, af::sinh(__f64{B(m)}).v, std::sinh(m));
        rec(4, af::cosh(__f64{B(m)}).v, std::cosh(m));
        double big = sgn() * std::ldexp(0.5 + U(rng), (int)(rng() % 80) - 40);
        rec(5, af::asinh(__f64{B(big)}).v, std::asinh(big));
        double ac = 1.0 + std::ldexp(U(rng), (int)(rng() % 70) - 50);
        rec(6, af::acosh(__f64{B(ac)}).v, std::acosh(ac));
        double at = sgn() * ((rng() & 1) ? U(rng) : std::ldexp(U(rng), -(int)(rng() % 40)));
        rec(7, af::atanh(__f64{B(at)}).v, std::atanh(at));
        rec(8, af::asin(__f64{B(at)}).v, std::asin(at));
        rec(9, af::acos(__f64{B(at)}).v, std::acos(at));
        double e1 = sgn() * 6.5 * U(rng);
        rec(10, af::erf(__f64{B(e1)}).v, std::erf(e1));
        double e2 = (rng() & 3) ? -1.0 + 28 * U(rng) : sgn() * std::ldexp(U(rng), -(int)(rng() % 30));
        rec(11, af::erfc(__f64{B(e2)}).v, std::erfc(e2));
        double g = (rng() & 3) ? 0.01 + 170 * U(rng) : -20 * U(rng);
        rec(12, af::tgamma(__f64{B(g)}).v, std::tgamma(g));
        double rx = sgn() * std::ldexp(U(rng), (int)(rng() % 60) - 20), ry = sgn() * std::ldexp(0.5 + U(rng), (int)(rng() % 30) - 10);
        if (rng() % 8 == 0) rx = ry * (double)((int)(rng() % 9)) * 0.5;                     // exact ties
        rec(13, af::remainder(__f64{B(rx)}, __f64{B(ry)}).v, std::remainder(rx, ry));
        rec(14, af::nextafter(__f64{B(rx)}, __f64{B(ry)}).v, std::nextafter(rx, ry));
    }
    int fail = 0;
    for (int k = 0; k < 15; ++k) {
        printf("%-10s differ_from_libm=%ld max_ulp=%ld (tolerance %ld)%s\n", names[k], bad[k], maxu[k], tol[k], maxu[k] > tol[k] ? "  FAIL" : "");
        fail |= maxu[k] > tol[k];
    }
    return fail;
}
