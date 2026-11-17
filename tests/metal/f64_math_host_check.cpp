// ULP error of arwen_f64_math.h (on the sf64 __f64) against the host libm.
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
    int64_t d = ord(a) - ord(B(r)); return d < 0 ? -d : d;
}
int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> U(0, 1);
    double lg_maxabs = 0, lg_maxabs_neg = 0;
    struct Stat { const char* name; long maxu = 0, bad = 0, cnt = 0; } st[11] = {{"exp"}, {"log"}, {"log10"}, {"pow"}, {"atan"}, {"atan2"}, {"sin"}, {"cos"}, {"hypot"}, {"cbrt"}, {"lgamma"}};
    auto rec = [&](int k, uint64_t got, double ref) { long u = ulps(got, ref); st[k].cnt++; if (u) st[k].bad++; if (u > st[k].maxu) st[k].maxu = u; };
    for (long i = 0; i < n; ++i) {
        double e = -745 + 1455 * U(rng);                       // exp domain incl. subnormal results
        rec(0, af::exp(__f64{B(e)}).v, std::exp(e));
        double l = std::ldexp(0.5 + U(rng), (int)(rng() % 2098) - 1074);
        rec(1, af::log(__f64{B(l)}).v, std::log(l));
        rec(2, af::log10(__f64{B(l)}).v, std::log10(l));
        double px = std::ldexp(0.5 + U(rng), (int)(rng() % 40) - 20) * ((rng() & 7) ? 1 : -1);
        double py = (rng() & 3) ? (U(rng) - 0.5) * 60 : (double)((int)(rng() % 41) - 20);
        rec(3, af::pow(__f64{B(px)}, __f64{B(py)}).v, std::pow(px, py));
        double a = std::ldexp(U(rng) - 0.5, (int)(rng() % 80) - 40);
        rec(4, af::atan(__f64{B(a)}).v, std::atan(a));
        double ay = (U(rng) - 0.5) * std::ldexp(1, (int)(rng() % 40) - 20), ax = (U(rng) - 0.5) * std::ldexp(1, (int)(rng() % 40) - 20);
        rec(5, af::atan2(__f64{B(ay)}, __f64{B(ax)}).v, std::atan2(ay, ax));
        double t = (U(rng) - 0.5) * std::ldexp(1, (int)(rng() % 24) - 2);   // |t| up to ~2e6
        rec(6, af::sin(__f64{B(t)}).v, std::sin(t));
        rec(7, af::cos(__f64{B(t)}).v, std::cos(t));
        double h1 = std::ldexp(U(rng), (int)(rng() % 2000) - 1000), h2 = std::ldexp(U(rng), (int)(rng() % 2000) - 1000);
        rec(8, af::hypot(__f64{B(h1)}, __f64{B(h2)}).v, std::hypot(h1, h2));
        double cb = std::ldexp(U(rng) - 0.5, (int)(rng() % 2000) - 1000);
        rec(9, af::cbrt(__f64{B(cb)}).v, std::cbrt(cb));
        double lg = (rng() & 1) ? 0.05 + 60 * U(rng) : -30 * U(rng);
        { double g = D(af::lgamma(__f64{B(lg)}).v), r = std::lgamma(lg); st[10].cnt++; double ae = std::fabs(g - r);
          long u = ulps(B(g), r); if (u) st[10].bad++; if (lg > 0 && ae > lg_maxabs) lg_maxabs = ae; if (lg < 0 && ae > lg_maxabs_neg) lg_maxabs_neg = ae; if (u > st[10].maxu && std::fabs(r) > 0.5) st[10].maxu = u; }
    }
    printf("lgamma max_abs_error x>0: %.3g  x<0: %.3g\n", lg_maxabs, lg_maxabs_neg);
    for (auto& s : st) printf("%-6s cases=%ld differ_from_libm=%ld (%.4f%%) max_ulp=%ld\n", s.name, s.cnt, s.bad, 100.0 * s.bad / s.cnt, s.maxu);
}
