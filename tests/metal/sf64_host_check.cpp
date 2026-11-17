// Bit-exact check of arwen_sf64.h against the host FPU (IEEE binary64, no FTZ).
// Usage: sf64_host_check N_PER_OP SEED   -> prints mismatch counts per operation, exit 1 on any mismatch.
#include "../../gpuwm/metal/include/arwen_sf64.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
static inline uint64_t B(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static inline double D(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static std::mt19937_64 rng;
static uint64_t special() {
    static const uint64_t s[] = {0, 0x8000000000000000ULL, 0x7FF0000000000000ULL, 0xFFF0000000000000ULL, 0x7FF8000000000000ULL,
        1, 0x000FFFFFFFFFFFFFULL, 0x0010000000000000ULL, 0x7FEFFFFFFFFFFFFFULL, 0x3FF0000000000000ULL, 0xBFF0000000000000ULL,
        0x3FE0000000000000ULL, 0x4330000000000000ULL, 0x3CA0000000000000ULL, 0x0008000000000000ULL};
    return s[rng() % (sizeof(s) / 8)] ^ ((rng() & 1) ? 0x8000000000000000ULL : 0);
}
static uint64_t gen() {
    switch (rng() % 8) {
    case 0: return special();
    case 1: return (rng() & 0x800FFFFFFFFFFFFFULL);                                 // subnormal
    case 2: { uint64_t e = 0x3FF + (int)(rng() % 120) - 60; return (rng() & 0x800FFFFFFFFFFFFFULL) | (e << 52); }
    case 3: { uint64_t e = rng() % 0x7FF; return (rng() & 0x800FFFFFFFFFFFFFULL) | (e << 52); }
    case 4: { uint64_t e = (rng() % 2) ? 1 + rng() % 60 : 0x7FE - rng() % 60; return (rng() & 0x800FFFFFFFFFFFFFULL) | (e << 52); }
    case 5: { uint64_t e = 0x3FF + (int)(rng() % 10) - 5; return (rng() & 0x8000000000000000ULL) | (e << 52) | (rng() & 0xFFFFF); } // short mantissas -> ties
    default: return rng();
    }
}
static bool same(uint64_t got, double ref) {
    uint64_t r = B(ref);
    if (std::isnan(ref)) return SF_ISNAN(got);
    return got == r;
}
int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    rng.seed(argc > 2 ? atol(argv[2]) : 1);
    long bad[16] = {0};
    const char* names[16] = {"add", "sub", "mul", "div", "sqrt", "fma", "fma_near", "to_f32", "from_f32", "from_i64", "to_i64", "rint", "floor", "ceil", "trunc", "round"};
    for (long i = 0; i < n; ++i) {
        uint64_t a = gen(), b = gen(), c = gen();
        double x = D(a), y = D(b), z = D(c);
        bad[0] += !same(sf_add(a, b), x + y);
        bad[1] += !same(sf_sub(a, b), x - y);
        bad[2] += !same(sf_mul(a, b), x * y);
        bad[3] += !same(sf_div(a, b), x / y);
        bad[4] += !same(sf_sqrt(a), std::sqrt(x));
        bad[5] += !same(sf_fma(a, b, c), std::fma(x, y, z));
        { double p = x * y; uint64_t cc = B(-p) ^ (rng() & 0xFF); bad[6] += !same(sf_fma(a, b, cc), std::fma(x, y, D(cc))); } // cancellation
        { float f = sf_to_f32(a), r = (float)x; uint32_t fu, ru; memcpy(&fu, &f, 4); memcpy(&ru, &r, 4);
          bad[7] += std::isnan(r) ? !std::isnan(f) : (fu != ru); }
        { uint32_t u = (uint32_t)rng(); float f; memcpy(&f, &u, 4); bad[8] += !same(sf_from_f32(f), (double)f); }
        { int64_t v = (int64_t)rng() >> (rng() % 64); bad[9] += !same(sf_from_i64(v), (double)v); }
        { if (!std::isnan(x) && std::fabs(x) < 9.2e18) bad[10] += sf_to_i64_rz(a) != (int64_t)x; }
        bad[11] += !same(sf_rint(a), std::rint(x));
        bad[12] += !same(sf_floor(a), std::floor(x));
        bad[13] += !same(sf_ceil(a), std::ceil(x));
        bad[14] += !same(sf_trunc(a), std::trunc(x));
        bad[15] += !same(sf_round(a), std::round(x));
    }
    long total = 0;
    for (int k = 0; k < 16; ++k) { printf("%-9s %ld\n", names[k], bad[k]); total += bad[k]; }
    printf("cases_per_op %ld total_mismatches %ld\n", n, total);
    return total != 0;
}
