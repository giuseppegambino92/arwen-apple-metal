// Exact binary32 with gradual underflow (sf32_* in arwen_sf64.h) vs the host FPU.
#include "../../gpuwm/metal/include/arwen_sf64.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
static std::mt19937_64 rng;
static uint32_t U(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float F(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static float gen() {
    switch (rng() % 6) {
    case 0: return F((uint32_t)(rng() & 0x807FFFFF));                                    // subnormal
    case 1: return F((uint32_t)(rng() & 0x80FFFFFF) | (1u << 23));                       // smallest normals
    case 2: { uint32_t e = 1 + rng() % 40; return F((uint32_t)(rng() & 0x807FFFFF) | (e << 23)); }
    case 3: { uint32_t e = 100 + rng() % 54; return F((uint32_t)(rng() & 0x807FFFFF) | (e << 23)); }
    case 4: { static const uint32_t s[] = {0, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 1, 0x00800000u, 0x7F7FFFFFu}; return F(s[rng() % 8] ^ ((rng() & 1) << 31)); }
    default: return F((uint32_t)rng());
    }
}
static bool same(float g, float r) { return std::isnan(r) ? std::isnan(g) : U(g) == U(r); }
int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000; rng.seed(argc > 2 ? atol(argv[2]) : 3);
    long bad[6] = {0}; const char* nm[6] = {"add", "sub", "mul", "div", "sqrt", "fma"};
    for (long i = 0; i < n; ++i) {
        float a = gen(), b = gen(), c = gen();
        bad[0] += !same(sf32_add(a, b), a + b);
        bad[1] += !same(sf32_sub(a, b), a - b);
        bad[2] += !same(sf32_mul(a, b), a * b);
        bad[3] += !same(sf32_div(a, b), a / b);
        bad[4] += !same(sf32_sqrt(a), std::sqrt(a));
        bad[5] += !same(sf32_fma(a, b, c), std::fma(a, b, c));
    }
    long t = 0; for (int k = 0; k < 6; ++k) { printf("%-5s %ld\n", nm[k], bad[k]); t += bad[k]; }
    printf("cases_per_op %ld total_mismatches %ld\n", n, t); return t != 0;
}
