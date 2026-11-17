"""binary64 on the GPU: a kernel's double + - * / sqrt fma vs NumPy float64
(the host FPU, IEEE round-to-nearest-even, no flush), on inputs rich in
specials, subnormals, ties, perfect squares and exact quotients.
Declared tolerance: 0 differing words (NaN compared as NaN)."""
import math
import os
import sys
os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401,E402
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402
from gpuwm.metal import runtime  # noqa: E402

SRC = r'''
extern "C" __global__ void ops(const double* x, const double* y, const double* z, double* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = x[i], b = y[i], c = z[i];
    o[6*i+0] = a + b; o[6*i+1] = a - b; o[6*i+2] = a * b; o[6*i+3] = a / b;
    o[6*i+4] = sqrt(a); o[6*i+5] = fma(a, b, c);
}'''

rng = np.random.default_rng(11)
n = 1 << 22


def gen():
    u = rng.integers(0, 2**63, n, dtype=np.uint64) | (rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63))
    k = rng.integers(0, 6, n)
    frac = u & np.uint64(0x800FFFFFFFFFFFFF)
    near1 = frac | (np.uint64(0x3FF) + rng.integers(0, 8, n, dtype=np.uint64) - np.uint64(4)) << np.uint64(52)
    short = (u & np.uint64(0x80000000000FFFFF)) | (np.uint64(0x3FD) << np.uint64(52))      # ties
    special = np.array([0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 1,
                        0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF, 0x3FF0000000000000],
                       dtype=np.uint64)[rng.integers(0, 10, n)]
    sub = u & np.uint64(0x800FFFFFFFFFFFFF)
    return np.select([k == 0, k == 1, k == 2, k == 3, k == 4], [u, near1, short, special, sub], frac | (
        np.uint64(0x3FF) << np.uint64(52))).view(np.float64)


x, y, z = gen(), gen(), gen()
# perfect squares and exact products in a quarter of the lanes
m = rng.integers(1, 2**26, n).astype(np.float64)
sq = rng.integers(0, 4, n) == 0
x = np.where(sq, m * m, x)
y = np.where(sq, np.where(rng.integers(0, 2, n) == 0, m, y), y)
k = cp.RawModule(code=SRC, options=("-std=c++17",)).get_function("ops")
o = cp.empty(6 * n, cp.float64)
k(((n + 255) // 256,), (256,), (cp.asarray(x), cp.asarray(y), cp.asarray(z), o, np.int32(n)))
g = cp.asnumpy(o).reshape(n, 6)


def fma1(a, b, c):
    """math.fma (Python >= 3.13, correctly rounded) with IEEE results instead of exceptions."""
    try:
        return math.fma(a, b, c)
    except ValueError:                 # invalid: inf * 0, inf - inf
        return math.nan
    except OverflowError:              # the exact value overflows: +-inf, its sign is that of a*b + c
        return math.copysign(math.inf, a * b + c if math.isfinite(a * b) else a * b)


with np.errstate(all="ignore"):
    fma = np.fromiter(map(fma1, x.tolist(), y.tolist(), z.tolist()), np.float64, n)
    ref = np.stack([x + y, x - y, x * y, x / y, np.sqrt(x), fma], axis=1)
bad = 0
for j, name in enumerate(("add", "sub", "mul", "div", "sqrt", "fma")):
    nan = np.isnan(ref[:, j]) & np.isnan(g[:, j])
    d = int(np.count_nonzero((g[:, j].view(np.uint64) != ref[:, j].view(np.uint64)) & ~nan))
    bad += d
    print(f"{name:5s} differing words {d} of {n}")
sys.exit(1 if bad else 0)
