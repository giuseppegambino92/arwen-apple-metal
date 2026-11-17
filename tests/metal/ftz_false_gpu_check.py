"""FP32 gradual underflow on the GPU: a --ftz=false module's + - * / sqrt and
comparisons vs NumPy float32 (IEEE, subnormals kept), subnormal-heavy inputs.
Declared tolerance: 0 differing words (NaN payloads compared as NaN)."""
import os, sys
os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401
import numpy as np
import cupy as cp
from gpuwm.metal import runtime

SRC = r'''
extern "C" __global__ void ops(const float* x, const float* y, float* o, unsigned int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i], b = y[i];
    o[5*i+0] = a + b; o[5*i+1] = a - b; o[5*i+2] = a * b; o[5*i+3] = a / b; o[5*i+4] = sqrtf(fabsf(a));
    c[i] = (a < b) | ((a <= b) << 1) | ((a > b) << 2) | ((a >= b) << 3) | ((a == b) << 4) | ((a != b) << 5);
}'''
rng = np.random.default_rng(7)
n = 1 << 22
def gen():
    k = rng.integers(0, 7, n)
    sub = (rng.integers(0, 1 << 23, n, dtype=np.uint32) | (rng.integers(0, 2, n, dtype=np.uint32) << 31)).view(np.float32)
    small = (rng.integers(1 << 23, 3 << 23, n, dtype=np.uint32)).view(np.float32) * np.where(rng.integers(0, 2, n), 1, -1).astype(np.float32)
    normal = rng.standard_normal(n).astype(np.float32)
    zero = np.where(rng.integers(0, 2, n), np.float32(0.0), np.float32(-0.0))
    # normals around the 2^-101 add/sub threshold, and huge/infinite values (subnormal x large)
    band = ((rng.integers(1, 33, n, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, n, dtype=np.uint32)
            | (rng.integers(0, 2, n, dtype=np.uint32) << 31)).view(np.float32)
    huge = ((rng.integers(200, 256, n, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, n, dtype=np.uint32)).view(np.float32)
    return np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [sub, small, normal, zero, band, huge], (rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)).view(np.float32))
x, y = gen(), gen()
m = runtime.Module(SRC, ("-std=c++17", "--ftz=false"))
k = m.get_function("ops")
o, c = cp.empty(5 * n, cp.float32), cp.empty(n, cp.uint32)
k(((n + 255) // 256,), (256,), (cp.asarray(x), cp.asarray(y), o, c, n))
g, gc = cp.asnumpy(o).reshape(n, 5), cp.asnumpy(c)
with np.errstate(all="ignore"):
    ref = np.stack([x + y, x - y, x * y, x / y, np.sqrt(np.abs(x))], axis=1)
    rc = ((x < y) | ((x <= y) << 1) | ((x > y) << 2) | ((x >= y) << 3) | ((x == y) << 4) | ((x != y) << 5)).astype(np.uint32)
bad = 0
for j, name in enumerate(("add", "sub", "mul", "div", "sqrt")):
    nan = np.isnan(ref[:, j]) & np.isnan(g[:, j])
    d = int(np.count_nonzero((g[:, j].view(np.uint32) != ref[:, j].view(np.uint32)) & ~nan))
    bad += d
    print(f"{name:5s} differing words {d} of {n}")
dc = int(np.count_nonzero(gc != rc)); bad += dc
print(f"cmp   differing words {dc} of {n}")
sys.exit(1 if bad else 0)
