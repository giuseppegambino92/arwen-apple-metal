"""Dispatch ordering on the Metal runtime: 1500 dependent kernels (read-after-write
chains through two arrays) per configuration must leave every element at 1500.

Exercises in-command-buffer ordering (serial encoder + barriers) and
cross-command-buffer ordering (the GPU event chain) at several batch sizes,
eager commit on and off.  Exit status 1 on any wrong result."""
import os
import sys
import time

os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401,E402
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402
from gpuwm.metal.runtime import Module, synchronize, rt  # noqa: E402

M = Module(r'''extern "C" __global__ void add1(int* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = y[i] + 1; }
extern "C" __global__ void shift(const int* a, int* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = a[(i + 1) % n] + 1; }''', ("-std=c++17",))
bad = 0
for n in (1000, 2_000_000):
    for barrier in (True, False):
        for batch in (1, 4, 64, 256):
            for eager in (0, 2, 16):
                r = rt()
                r.barrier_between, r.batch, r.eager = barrier, batch, eager
                y, z = cp.zeros(n, cp.int32), cp.zeros(n, cp.int32)
                synchronize()
                k, s = M.get_function("add1"), M.get_function("shift")
                g = ((n + 255) // 256,)
                t0 = time.time()
                for _ in range(500):
                    k(g, (256,), (y, np.int32(n)))
                    s(g, (256,), (y, z, np.int32(n)))
                    s(g, (256,), (z, y, np.int32(n)))
                synchronize()
                dt = (time.time() - t0) / 1500
                h = cp.asnumpy(y)
                ok = int(h.min()) == int(h.max()) == 1500
                bad += not ok
                if not ok or (barrier and eager == 16):
                    print(f"n={n:>8} barrier={barrier!s:5} batch={batch:3} eager={eager:2}: "
                          f"{'EXACT' if ok else 'WRONG'} min={h.min()} max={h.max()} {1e6 * dt:.1f} us/dispatch")
r.barrier_between, r.batch, r.eager = True, 256, 16
print(f"{bad} wrong configurations")
sys.exit(1 if bad else 0)
