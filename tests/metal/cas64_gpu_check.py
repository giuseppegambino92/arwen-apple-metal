"""64-bit atomicCAS on the GPU (emulated under a lock word: Apple GPUs have no
64-bit compare-and-swap).  Every thread of many blocks adds to 64-bit counters
through CAS retry loops -- one counter in device memory shared by the whole
grid, one in threadgroup memory per block -- with increments that carry across
the 32-bit halves; and the lanes of each block race to publish one pointer-sized
word with CAS(slot, 0, v), the Grell-Freitas spill pattern.
Declared tolerance: exact counts, exactly one winner per block, every loser
reads the winner's word, and the kernel finishes (no lane waits on its own
SIMD group)."""
import os, sys
os.environ.setdefault("GPUWM_BACKEND", "metal")
import gpuwm  # noqa: F401
import numpy as np
import cupy as cp
from gpuwm.metal import runtime

SRC = r'''
typedef unsigned long long u64;
extern "C" __global__ void cas64(u64* grid_counter, u64* block_out, u64* winners, u64* seen, int rounds) {
    __shared__ u64 local;
    __shared__ u64 slot;
    if (threadIdx.x == 0) { local = 0; slot = 0; }
    __syncthreads();
    const u64 step = 0x00000000ffffffffull + threadIdx.x;      // carries into the high half
    for (int r = 0; r < rounds; r++) {
        u64 old = *(volatile u64*)grid_counter, assumed;
        do { assumed = old; old = atomicCAS(grid_counter, assumed, assumed + step); } while (old != assumed);
        u64 o2 = local, a2;
        do { a2 = o2; o2 = atomicCAS(&local, a2, a2 + step); } while (o2 != a2);
    }
    const u64 mine = 0x0123456700000000ull + ((u64)blockIdx.x << 12) + (u64)threadIdx.x * 16 + 16;
    u64 prev = atomicCAS(&slot, 0ull, mine);
    if (prev == 0ull) atomicAdd(&winners[blockIdx.x], 1ull);
    __syncthreads();
    seen[blockIdx.x * blockDim.x + threadIdx.x] = (prev == 0ull ? mine : prev) == slot ? 1ull : 0ull;
    if (threadIdx.x == 0) block_out[blockIdx.x] = local;
}'''
blocks, threads, rounds = 64, 128, 8
m = runtime.Module(SRC, ("-std=c++17",))
k = m.get_function("cas64")
g = cp.zeros(1, cp.uint64)
b = cp.zeros(blocks, cp.uint64)
w = cp.zeros(blocks, cp.uint64)
s = cp.zeros(blocks * threads, cp.uint64)
k((blocks,), (threads,), (g, b, w, s, np.int32(rounds)))
cp.cuda.runtime.deviceSynchronize()
per_block = sum(0x00000000ffffffff + t for t in range(threads)) * rounds
exp_grid = (per_block * blocks) % (1 << 64)
bad = 0
got = int(cp.asnumpy(g)[0])
print(f"grid counter {got:#x} expected {exp_grid:#x}"); bad += got != exp_grid
bo = cp.asnumpy(b)
d = int(np.count_nonzero(bo != np.uint64(per_block % (1 << 64))))
print(f"block counters differing {d} of {blocks}"); bad += d
wo = cp.asnumpy(w)
d = int(np.count_nonzero(wo != 1))
print(f"blocks without exactly one publisher {d} of {blocks}"); bad += d
so = cp.asnumpy(s)
d = int(np.count_nonzero(so != 1))
print(f"lanes that read another word than the published one {d} of {blocks * threads}"); bad += d
sys.exit(1 if bad else 0)
