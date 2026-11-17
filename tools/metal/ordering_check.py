import numpy as np, time
from gpuwm.metal.runtime import Module, DeviceMemory, synchronize, rt
class A(np.ndarray): pass
def dev(n, dtype):
    m = DeviceMemory(n*np.dtype(dtype).itemsize); a = np.frombuffer(m, dtype=dtype, count=n).view(A); a._gpuwm_mem = m; return a
M = Module(r'''extern "C" __global__ void add1(int* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = y[i] + 1; }
extern "C" __global__ void shift(const int* a, int* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = a[(i + 1) % n] + 1; }''', ("-std=c++17",))
for n in (1000, 2_000_000):
  for conc in (True, False):
    for barrier in (True, False):
      for batch in (1, 64, 256):
        r = rt(); r.concurrent = conc; r.barrier_between = barrier; r.batch = batch
        y = dev(n, np.int32); z = dev(n, np.int32); y[:] = 0; z[:] = 0; synchronize()
        k = r and M.get_function("add1"); s = M.get_function("shift")
        g = ((n+255)//256,)
        t0 = time.time()
        for _ in range(500):
            k(g, (256,), (y, n)); s(g, (256,), (y, z, n)); s(g, (256,), (z, y, n))
        synchronize(); dt = (time.time()-t0)/1500
        ok = (y.min() == y.max() == 1500)
        print(f"n={n:>8} concurrent={conc!s:5} barrier={barrier!s:5} batch={batch:3}: {'EXACT' if ok else 'WRONG'} min={y.min()} max={y.max()} {1e6*dt:.1f} us/dispatch")
