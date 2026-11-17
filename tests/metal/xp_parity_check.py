"""Metal xp layer vs NumPy: every result must match exactly (integers, comparisons,
exact float ops) or within the stated ULP bound (transcendental float functions)."""
import sys
import numpy as np
import gpuwm.metal.xp as cp

rng = np.random.default_rng(0)
fails, checks = [], 0


def check(name, got, ref, ulp=0):
    global checks
    checks += 1
    got = cp.asnumpy(got) if cp.is_device(got) else np.asarray(got)
    ref = np.asarray(ref)
    ok = got.shape == ref.shape and got.dtype == ref.dtype
    if ok:
        if ref.dtype.kind == "f":
            g, r = got.astype(ref.dtype), ref
            both_nan = np.isnan(g) & np.isnan(r)
            if ulp == 0:
                ok = bool(np.all((g.view(f"u{ref.itemsize}") == r.view(f"u{ref.itemsize}")) | both_nan))
            else:
                ii = {4: np.int32, 8: np.int64}[ref.itemsize]
                d = np.abs(g.view(ii).astype(np.int64) - r.view(ii).astype(np.int64))
                ok = bool(np.all((d <= ulp) | both_nan | (g == r)))
        else:
            ok = np.array_equal(got, ref)
    if not ok:
        fails.append(name)
        print(f"FAIL {name}: got {got.dtype}{got.shape} {got.ravel()[:4]} ref {ref.dtype}{ref.shape} {ref.ravel()[:4]}")


def check_sum(name, got, ref, abs_sum, n):
    """Floating-point reductions: summation order differs from numpy's pairwise sum;
    the declared bound is the standard one, |err| <= n * eps * sum|x|."""
    global checks
    checks += 1
    got = cp.asnumpy(got)
    eps = np.finfo(ref.dtype).eps
    err = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    ok = got.shape == ref.shape and got.dtype == ref.dtype and bool(np.all(err <= n * eps * abs_sum + 1e-300))
    if not ok:
        fails.append(name)
        print(f"FAIL {name}: max err {err.max()} bound {(n * eps * abs_sum).min()}")


for dt in (np.float32, np.float64, np.int32, np.int64):
    a_h = (rng.standard_normal((7, 33, 65)) * 50).astype(dt)
    b_h = (rng.standard_normal((33, 65)) * 50).astype(dt) + (dt(1) if np.dtype(dt).kind != "f" else 0)
    b_h[b_h == 0] = 3
    a, b = cp.asarray(a_h), cp.asarray(b_h)
    n = np.dtype(dt).name
    check(f"add {n}", a + b, a_h + b_h)
    check(f"sub scalar {n}", a - 3, a_h - 3)
    check(f"mul view {n}", a[:, ::2, 1:] * b[::2, 1:], a_h[:, ::2, 1:] * b_h[::2, 1:])
    check(f"maximum {n}", cp.maximum(a, b), np.maximum(a_h, b_h))
    check(f"where {n}", cp.where(a > 0, a, b), np.where(a_h > 0, a_h, b_h))
    check(f"clip {n}", cp.clip(a, -10, 20), np.clip(a_h, -10, 20))
    if np.dtype(dt).kind == "f":
        check_sum(f"sum axis0 {n}", a.sum(axis=0), a_h.sum(axis=0), np.abs(a_h).sum(axis=0), 7)
    else:
        check(f"sum axis0 {n}", a.sum(axis=0), a_h.sum(axis=0))
    check(f"max all {n}", a.max(), a_h.max())
    check(f"min axis(1,2) {n}", a.min(axis=(1, 2)), a_h.min(axis=(1, 2)))
    check(f"argmax {n}", cp.argmax(a), np.argmax(a_h))
    check(f"any {n}", (a > 100).any(), (a_h > 100).any())
    check(f"floor_divide {n}", a // b, a_h // b_h)
    check(f"remainder {n}", a % b, a_h % b_h)
    check(f"astype f32 {n}", a.astype(np.float32), a_h.astype(np.float32))
    check(f"transpose copy {n}", a.transpose(2, 0, 1).copy(), a_h.transpose(2, 0, 1).copy())
    check(f"concatenate {n}", cp.concatenate([a, a[:2]], axis=0), np.concatenate([a_h, a_h[:2]], axis=0))
    check(f"roll {n}", cp.roll(a, 3, axis=2), np.roll(a_h, 3, axis=2))
    check(f"pad edge {n}", cp.pad(a, ((0, 0), (1, 2), (2, 1)), mode="edge"), np.pad(a_h, ((0, 0), (1, 2), (2, 1)), mode="edge"))
    if np.dtype(dt).kind == "f":
        check_sum(f"cumsum {n}", cp.cumsum(a, axis=1), np.cumsum(a_h, axis=1), np.cumsum(np.abs(a_h), axis=1), 33)
    else:
        check(f"cumsum {n}", cp.cumsum(a, axis=1), np.cumsum(a_h, axis=1))
    check(f"bool mask getitem {n}", a[a > 10], a_h[a_h > 10])
    check(f"int index getitem {n}", a[cp.asarray([3, 0, 6])], a_h[[3, 0, 6]])
    check(f"count_nonzero {n}", cp.count_nonzero(a > 0), np.count_nonzero(a_h > 0))
    c = a.copy(); c_h = a_h.copy()
    c[c > 0] = 0; c_h[c_h > 0] = 0
    check(f"mask setitem {n}", c, c_h)
    c[1:3, ::4] = b[::4]; c_h[1:3, ::4] = b_h[::4]
    check(f"slice setitem {n}", c, c_h)
    if np.dtype(dt).kind == "f":
        check(f"sqrt {n}", cp.sqrt(cp.abs(a)), np.sqrt(np.abs(a_h)))
        check(f"true_divide {n}", a / b, a_h / b_h)
        check(f"exp {n}", cp.exp(a / 50), np.exp(a_h / 50), ulp=4)
        check(f"log {n}", cp.log(cp.abs(a) + 1), np.log(np.abs(a_h) + 1), ulp=4)
        check(f"power {n}", cp.power(cp.abs(a) / 10, 1.7), np.power(np.abs(a_h) / 10, dt(1.7)), ulp=4)
        check_sum(f"mean {n}", a.mean(axis=2), a_h.mean(axis=2), np.abs(a_h).mean(axis=2), 65)
        check(f"isnan {n}", cp.isnan(a / a), np.isnan(a_h / a_h))

x = cp.arange(10, dtype=np.float32); check("arange f32", x, np.arange(10, dtype=np.float32))
check("arange int", cp.arange(3, 20, 4), np.arange(3, 20, 4))
check("zeros", cp.zeros((3, 4), np.int32), np.zeros((3, 4), np.int32))
check("full", cp.full((2, 3), 7.5, np.float32), np.full((2, 3), 7.5, np.float32))
ek = cp.ElementwiseKernel("float32 a, float32 b, float32 c", "float32 out", "out = __fadd_rn(__fadd_rn(a, b), c);", "compose")
ah, bh, chh = (rng.standard_normal((40, 50)).astype(np.float32) for _ in range(3))
check("ElementwiseKernel", ek(cp.asarray(ah), cp.asarray(bh), cp.asarray(chh)), (ah + bh) + chh)
ek2 = cp.ElementwiseKernel("raw T x, int32 n", "T y", "y = x[(i + 1) % n] * (T)2;", "rawk")
check("ElementwiseKernel raw+T", ek2(cp.asarray(ah.ravel()), ah.size, cp.empty(ah.size, np.float32)), np.roll(ah.ravel(), -1) * 2)
rk = cp.ReductionKernel("T x", "int32 y", "isfinite(x) ? 0 : 1", "a | b", "y = a", "0", "nonfinite")
zz = ah.copy(); zz[3, 7] = np.inf
check("ReductionKernel", rk(cp.asarray(zz), axis=1), (~np.isfinite(zz)).any(axis=1).astype(np.int32))
# many operands with different shapes (regression: the layout block once held 4 operands)
nz_, ny_, nx_ = 5, 7, 9
ins = [rng.uniform(1, 2, s).astype(np.float32) for s in [(nz_, ny_, nx_)] * 4 + [(ny_, nx_)] * 3 + [(nz_, 1, 1)] * 3]
names = [f"x{i}" for i in range(len(ins))]
ekm = cp.ElementwiseKernel(", ".join(f"float32 {n}" for n in names), "float32 o1, float32 o2",
                           "o1 = " + " + ".join(names) + "; o2 = x0 * x4 * x7;", "many_ops")
o1, o2 = ekm(*[cp.asarray(a) for a in ins])
ref1 = ins[0]
for a in ins[1:]:
    ref1 = ref1 + a
check("ElementwiseKernel 12 operands broadcast o1", o1, ref1)
check("ElementwiseKernel 12 operands broadcast o2", o2, ins[0] * ins[4] * ins[7])
print(f"{checks} checks, {len(fails)} failures")
sys.exit(1 if fails else 0)
