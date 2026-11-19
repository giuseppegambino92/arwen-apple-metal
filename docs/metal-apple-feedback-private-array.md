# Apple GPU compiler: constant-index read of a private array returns 0

**Hardware / software:** Apple M3 Max (Apple9 GPU family), macOS 27.2, Xcode 27.2,
Metal toolchain 32023.922.

**Reproduction:** `swiftc -O tools/metal/apple_private_array_repro.swift -o repro && ./repro`
(plain Metal, no third-party code).  The kernel fills `float rho[N]` from a
buffer in a loop with a run-time bound, sums it in a second loop, then stores
`rho[0]`:

```metal
float rho[N];
for (int k = 0; k < nz; ++k) { rho[k] = air[k]; }
float s = 0.0f;
for (int k = 0; k < nz; ++k) { s += rho[k]; }
qnr[0] = s;
out[0] = rho[0];
```

**Expected:** `out[0] = air[0] = 1.2` for every N.
**Actual:** `1.2` for N = 32; `0.0` for N = 63, 64, 65, 128.  The same with
`MTLLibrary` built from source or from `xcrun metal` AIR at -O3, -Os and -O0,
and with or without `-fno-fast-math`, `-ffp-contract=off`,
`-fmetal-math-fp32-functions=precise`.  A read with a run-time index
(`rho[zero]`, `zero` a kernel argument equal to 0) returns 1.2, and so do the
other constant-index reads of the same array once one run-time-index access
exists.

**Workaround used by the ArWen Metal backend:** every integer-literal
subscript on a private array of 33 or more elements, on a local pointer, and on
a pointer/array parameter of a device function gets a run-time zero
(`gpuwm/metal/transpile.py`, `edit_private_const_index`).  Found through the
NSSL two-moment microphysics: its rain-sedimentation kernel exported zero
surface precipitation because `rho[0]` read 0 after the substep loop.
