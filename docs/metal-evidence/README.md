# Regression and comparison receipts

`bitwise_*.json` compare every word of a 1 h forecast's `wrfout` (one file per
domain) written by the Metal backend at one revision with the one written at
the revision before it (`tools/metal/wrfout_bitwise.py`).  `differing_words` is
0 in every file: 142 775 319 words per run (46 287 130 in d01, 96 488 189 in
d02).

* r12, r13, r14 — translator, runtime and array-layer changes of 2026-11-18,
  each against the run before it;
* r16 vs r15, r18 vs r17 — changes that must not move the forecast (select
  NaN semantics, initialiser lists; gradual underflow in template code, the
  correctly rounded libm of WRF-exact builds);
* r19b vs r19 — the same build run twice: the forecast is deterministic;
* r21 vs r20 — the first hour of the 6 h comparison run;
* r23 vs r20 — after Grell-Freitas support (64-bit CAS emulation, per-thread
  device frames): the operational forecast is unchanged.

r15, r17 and r19 changed numerics on purpose (correctly rounded integer
powers; the Apple compiler workaround and glibc `powf` for `cupy.power`;
glibc's `powf` algorithm for the GPU's own `powf`) and became the new
reference.  `r19_vs_cuda_*.json` compare the r19 forecast with the RTX 5090's
(`tools/metal/compare_wrfout.py`): RMSE, bias and correlation per field;
`r21_vs_cuda_*_06h.json` do the same after 6 h with the release build.
