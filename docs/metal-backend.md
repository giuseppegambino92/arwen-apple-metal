# Metal backend (Apple Silicon) — design notes and measurements

Status: **in progress**. Milestones 1–2 (environment, CUDA→MSL translator)
done; Milestone 3 (runtime, `xp` layer, `--backend metal`) working end to end
up to the first model step; Milestone 4 (cu6 Tiedtke) bit-exact on the M3 Max; kernel parity and the real run (Milestones 4–7)
not yet — the first physics step currently produces non-finite RRTMG LW and
MYNN tendencies.

The CUDA sources stay canonical. `gpuwm/metal/` translates, at build/JIT
time, exactly the source string gpuwm hands to NVRTC; nothing on the Metal
side is maintained by hand except the compatibility headers in
`gpuwm/metal/include/`.

## What a real run compiles (measured)

`tools/probe/sitecustomize.py` (arwen-metal repo) plus `GPUWM_AOT_RECORD`
recorded every compile request of a 1-hour `gpuwm go` of the target domain
(Sicily2KM_ECM, d01 176×152 6 km + d02 238×232 2 km, 45 levels, mp=8, cu=6 on
d01, ra=4 legacy, bl=5/sfclay=5, sf=2) on the RTX 5090 reference machine:

| | modules | kernels |
|---|---:|---:|
| hand-written CUDA (RawModule + direct NVRTC) | 125 | 1 058 (clang AST count) |
| CuPy-generated (ufuncs, reductions, ElementwiseKernel) | 108 | 108 |
| modules using `double` | 67 | |

The 125 modules come from 54 source files: dynamics (acoustic, advection,
dycore, diff6, smag2d, diff_opt1, pd_advection, lbc_*, spec_bdy, nest,
vert_interp), physics (thompson, mynn_pbl/surface/scalar_mix, rrtmg_lw_chain,
rrtmg_sw, rrtmg_legacy_*, rrtmg_mcica_wrf, noah, tiedtke, terrain_drag),
diagnostics (refl, hail/HAILCAST, nwp_hail, uh_diag, nwp_windows, health),
and hydrology (troute_mc, nextgen_reservoirs, runoff, noahowp coupling) —
the default route runs more than the scheme list in the brief.

## Apple GPU floating point (measured on M3 Max)

2^20 random float32 operands per operation, against IEEE on the host:

| compile flags | `a*b`, `a/b`, `sqrt` | `a*b+c` | subnormal results |
|---|---|---|---|
| `-ffast-math` (Metal default) | `sqrt` wrong in 272 717 cases | not contracted | flushed |
| `-fno-fast-math` | correct | **contracted to FMA** in 277 667 cases | flushed |
| `-fno-fast-math -ffp-contract=off` | correct | correct | flushed |

* Only the 1 000 subnormal test operands differ in the last row: **the GPU
  flushes FP32 subnormals in hardware**; no compiler flag changes it.
* The translator therefore compiles every module with
  `-fno-fast-math -ffp-contract=off -fmetal-math-fp32-functions=precise`.
  CUDA's `__f*_rn` intrinsics are then single IEEE operations. Plain
  expressions round like the unfused Fortran/CPU oracles; CUDA with
  `--fmad=true` (NVRTC default) may have fused some of them, so bit parity
  with the CUDA run is not expected outside `_rn`-written code.
* CUDA modules compiled with `--ftz=false` (MYNN, RRTMG, diffusion) rely on
  subnormals in a few places; on Metal those values become zero.

## FP64

ArWen's device `double` code is written for bit-exact IEEE binary64:
in the 125 modules, `__dmul_rn` ×1 771, `__dadd_rn` ×1 147,
`__longlong_as_double` ×553, `__double_as_longlong` ×153, `__umul64hi` ×56
(the correctly rounded math of `f64cr.cuh` and the glibc transcriptions work
on the bit patterns). A float-float (`df64`) representation cannot honour
those bit manipulations, so the reference mode emulates binary64 exactly:

* `include/arwen_sf64.h` — add/sub/mul/div/sqrt/fma, conversions,
  rint/floor/ceil/trunc/round on 64-bit integers, round-to-nearest-even,
  gradual underflow. **Checked bit for bit against the host FPU:
  800 M cases × 16 operations, 0 mismatches**
  (`tests/metal/sf64_host_check.cpp`, special-value-heavy generators,
  catastrophic-cancellation fma cases).
* `include/arwen_f64_math.h` — fdlibm 5.3 exp/log/log10/pow/atan/atan2/sin/
  cos, plus hypot, cbrt, lgamma. Against macOS libm, 2–4 M cases each
  (`tests/metal/f64_math_host_check.cpp`): max 1 ulp (log10 2 ulp);
  lgamma absolute error 8.5e-14 for x > 0 (the only domain used, by
  HAILCAST's incomplete gamma) and 4e-11 near the poles for x < 0.
  Trig argument reduction is exact for |x| < 1.6e6.
* The translator spells `double` as `__f64` and makes **every** conversion
  involving double explicit (`__arwen_cvt<T>`), implicit or written, using the
  AST; double literals are emitted as their exact bit patterns.
* `df64` and `fp32` modes: selectable per module (`f64_mode=`), **not yet
  implemented** — selecting them is a compile-time `#error`, never a silent
  substitution.

## Translator (`gpuwm/metal/transpile.py`)

Pipeline: clang preprocess with the module's NVRTC `-D/-U/-I` options and
NVRTC's predefined macros (`__CUDA_ARCH__=1200`, as on the reference GPU) →
libclang parse in CUDA device mode against `cuda_parse.h` (the authoritative
list of supported CUDA builtins: an unknown one is a parse error) → edits from
the AST → MSL.

| CUDA | MSL |
|---|---|
| `__global__` params | one `constant` argument struct; pointers are 64-bit GPU addresses (views are address + offset, as in CUDA); pointer-to-pointer params travel as `ulong` (Metal forbids them in buffers) |
| `threadIdx`… | per-thread context built in the kernel prologue, passed (first parameter) to every device function that needs it, transitively |
| generic pointers | device functions become templates over pointer parameters (deduction monomorphises per address space); `T&` → pointer; local pointers `auto`/`decltype`; pointer members `device`, or `thread` when the struct points into its own array member |
| pointer re-pointed between global and shared memory at run time | `__arwen_gptr<T>` (address-space tag) |
| `__shared__` in kernels / in device functions / `extern` | `threadgroup` / a static block in each kernel that reaches the function / `[[threadgroup(0)]]` |
| mutable `__device__` globals | per-module globals buffer (initialiser kernel generated) |
| `__constant__` / `const __device__` tables | MSL `constant` data |
| inline PTX | only the 7 patterns present in ArWen (`add/sub/mul/div.rn.f32`, `setp.gt`+`selp`, compiler barrier); anything else is a `TranslationError` |
| `goto` (HAILCAST's translated Fortran) | forward gotos only: `break` when the label follows the loop, else a skip flag; others are a `TranslationError` |
| inheritance | flattened (base members copied, conversion to the base kept for the CUDA parse) |
| atomics | Metal atomics; float add in threadgroup memory and 64-bit OR/AND/XOR built from 32-bit operations |
| warp shuffles/ballot | `simd_shuffle*`/`simd_ballot` (SIMD-group = 32 lanes, threadgroup-linear like warps), CUDA segment semantics for `width` < 32 |

### Milestone 2 gate (measured)

`tools/metal/compile_corpus.py CORPUS` on the 125 recorded modules,
f64 mode `sf64`, `xcrun metal -O3` (Metal toolchain 32023.922, Xcode 27.2):

| MSL standard | modules | kernels | wall (10 processes) |
|---|---|---|---|
| `metal4.0` (default) | 125/125 | 1 058/1 058 | 84 s |
| `metal3.2` | 125/125 | 1 058/1 058 | 108 s |

The kernel set of every module equals the `__global__` set of its clang AST.
Compiling is necessary, not sufficient: numerical parity is Milestones 4–5.

## Runtime (`gpuwm/metal/native/arwen_metal_rt.mm`, `gpuwm/metal/runtime.py`)

Objective-C++ (nanobind) extension built on first use with `xcrun clang++`
(no CMake), cached by source hash. Shared-storage MTLBuffers (CPU and GPU
address the same bytes), one MTLResidencySet on the queue, kernels launched
with their argument struct via `setBytes`, metallib cache keyed by SHA-256 of
(source, options, FP64 mode, translator revision, headers, toolchain).

**Ordering, measured** (`tools/metal/ordering_check.py`: 1 500 dependent
dispatches, 1 000 and 2 000 000 elements). Resources are untracked (they are
reached through GPU addresses), so Metal sees no dependencies:

* without ordering between command buffers, 200–340 of 2 000 dependent
  increments were lost, even with one dispatch per command buffer — the GPU
  overlaps consecutive command buffers. Fixed with an MTLSharedEvent chain
  (each command buffer waits for the previous one's signal);
* a *concurrent* encoder without barriers loses updates (as expected); serial,
  serial + barrier, and concurrent + barrier are exact. Default: serial +
  buffer barrier.

Cost: 6.5–9 µs per dispatch (Python packing + encoding) with 256 dispatches
per command buffer; ~300–550 µs with one dispatch per command buffer.

## Array layer and CuPy shim (`gpuwm/metal/xp`, `gpuwm/metal/shim.py`)

`--backend metal` (or `GPUWM_BACKEND=metal`) installs `gpuwm.metal.xp` as
`cupy` before any gpuwm module imports CuPy (hook at the top of
`gpuwm/__init__.py`), registers the `metal` backend, forces `GPUWM_AOT=off`
(CUDA kernel packs cannot run here) and makes `gpu_smi.py` answer the
supervisor's `nvidia-smi` queries from measured Metal/macOS data (working set,
`vm_stat` available memory, IOAccelerator utilisation, Metal client processes
as `C+G` with unmeasured per-process memory).

* `ndarray` is a numpy.ndarray subclass over an MTLBuffer: views/slicing are
  numpy's (no copies); ufuncs (NumPy 2 type resolution), reductions,
  `where/clip/concatenate/stack/roll/pad/take/nonzero/...`, ElementwiseKernel,
  ReductionKernel, RawModule/RawKernel, `cuda.function.Module` + NVRTC tokens
  are GPU kernels generated as CUDA and run through the same translator.
  CuPy-generated kernels compile with CuPy's `-ftz=true`.
* A NumPy function without a GPU implementation raises NotImplementedError
  naming it. There is no host fallback.
* NumPy hands out views of an ndarray subclass without asking it, so
  `np.asarray/np.array/...` of a device array would read GPU memory with work
  still queued (CuPy raises instead): those entry points wait for the GPU.
  This was the cause of a deterministic NaN in the initial pressure field
  when two dispatches shared a command buffer.
* `tests/metal/xp_parity_check.py`: 117 checks against NumPy (4 dtypes,
  broadcasting, strided views, reductions, scans, masks, gathers, custom
  kernels): 0 failures — bit-exact, or within |err| <= n·eps·Σ|x| for
  floating-point reductions whose summation order differs from NumPy's.

## FP32 subnormals (CUDA `--ftz=false` modules)

ArWen's own device preflights caught it: the legacy RRTMG LW kernel returned
0 where the Fortran keeps 9.9999461e-41. Modules CUDA compiles without FTZ
(MYNN, RRTMG, smag2d, terrain drag, diffusion, hydrology — 20 source files)
now route every float `+ - * /`, compound assignment, comparison, negation,
`_rn` intrinsic and math function through helpers that take the hardware
result unless an operand is subnormal or the result is below 2^-125 and not
an exact zero; then the correctly rounded IEEE result is computed in software
(binary64 rounded to odd, then once to binary32). Checked on the host:
20 M cases × 6 operations, subnormal-heavy, 0 mismatches
(`tests/metal/sf32_host_check.cpp`). The RRTMG LW/SW and legacy-prep device
preflights now pass on the M3 Max.

Hardware support was searched for: Apple's Metal feature tables list "FP32
denormalized numbers" for the Apple11 family only (the M3 Max is Apple9), MSL
allows flushing with fast math off, there is no compile option or capability
to turn it off, and the Linux AGX drivers (Mesa/Asahi forks) emulate denormal
preservation in software too.  The emulation is therefore permanent on this
GPU; its fast path tests the *result* only: the hardware reads a subnormal
operand as zero, so a subnormal operand of `*` or `/` always produces a result
in exponent band 0/1/255, and one of `+`/`-` changes the IEEE result only below
2^-101.  Only inside those bands do the exact conditions run.  GPU check
(`tests/metal/ftz_false_gpu_check.py`): `+ - * / sqrt` and the six comparisons
on 4.2 M subnormal-heavy cases each, 0 differing words vs NumPy.  Cost on a
dependent FP32 chain: 1.8× the flushing hardware on normal data (was 2.4×);
3–6× on data that is mostly exact zeros.

`a > b ? a : b` on floats is translated to `fmaxf` (and the three sibling
forms): NVRTC emits setp/selp, but CUDA's results follow ptxas's FMNMX
selection (max(+0, −0) = +0), which ArWen's legacy prep preflight checks.

## Compile cost (measured, `tools/metal/pipeline_timing.py`)

Soft-float routines are out of line; inlined into every operation they made
one MYNN pipeline compile run for more than 6 minutes. Now: all 54 source files
translate + compile to metallib + build their pipelines; worst cases thompson
(55 s + 22 s for 105 pipelines), smag2d (44 s + 14 s), tiedtke (41 s + 4 s).
Metallibs are cached on disk; pipelines (~3 min in total) are rebuilt by each
process — an MTLBinaryArchive cache is pending (Milestone 7).

## macOS portability fixes outside gpuwm/metal

* `gpuwm/bridges.py`: Rust bridges are Mach-O on macOS (the probe only knew
  ELF/PE).
* Rust tools built with `cargo build --release --locked --offline -p rw-netcdf
  -p netcdf-writer -p static-fields -p rw-fetch` (crates using cudarc are not
  built; the renderer is therefore unavailable and runs draw no pictures).

## Milestone 4: cu6 (original Tiedtke), bit for bit

`tools/metal/cu6_parity.py` replays the inputs of `tools/tiedtke_wrf461_oracle`
(generated with the WRF 4.6.1 Fortran on a Linux CPU host, glibc 2.41) through
the production loader `gpuwm.core.tiedtke.column_step` on Metal and compares
every output word with `check.compare_words`, against the Fortran and against
the same CUDA kernel run on the RTX 5090. 4 vertical grids × 3 configurations
× 72 columns, launched singly and as rows of 12 and 8 columns; all six
tendencies, RAINCV, PRATEC, CU_ACT_FLAG:

| mode | vs Fortran | vs CUDA (RTX 5090) |
|---|---|---|
| default | 0 differing words | 0 |
| `GPUWM_WRF_EXACT=1`, `GPUWM_WRF_GLIBC=2.41` | 0 | 0 |

(The exact mode's default referee is glibc 2.39, whose `tanf` is not
correctly rounded; against a 2.41 oracle that gives the documented residual in
deep convective columns, 6 342 words here — it is the referee choice, not the
port.) `GPUWM_WRF_EXACT` reaches Metal compiles: every compile asks
`cupy.cuda.compiler._NVRTCProgram(src).compile(options)`, which gpuwm's
option hooks wrap exactly as on CUDA.

## First hour on Metal vs the RTX 5090 run (measured)

`gpuwm go Sicily2KM_ECM --backend metal --run-seconds 3600`: SUCCESS, 309 steps
(d01 80, d02 229 — the same adaptive step sequence as the CUDA run, which
ends the hour at dt 24 s / 12 s too), 436 s wall on the M3 Max (CUDA: ~10.6 s
per simulated hour). `tools/metal/compare_wrfout.py` at 01:00, Metal − CUDA:

| field | d01 bias | d01 RMSE | d01 corr | d02 bias | d02 RMSE | d02 corr |
|---|---|---|---|---|---|---|
| T2 (K) | −3.0e-5 | 0.039 | 0.99995 | +9.7e-5 | 0.054 | 0.99985 |
| 10 m wind speed (m/s) | −2.0e-6 | 0.015 | 0.99998 | −3.8e-5 | 0.015 | 0.99997 |
| PSFC (Pa) | −0.003 | 0.30 | 1.0000 | −0.006 | 0.075 | 1.0000 |
| rain (mm) | +3.5e-7 | 7.6e-5 | 0.9991 | +1.8e-7 | 2.4e-5 | 0.99997 |

At 00:00 every surface field and (on d02) every prognostic is word-identical;
d01's initial T/U/V/PH/QVAPOR/MU differ in rounding (T max 6e-5 K) and the
diagnostic P by up to 0.086 Pa — under investigation. The hour-1 differences
are the expected growth of rounding-level differences (FMA contraction, libm)
through moist convection: no bias, correlations ≥ 0.9985, isolated maxima
(T2 1.4–1.7 K).

## Verified fast paths for the glibc float libm copies (`gpuwm/metal/fastlibm.py`)

ArWen's physics carries per-scheme device copies of glibc's logf/expf/powf
(`mynn_glibc_*`, `gfk_*`, `rlw_*`, `rsw_*`, `r_*`): table + polynomial in
binary64, rounded once to binary32.  With binary64 emulated on integers
(Apple GPU: shifts 4–8 cycles, 32-bit multiply 4, mulhi 8, FP32 FMA 1; measured
here: binary64 add 21 Gop/s, mul 38, fma 5, against FP32 fma 1011) these copies
dominated MYNN, Noah, Tiedtke and radiation kernels (a diagnostic build with
hardware libm cut `mynn_surface_column` from 48× to 7× the RTX 5090 time).

`arwen_fastlibm_{logf,expf,powf}.h` evaluate glibc's own formula (its tables
and coefficients, `tools/metal/gen_fastlibm_tables.py`) in float-float on the
FP32 FMA hardware and return a value only when every number within a declared
error budget of it (2^-36 relative, plus terms for cancellation and |y log2 x|)
rounds to the same binary32; otherwise — specials, tiny/huge results, values
near a rounding boundary — the exact original runs.  The translator routes a
copy only when its fingerprint (the function, every table/helper it reaches,
the compile options, the fast-path sources) is in `fastlibm_receipt.json`,
which `tools/metal/verify_fastlibm.py` fills after comparing routed and
original on **all 2^32 binary32 inputs** on the GPU.  powf is routed per call
site, only where the exponent is a compile-time constant, and each (copy,
exponent) pair is verified on all 2^32 x; variable-exponent calls stay exact.
A constant exponent is a literal, an expression of literals under binary32
`+ - * /` (`__f*_rn` or helpers that are exactly one PTX `*.rn.f32`), or a
name every declaration of which is `const float|real` with such an
initializer and one value; pure forwarding aliases (`mynn_powf`) are followed.  Unverified copies
are recorded under the cache directory for the verifier.

Receipt for the Sicily2KM_ECM run: 55 (copy, options[, exponent]) entries —
logf, expf, powf in MYNN, Noah/Tiedtke (gfk), RRTMG LW/SW — each 4,294,967,296
inputs, 0 differing words; fast-path share of all inputs 0.50 (logf),
0.52 (expf), 0.12–0.50 (powf, by exponent).

End-to-end: the 1-hour forecast with every optimization of this section and
the two before it is **bit-identical** to the 1-hour run made before them
(`tools/metal/wrfout_bitwise.py`, 01:00 d01 + d02: 142,775,319 words, 0 differ).
Per-step wall time over the hour: d02 median 1.238 → 0.471 s, d01 0.659 →
0.311 s; sum of steps 408 → 179 s (second 1-hour run, constant exponents
included: again 0 differing words of 142.8 M).

## Host/GPU overlap (measured with `ARWEN_METAL_TIMELINE`)

`ARWEN_METAL_TIMELINE=<file{pid}.npz>` records every command buffer's GPU
start/end and every host synchronize() in one time base.  On the 10-minute run
it showed ~1100 idle gaps of ~9 ms between command buffers.  A microbenchmark
isolated the cause: command buffers were ordered by an `MTLSharedEvent` chain
(needed: resources are untracked and addressed bindlessly, so Metal would let
consecutive command buffers overlap), and each wait/signal through the shared
event cost the GPU 0.3–2.6 ms.  A plain `MTLEvent` (GPU-side only) gives the
same ordering with a 1 µs gap (`tools/metal/ordering_check.py`: dependent
read-after-write chains, batch 1–256, eager 0–16, 48 configurations exact).
Commits are also issued after 16 dispatches when the GPU has run dry
(`ARWEN_METAL_EAGER`), so work after a host synchronization starts at once.

1-hour forecast after this change: 165 s wall in total (first Metal hour:
436 s), steps d02 median 0.406 s, d01 0.284 s, sum 152 s; 01:00 wrfout again
bit-identical to the pre-optimization run (142.8 M words).
