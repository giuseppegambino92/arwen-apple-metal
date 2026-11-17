# Metal backend (Apple Silicon) — design notes and measurements

Status: **in progress**. Milestones 1–2 (environment, CUDA→MSL translator) done;
runtime, `xp` layer and kernel parity (Milestones 3–7) not yet.

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
