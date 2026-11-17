// arwen_fastlibm.h -- fast paths for ArWen's glibc-exact float libm copies.
//
// ArWen's CUDA physics carries device copies of glibc's logf/expf (ARM
// optimized-routines algorithms: a table plus a short polynomial evaluated in
// binary64, rounded once to binary32).  On the Apple GPU every binary64
// operation is integer emulation (arwen_sf64.h), so one call costs thousands of
// cycles.
//
// The functions here evaluate the SAME formula (glibc's tables and
// coefficients, gen_fastlibm_tables.py) in float-float arithmetic on the FP32
// FMA hardware, about 2^-44 relative, and return a result only when it is
// certain: the float-float value y is farther than a declared error budget E
// from every binary32 rounding boundary, so the binary64 value glibc's formula
// produces (within E of y; this also covers the fma / mul+add variants of the
// copies, which differ by about 2^-53) rounds to the same binary32.  Otherwise
// -- specials, results that are tiny, subnormal, powers of two, near overflow,
// or close to a boundary -- they return ARWEN_FASTLIBM_UNSURE and the caller
// runs the original, exactly emulated function.
//
// A copy is only routed through here when tools/metal/verify_fastlibm.py has
// compared the routed function with the original on all 2^32 inputs on the GPU
// (gpuwm/metal/fastlibm_receipt.json); see gpuwm/metal/fastlibm.py.
#pragma once
#include "arwen_fastlibm_logf.h"
#include "arwen_fastlibm_expf.h"
#include "arwen_fastlibm_powf.h"
