// Parse-only declarations of the CUDA device builtins ArWen's kernels use.
// clang parses kernel sources in CUDA device mode with -nocudainc; this file
// is the authoritative list of what the CUDA->MSL translator supports: an
// identifier missing here is a parse error, never a silent miscompile.
#pragma once
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __grid_constant__ __attribute__((annotate("grid_constant")))
#define __restrict__ __restrict
typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef signed char int8_t; typedef short int16_t; typedef int int32_t; typedef long long int64_t;
typedef unsigned char uint8_t; typedef unsigned short uint16_t; typedef unsigned int uint32_t; typedef unsigned long long uint64_t;
#define warpSize 32
#define FLT_MIN 1.17549435e-38f
#define FLT_MAX 3.40282347e+38f
#define FLT_EPSILON 1.19209290e-7f
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e+308
#define DBL_EPSILON 2.2204460492503131e-16
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#define HUGE_VALF (__builtin_huge_valf())
#define HUGE_VAL (__builtin_huge_val())
#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)
#define UINT_MAX 4294967295U
struct __cuda_idx3 { unsigned int x, y, z; };
extern __device__ const __cuda_idx3 threadIdx, blockIdx, blockDim, gridDim;
#define ARWEN_VEC2(T, N) struct N { T x, y; }; __device__ inline N make_##N(T a, T b) { return N{a, b}; }
#define ARWEN_VEC3(T, N) struct N { T x, y, z; }; __device__ inline N make_##N(T a, T b, T c) { return N{a, b, c}; }
#define ARWEN_VEC4(T, N) struct N { T x, y, z, w; }; __device__ inline N make_##N(T a, T b, T c, T d) { return N{a, b, c, d}; }
ARWEN_VEC2(float, float2) ARWEN_VEC3(float, float3) ARWEN_VEC4(float, float4)
ARWEN_VEC2(double, double2) ARWEN_VEC2(int, int2) ARWEN_VEC3(int, int3) ARWEN_VEC4(int, int4)
ARWEN_VEC2(unsigned int, uint2) ARWEN_VEC3(unsigned int, uint3) ARWEN_VEC4(unsigned int, uint4)
ARWEN_VEC2(long long, longlong2) ARWEN_VEC4(double, double4) ARWEN_VEC3(double, double3)
struct dim3 { unsigned int x, y, z; };
// synchronisation and warp-level primitives
__device__ void __syncthreads(); __device__ void __syncwarp(unsigned m = 0xffffffffu);
__device__ void __threadfence(); __device__ void __threadfence_block();
__device__ unsigned __ballot_sync(unsigned, int);
__device__ int __any_sync(unsigned, int); __device__ int __all_sync(unsigned, int);
#define ARWEN_SHFL(T) \
  __device__ T __shfl_sync(unsigned, T, int, int w = 32); __device__ T __shfl_up_sync(unsigned, T, unsigned, int w = 32); \
  __device__ T __shfl_down_sync(unsigned, T, unsigned, int w = 32); __device__ T __shfl_xor_sync(unsigned, T, int, int w = 32);
ARWEN_SHFL(int) ARWEN_SHFL(unsigned) ARWEN_SHFL(float) ARWEN_SHFL(double) ARWEN_SHFL(long long) ARWEN_SHFL(unsigned long long)
// atomics
#define ARWEN_ATOM(T) \
  __device__ T atomicAdd(T*, T); __device__ T atomicSub(T*, T); __device__ T atomicExch(T*, T); \
  __device__ T atomicMin(T*, T); __device__ T atomicMax(T*, T); __device__ T atomicCAS(T*, T, T); \
  __device__ T atomicOr(T*, T); __device__ T atomicAnd(T*, T); __device__ T atomicXor(T*, T);
ARWEN_ATOM(int) ARWEN_ATOM(unsigned int) ARWEN_ATOM(unsigned long long) ARWEN_ATOM(long long)
__device__ float atomicAdd(float*, float); __device__ double atomicAdd(double*, double); __device__ float atomicExch(float*, float);
// bit reinterpretation and conversions
__device__ unsigned __float_as_uint(float); __device__ int __float_as_int(float);
__device__ float __uint_as_float(unsigned); __device__ float __int_as_float(int);
__device__ long long __double_as_longlong(double); __device__ double __longlong_as_double(long long);
__device__ int __double2hiint(double); __device__ int __double2loint(double); __device__ double __hiloint2double(int, int);
__device__ float __double2float_rn(double); __device__ float __double2float_rz(double);
__device__ int __double2int_rz(double); __device__ int __double2int_rn(double); __device__ long long __double2ll_rn(double); __device__ long long __double2ll_rz(double);
__device__ double __int2double_rn(int); __device__ float __int2float_rn(int); __device__ double __ll2double_rn(long long);
__device__ int __float2int_rz(float); __device__ int __float2int_rn(float); __device__ int __float2int_rd(float); __device__ unsigned __float2uint_rz(float);
__device__ long long __float2ll_rz(float); __device__ long long __float2ll_rn(float);
// IEEE round-to-nearest arithmetic
__device__ float __fadd_rn(float, float); __device__ float __fsub_rn(float, float); __device__ float __fmul_rn(float, float);
__device__ float __fdiv_rn(float, float); __device__ float __fsqrt_rn(float); __device__ float __fmaf_rn(float, float, float);
__device__ float __frcp_rn(float); __device__ float __fdividef(float, float); __device__ float __expf(float); __device__ float __logf(float); __device__ float __powf(float, float);
__device__ double __dadd_rn(double, double); __device__ double __dsub_rn(double, double); __device__ double __dmul_rn(double, double);
__device__ double __ddiv_rn(double, double); __device__ double __dsqrt_rn(double); __device__ double __fma_rn(double, double, double); __device__ double __drcp_rn(double);
// integer intrinsics
__device__ int __clz(int); __device__ int __clzll(long long); __device__ int __popc(unsigned); __device__ int __popcll(unsigned long long);
__device__ int __ffs(int); __device__ int __ffsll(long long); __device__ unsigned long long __umul64hi(unsigned long long, unsigned long long);
__device__ long long __mul64hi(long long, long long); __device__ unsigned __umulhi(unsigned, unsigned); __device__ unsigned __brev(unsigned);
// cache-hinted loads
template <class T> __device__ T __ldg(const T*); template <class T> __device__ T __ldcg(const T*); template <class T> __device__ T __ldca(const T*);
// math: float, double and the CUDA overloads of the double names for float
#define ARWEN_M1(n) __device__ float n##f(float); __device__ double n(double); __device__ float n(float);
#define ARWEN_M2(n) __device__ float n##f(float, float); __device__ double n(double, double); __device__ float n(float, float);
ARWEN_M1(exp) ARWEN_M1(exp2) ARWEN_M1(exp10) ARWEN_M1(expm1) ARWEN_M1(log) ARWEN_M1(log2) ARWEN_M1(log10) ARWEN_M1(log1p)
ARWEN_M1(sqrt) ARWEN_M1(rsqrt) ARWEN_M1(cbrt) ARWEN_M1(sin) ARWEN_M1(cos) ARWEN_M1(tan) ARWEN_M1(asin) ARWEN_M1(acos) ARWEN_M1(atan)
ARWEN_M1(sinh) ARWEN_M1(cosh) ARWEN_M1(tanh) ARWEN_M1(asinh) ARWEN_M1(acosh) ARWEN_M1(atanh) ARWEN_M1(fabs) ARWEN_M1(floor) ARWEN_M1(ceil)
ARWEN_M1(trunc) ARWEN_M1(round) ARWEN_M1(rint) ARWEN_M1(nearbyint) ARWEN_M1(erf) ARWEN_M1(erfc) ARWEN_M1(lgamma) ARWEN_M1(tgamma)
ARWEN_M2(pow) ARWEN_M2(atan2) ARWEN_M2(fmax) ARWEN_M2(fmin) ARWEN_M2(fmod) ARWEN_M2(hypot) ARWEN_M2(copysign) ARWEN_M2(fdim) ARWEN_M2(remainder) ARWEN_M2(nextafter)
__device__ float fmaf(float, float, float); __device__ double fma(double, double, double); __device__ float fma(float, float, float);
__device__ float ldexpf(float, int); __device__ double ldexp(double, int); __device__ float frexpf(float, int*); __device__ double frexp(double, int*);
__device__ float modff(float, float*); __device__ double modf(double, double*); __device__ void sincosf(float, float*, float*); __device__ void sincos(double, double*, double*);
__device__ long lrintf(float); __device__ long lround(double); __device__ long long llrint(double); __device__ long long llroundf(float); __device__ long lroundf(float);
__device__ float nanf(const char*); __device__ double nan(const char*);
__device__ float rnormf(int, const float*); __device__ float norm3df(float, float, float);
__device__ bool isnan(float); __device__ bool isnan(double); __device__ bool isinf(float); __device__ bool isinf(double);
__device__ bool isfinite(float); __device__ bool isfinite(double); __device__ bool signbit(float); __device__ bool signbit(double);
#define ARWEN_MM(T) __device__ T min(T, T); __device__ T max(T, T);
ARWEN_MM(int) ARWEN_MM(unsigned int) ARWEN_MM(long long) ARWEN_MM(unsigned long long) ARWEN_MM(float) ARWEN_MM(double) ARWEN_MM(long) ARWEN_MM(unsigned long)
__device__ int abs(int); __device__ long long llabs(long long); __device__ long labs(long);
__device__ void __trap(); __device__ void __brkpt();
// CUDA's mixed-type min/max overloads (crt/math_functions.hpp): the result is the wider/unsigned type.
#define ARWEN_MMX(R, A, B) __device__ R min(A, B); __device__ R max(A, B);
ARWEN_MMX(unsigned int, int, unsigned int) ARWEN_MMX(unsigned int, unsigned int, int)
ARWEN_MMX(unsigned long long, long long, unsigned long long) ARWEN_MMX(unsigned long long, unsigned long long, long long)
ARWEN_MMX(unsigned long, long, unsigned long) ARWEN_MMX(unsigned long, unsigned long, long)
ARWEN_MMX(double, float, double) ARWEN_MMX(double, double, float)

// ArWen Metal: fast paths of the glibc float libm copies (arwen_fastlibm.h),
// called only by wrappers gpuwm.metal.fastlibm inserts for verified copies.
__device__ float __arwen_fast_glibc_logf(float); __device__ float __arwen_fast_glibc_expf(float);
__device__ float __arwen_fast_glibc_powf(float, float);
