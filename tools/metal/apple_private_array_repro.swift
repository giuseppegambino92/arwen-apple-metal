// Apple GPU compiler: a constant-index read of a 64-element private array,
// after a loop that reads the array, returns 0.  Standalone Metal repro.
import Metal
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void v(device const float* air [[buffer(0)]], device float* qnr [[buffer(1)]],
              device float* out [[buffer(2)]], constant int& nz [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float rho[N];
    for (int k = 0; k < nz; ++k) { rho[k] = air[k]; }
    float s = 0.0f;
    for (int k = 0; k < nz; ++k) { s += rho[k]; }
    qnr[0] = s;
    out[0] = rho[0];
}
"""
let dev = MTLCreateSystemDefaultDevice()!
print("device:", dev.name)
for n in [32, 63, 64, 65, 128] {
    let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "rho[N]", with: "rho[\(n)]"), options: nil)
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "v")!)
    var air: [Float] = (0..<12).map { 1.2 - Float($0) * 0.8 / 11 }
    var nz: Int32 = 12
    let a = dev.makeBuffer(bytes: &air, length: 48, options: .storageModeShared)!
    let q = dev.makeBuffer(length: 48, options: .storageModeShared)!
    let o = dev.makeBuffer(length: 4, options: .storageModeShared)!
    let queue = dev.makeCommandQueue()!, cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso); enc.setBuffer(a, offset: 0, index: 0); enc.setBuffer(q, offset: 0, index: 1)
    enc.setBuffer(o, offset: 0, index: 2); enc.setBytes(&nz, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let got = o.contents().load(as: Float.self)
    print("float rho[\(n)]: rho[0] after the loops = \(got)  (expected 1.2)")
}
