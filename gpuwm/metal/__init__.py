"""Apple Metal backend for gpuwm (in progress; see docs/metal-backend.md).

The CUDA kernel sources remain canonical: ``gpuwm.metal.transpile`` translates
the exact source string gpuwm hands to NVRTC into Metal Shading Language.
"""
