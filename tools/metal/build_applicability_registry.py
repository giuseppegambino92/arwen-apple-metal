"""Build tests/metal_not_applicable.py's ENTRIES from classified Metal failures.

    python tools/metal/build_applicability_registry.py FAILURES.json [--cuda-same SAME.json]
        [--cuda-pass PASS.json] [--cuda-junit 'DIR/*.xml'] [--write]

FAILURES.json maps node ids (JUnit ``classname::name``) to [state, message] for
the tests failing on Metal.  A test enters the registry only with evidence for
its class:

* message evidence for machinery Metal does not have (PTX/NVRTC/nvJitLink,
  CUDA introspection) or for data/tools present only on the CUDA/Linux host;
* --cuda-same: the test fails on an RTX 5090 with the same message (ArWen drift);
* --cuda-pass: the test passes on the RTX 5090 and pins CUDA's output words;
* --cuda-junit: JUnit files of the same tests run on the RTX 5090 (one process
  per file); a test failing there with the same kind of failure -- the same
  first characters of the message once checkout paths are removed -- counts as
  --cuda-same, a passing one as --cuda-pass;
* the files whose CPU reference was shown to run on the Mac's own libm/NumPy;
* the KF default-tolerance tests (pass with ARWEN_METAL_STRICT_GLIBC=1).

Everything else stays a real failure.  Without --write it prints the counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

RULES = [
    ("ptx", r"too many values to unpack|createProgram|compileProgram|getPTX|nvJitLink|cubin|"
            r"test_literal_division_gate|constant_division_rewrites"),
    ("cuda_introspection", r"local_size|local frame|frame bytes|funcGetAttribute: not a Metal|kernel_local_bounds|"
                           r"tgammaf matched|no longer flushes subnormals on this toolchain|"
                           r"widens the unpinned|NVRTC_UNPINNED"),
    ("cuda_host_only", r"needs Linux x86_64|HYDRO_CLOCK_BASELINE|NEXTGEN_LAKE_FRAGMENT|ISHMAEL\) needs WRF|"
                       r"MissingWifClimatologyDataset|GPUWM_THOMPSON_HAIL_TABLE"),
    # the message itself says the comparison is against CUDA's recorded output words
    ("cuda_hash_pin", r"drift on sm_|complete output case differs from every current-source pin"),
    # errors raised by the test's own fixture: independent of the backend
    ("arwen_drift", r"'types\.SimpleNamespace' object has no attribute|"
                    r"RunConfig\.__init__\(\) got an unexpected keyword argument|"
                    r"<locals>\.\w+\(\) got an unexpected keyword argument"),
]
#: measured: ArWen's default numerics use the GPU's own powf; Metal's is farther
#: from glibc than CUDA's (pi_phy: 1826 vs 304 words differing from WRF)
NATIVE_LIBM_TESTS = {
    "tests.test_bigstep_prep_wrf471_parity::test_bigstep_phy_prep_compiled_wrf471_exposed_outputs",
    # ratchets at the error of CUDA's hardware powf(smob, b) (non-integer exponent)
    "tests.test_thompson_aerosol_device_helpers::test_snow_number_end_to_end_through_the_fits_is_the_cold_network_path",
}
#: tests run individually on 2026-11-19 after the array-layer fixes, each read:
EXPLICIT = {
    # negative controls of nvrtc/ptxas FMA contraction: the effect they pin is
    # the NVIDIA compiler's; Metal compiles without contraction
    "tests.test_thompson_aerosol_device_helpers::test_the_hoisted_mp8_field_fit_really_does_disagree_with_wrf": "cuda_contraction",
    "tests.test_thompson_aerosol_adapter::test_the_contraction_pinned_saturation_fit_is_load_bearing_for_g3": "cuda_contraction",
    # two CUDA streams racing (a spin-wait gate released from another stream);
    # the Metal backend runs every stream on one in-order queue
    "tests.test_rrtmg_sw_stream_order::test_delayed_table_upload_orders_two_member_streams": "cuda_streams",
    # AOT cubin packs ("Real cubin load") and the NVRTC library digest
    "tests.test_kernel_aot_gpu::test_real_cupy_bridge_preserves_template_name_mapping": "ptx",
    "tests.test_kernel_aot_gpu::test_direct_pack_keeps_both_numerical_modes_when_builder_defaults_fast": "ptx",
    # gpuwm.nvrtc_ptx_cache: the native (cubin) image cache
    "tests.test_nvrtc_ptx_cache::test_native_image_cache_retains_subnormal_words_and_recovers_corruption": "ptx",
    # RTX 5090: AttributeError: 'cupy.cuda.function.Module' object has no attribute 'module'
    "tests.test_ssib_wrf461_parity::test_table_views_accept_both_cuda_module_interfaces": "arwen_drift",
}
#: files whose CPU reference was measured to be the Mac's own libm / NumPy (2026-11-18)
MACOS_REFERENCE_FILES = {
    "tests.test_fast_sbm_device": "fast_sbm host authority links Apple libm (ArWen's host.py documents it)",
    "tests.test_wrf_vert_identity_gpu": "the Rust bridge's f32::ln is Apple logf; Metal = glibc 0x41116539",
    "tests.test_real_nextafter_gpu": "NumPy arm64 nextafter returns a fixed NaN; Metal propagates payloads as glibc",
}
MACOS_REFERENCE_TESTS = {
    "tests.test_rrtmg_legacy_prep_device::test_preflight_and_numpy_minmax",
}
KF_FILES = ("tests.test_kfcup_cuda_oracle", "tests.test_kfcup_runtime")
HASH_PIN = r"current-source pin|output drift|drift on sm_|reference_sha256|actual_sha256"


def classify(failures, same=(), cuda_pass=()):
    same, cuda_pass = set(same), set(cuda_pass)
    out = {}
    for nodeid, (state, message) in failures.items():
        module = nodeid.split("::", 1)[0]
        text = nodeid + " " + (message or "")
        cls = EXPLICIT.get(nodeid.split("[", 1)[0])
        cls = cls or next((name for name, pat in RULES if re.search(pat, text)), None)
        # before the CUDA evidence: this class keeps them running under STRICT_GLIBC
        if cls is None and module in KF_FILES:
            cls = "kf_default_tolerance"
        if cls is None and nodeid in same:
            cls = "arwen_drift"
        if cls is None and nodeid in cuda_pass and re.search(HASH_PIN, text):
            cls = "cuda_hash_pin"
        if cls is None and (module in MACOS_REFERENCE_FILES or nodeid in MACOS_REFERENCE_TESTS):
            cls = "macos_reference"
        if cls is None and module in KF_FILES:
            cls = "kf_default_tolerance"
        if cls is None and nodeid.split("[", 1)[0] in NATIVE_LIBM_TESTS:
            cls = "native_libm"
        if cls is not None:
            out[nodeid] = cls
    return out


def _junit(pattern):
    import glob
    import xml.etree.ElementTree as ET
    out = {}
    for f in sorted(glob.glob(pattern)):
        for tc in ET.parse(f).getroot().iter("testcase"):
            r = next((c for c in tc if c.tag in ("failure", "error", "skipped")), None)
            state = "passed" if r is None else ("skipped" if r.tag == "skipped" else "failed")
            out[tc.get("classname") + "::" + tc.get("name")] = (state, (r.get("message") if r is not None else "") or "")
    return out


def _head(message, n=25):
    """The kind of a failure: its first characters without checkout paths."""
    m = re.sub(r"(/Users|/home)/[^\s'\"]*?/(arwen|pr_final)/", "<root>/", message.split("\n", 1)[0])
    return m[:n]


def same_on_cuda(failures, cuda):
    same, passed = [], []
    for nodeid, (_, message) in failures.items():
        state, cmsg = cuda.get(nodeid, (None, ""))
        if state == "failed" and _head(cmsg) == _head(message or ""):
            same.append(nodeid)
        elif state == "passed":
            passed.append(nodeid)
    return same, passed


def to_pytest_id(nodeid):
    """JUnit ``tests.test_x::name`` (or ``tests.test_x.Class::name``) -> pytest node id."""
    module, name = nodeid.split("::", 1)
    parts = module.split(".")
    for n in range(len(parts), 0, -1):
        path = "/".join(parts[:n]) + ".py"
        if os.path.isfile(path):
            return "::".join([path] + parts[n:] + [name])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("failures")
    ap.add_argument("--cuda-same")
    ap.add_argument("--cuda-pass")
    ap.add_argument("--cuda-junit")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    failures = json.load(open(a.failures))
    same = json.load(open(a.cuda_same)) if a.cuda_same else []
    cuda_pass = json.load(open(a.cuda_pass)) if a.cuda_pass else []
    if a.cuda_junit:
        s2, p2 = same_on_cuda(failures, _junit(a.cuda_junit))
        same, cuda_pass = list(same) + s2, list(cuda_pass) + p2
    entries = classify(failures, same, cuda_pass)
    ids = {}
    for nodeid, cls in entries.items():
        pid = to_pytest_id(nodeid)
        if pid:
            ids[pid] = cls
    counts = {}
    for cls in ids.values():
        counts[cls] = counts.get(cls, 0) + 1
    print(f"{len(failures)} failing on Metal; registered {len(ids)}: {dict(sorted(counts.items()))}; "
          f"{len(failures) - len(ids)} stay real failures")
    if a.write:
        path = "tests/metal_not_applicable.py"
        src = open(path).read()
        body = "ENTRIES: dict[str, str] = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(ids.items())) + "}\n"
        new = re.sub(r"ENTRIES: dict\[str, str\] = \{[^}]*\}\n", lambda m: body, src, count=1)
        open(path, "w").write(new)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
