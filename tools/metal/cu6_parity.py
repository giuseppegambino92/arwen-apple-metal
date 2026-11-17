"""cu6 (original Tiedtke) on Metal vs WRF's Fortran oracle and the CUDA kernel.

Inputs and reference outputs come from tools/tiedtke_wrf461_oracle run on a
Linux CPU host (gfortran -O0 -ffp-contract=off, glibc >= 2.41) plus the same
production CUDA kernel on an RTX 5090, saved as one npz (dump_cu6.py in the
arwen-metal repo).  Here the production loader (gpuwm.core.tiedtke.column_step)
runs on Metal on the same float32 words; every output word is compared with
the oracle's compare_words.  Declared tolerance: 0 differing words (the CUDA
kernel's documented result).

    GPUWM_BACKEND=metal python tools/metal/cu6_parity.py CASES.npz RECEIPT.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GPUWM_BACKEND", "metal")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "tiedtke_wrf461_oracle"))
import gpuwm  # noqa: E402,F401  (installs the Metal cupy shim)
import numpy as np  # noqa: E402
import check as C  # noqa: E402

cases = np.load(sys.argv[1])
meta = json.loads(Path(sys.argv[1] + ".json").read_text())
receipt, failed = {"backend": "metal", "f64_mode": os.environ.get("ARWEN_METAL_F64", "sf64"),
                   "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "0"), "cases": []}, False
for case in meta["cases"]:
    key, cfg = case["key"], case["cfg"]
    inputs = {n.split("/")[-1]: cases[n] for n in cases.files if n.startswith(f"{key}/in/")}
    entry = {"key": key, "nz": case["nz"], "cfg": cfg}
    runs = {"single": C.gpu(inputs, **cfg)}
    for w in C.ROW_WIDTHS:
        runs[f"rows{w}"] = C.gpu(inputs, **cfg, row_width=w)
    for label, ref_prefix in (("vs_fortran", "fortran"), ("vs_cuda", "cuda")):
        out = {}
        for run, actual in runs.items():
            suffix = "" if run == "single" else "_" + run
            fields = {}
            for name in C.FIELDS:
                ref = cases[f"{key}/{ref_prefix}{suffix}/{name}"]
                r = C.compare_words(ref, actual[name])
                fields[name] = {k: r[k] for k in ("differing_words", "max_ulp", "max_abs_difference", "nonfinite_words")}
                failed |= bool(r["differing_words"])
            out[run] = fields
        entry[label] = out
    receipt["cases"].append(entry)
    s = {n: (v["differing_words"], v["max_ulp"]) for n, v in entry["vs_fortran"]["single"].items()}
    print(key, cfg, "vs Fortran (differing words, max ULP):", s, flush=True)
receipt["pass"] = not failed
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=1))
tot = sum(v["differing_words"] for c in receipt["cases"] for lab in ("vs_fortran", "vs_cuda")
          for run in c[lab].values() for v in run.values())
print(f"pass={not failed} total differing words (all cases, runs, both references)={tot}")
sys.exit(1 if failed else 0)
