"""Out-of-process CUDA -> MSL translation.

The translator drives libclang through ctypes.  Run inside a forecast process,
that native code corrupted unrelated Python objects (module globals replaced by
translator tokens, under free-threaded Python 3.14 as well as with the GIL;
none once every module was cached).  The runtime therefore translates in a
child process: this module reads a JSON request, translates, and writes the
MSL, the Metal flags and the module metadata (or the TranslationError) as JSON.

    python -m gpuwm.metal.translate_worker REQUEST.json RESULT.json
"""
from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    request, result = argv
    req = json.loads(open(request, encoding="utf-8").read())
    from gpuwm.metal import transpile
    from gpuwm.metal.frontend import TranslationError
    try:
        t = transpile.translate(req["source"], tuple(req["options"]), f64_mode=req["f64_mode"],
                                filename=req["filename"], cupy_raw_module=req["cupy_raw_module"])
        out = {"ok": True, "msl": t.msl, "metal_flags": list(t.metal_flags), "metadata": t.metadata()}
    except TranslationError as e:
        out = {"ok": False, "error": str(e)}
    with open(result, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
