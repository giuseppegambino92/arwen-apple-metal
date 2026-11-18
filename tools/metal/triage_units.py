"""First error of every failing unit in a compile_all_units.py report, with the MSL line.

    python tools/metal/triage_units.py OUT_DIR [--grep TEXT]
"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
pat = sys.argv[sys.argv.index("--grep") + 1] if "--grep" in sys.argv else None
for r in json.loads((out / "report.json").read_text()):
    if r["stage"] == "ok":
        continue
    e = r["error"]
    if r["stage"] == "msl":
        m = re.search(r"module\.metal:(\d+):(\d+): error: ([^\n]*)", e)
        line = ""
        if m and (out / f"{r['unit']}.metal").exists():
            src = (out / f"{r['unit']}.metal").read_text().splitlines()
            line = src[int(m.group(1)) - 1].strip()[:220]
        msg = f"{m.group(3)[:160]}\n      >> {line}" if m else e[-300:]
    else:
        msg = e.replace("\n", " | ")[:300]
    if pat and pat not in msg:
        continue
    print(f"{r['unit']:32s} [{r['stage']}] {msg}")
