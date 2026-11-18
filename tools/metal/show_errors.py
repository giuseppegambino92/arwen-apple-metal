"""First MSL errors of one unit with the offending MSL lines: show_errors.py OUT_DIR UNIT [N]"""
import json
import re
import sys
from pathlib import Path

out, unit = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
r = next(x for x in json.loads((out / "report.json").read_text()) if x["unit"] == unit)
print(unit, r["stage"])
if r["stage"] != "msl":
    print(r.get("error", "")[:1500])
    sys.exit()
src = (out / f"{unit}.metal").read_text().splitlines()
err = (out / f"{unit}.err").read_text() if (out / f"{unit}.err").exists() else r["error"]
for m in list(re.finditer(r"module\.metal:(\d+):(\d+): error: ([^\n]*)", err))[:n]:
    ln = int(m.group(1))
    print(f"--- {m.group(3)[:200]}")
    for k in range(max(0, ln - 2), min(len(src), ln + 1)):
        print(f"  {k + 1}: {src[k][:240]}")
