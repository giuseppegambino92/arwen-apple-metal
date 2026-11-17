"""Time Metal pipeline creation (backend ISA compile) per kernel for corpus modules."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.metal import runtime

corpus = Path(sys.argv[1])
pattern = sys.argv[2] if len(sys.argv) > 2 else ""
idx = [e for e in json.loads((corpus / "index.json").read_text()) if not e["cupy_generated"] and pattern in e["srcfile"]]
seen = set()
for e in idx:
    if e["srcfile"] in seen:
        continue
    seen.add(e["srcfile"])
    site = e["sites"][0]
    raw = site.get("kind") == "aot_record" and site.get("route") != "direct-nvrtc"
    t0 = time.time()
    m = runtime.Module((corpus / "src" / f"{e['sha']}.cu").read_text(), site.get("options", []), cupy_raw_module=raw,
                       filename=site.get("filename") or "kernel.cu")
    t1 = time.time()
    worst = []
    for name in m.meta["kernels"]:
        a = time.time()
        p = m.pipeline(name)
        worst.append((time.time() - a, name, p.max_threads))
    worst.sort(reverse=True)
    print(f"{e['srcfile']}: translate+metallib {t1-t0:.1f}s, {len(worst)} pipelines {sum(w[0] for w in worst):.1f}s; "
          f"slowest {[(round(w[0],1), w[1], w[2]) for w in worst[:3]]}", flush=True)
