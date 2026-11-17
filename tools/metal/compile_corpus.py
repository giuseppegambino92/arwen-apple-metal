"""Translate and compile a recorded CUDA module corpus to Metal (Milestone 2 gate).

A corpus is a directory with ``index.json`` and ``src/<sha>.cu`` as written by the
arwen-metal recording probe (every module a real gpuwm run compiled).  For each
hand-written module: CUDA -> MSL (gpuwm.metal.transpile) -> metallib (xcrun metal).

    python tools/metal/compile_corpus.py CORPUS_DIR [--mode sf64] [--jobs 10] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def one(args):
    corpus, entry, mode, out = args
    from gpuwm.metal.transpile import translate
    from gpuwm.metal.frontend import TranslationError
    from gpuwm.metal.msl_compile import compile_msl, MetalCompileError
    src = (Path(corpus) / "src" / f"{entry['sha']}.cu").read_text()
    site = entry["sites"][0]
    raw = site.get("kind") == "aot_record" and site.get("route") != "direct-nvrtc"
    t0 = time.time()
    try:
        t = translate(src, site.get("options", []), f64_mode=mode, filename=site.get("filename") or "kernel.cu",
                      cupy_raw_module=raw)
    except TranslationError as e:
        return dict(sha=entry["sha"], stage="translate", error=str(e)[:2000])
    except Exception:
        return dict(sha=entry["sha"], stage="translate-crash", error=traceback.format_exc()[-2000:])
    t1 = time.time()
    Path(out, "msl").mkdir(parents=True, exist_ok=True)
    Path(out, "msl", f"{entry['sha']}.metal").write_text(t.msl)
    try:
        compile_msl(t.msl, t.metal_flags, out=Path(out, "lib", f"{entry['sha']}.metallib"))
    except MetalCompileError as e:
        errs = [l for l in str(e).splitlines() if ": error:" in l]
        return dict(sha=entry["sha"], stage="compile", error="\n".join(errs[:30]) or str(e)[-2000:],
                    nerr=len(errs), t_translate=t1 - t0)
    return dict(sha=entry["sha"], stage="ok", kernels=len(t.kernels), t_translate=t1 - t0, t_compile=time.time() - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--mode", default="sf64")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="/tmp/arwen-metal-corpus")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    idx = json.loads(Path(args.corpus, "index.json").read_text())
    hand = [e for e in idx if not e["cupy_generated"] and (not args.only or args.only in e["srcfile"] or args.only == e["sha"])]
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(one, [(args.corpus, e, args.mode, args.out) for e in hand]))
    by = {e["sha"]: e for e in hand}
    stages = collections.Counter(r["stage"] for r in res)
    kern = sum(r.get("kernels", 0) for r in res)
    print(f"modules={len(res)} {dict(stages)} kernels_compiled={kern} wall={time.time()-t0:.0f}s mode={args.mode}")
    causes = collections.Counter()
    for r in res:
        if r["stage"] != "ok":
            print(f"--- {r['sha']} {by[r['sha']]['srcfile']} [{r['stage']}]")
            print("    " + r["error"].replace("\n", "\n    ")[:1500])
            for l in r["error"].splitlines():
                m = re.search(r"error: (.*)", l)
                if m:
                    causes[re.sub(r"'[^']*'", "'X'", m.group(1))[:120]] += 1
    print("\nerror causes:")
    for k, v in causes.most_common(40):
        print(f"{v:5d}  {k}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "report.json").write_text(json.dumps(res, indent=1))
    return 0 if stages.get("ok", 0) == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
