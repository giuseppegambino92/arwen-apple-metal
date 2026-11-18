"""Group the failures of a pytest JUnit XML report by error signature.

    python tools/metal/triage_pytest.py REPORT.xml [--top 40] [--json OUT]

A signature is the exception type plus its message with numbers, hex, paths and
quoted names normalised, so a hundred instances of one missing CuPy attribute
count as one line.  Each group lists its test files.
"""
import argparse
import collections
import json
import re
import xml.etree.ElementTree as ET


def signature(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    pick = next((l for l in lines if re.match(r"^E\s+\w*(Error|Exception|Failed|Interrupt|Exit)\b", l)), None)
    if pick is None:
        pick = next((l for l in reversed(lines) if re.search(r"(Error|Exception):", l)), lines[-1] if lines else "")
    s = re.sub(r"^E\s+", "", pick.strip())
    s = re.sub(r"/[\w./+-]+", "<path>", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "<hex>", s)
    s = re.sub(r"\b\d+(\.\d+)?(e[-+]?\d+)?\b", "<n>", s)
    s = re.sub(r"(['\"]).{40,}?\1", r"\1…\1", s)
    return s[:220]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json")
    a = ap.parse_args()
    root = ET.parse(a.report).getroot()
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    for tc in root.iter("testcase"):
        res = next((c for c in tc if c.tag in ("failure", "error", "skipped")), None)
        state = "passed" if res is None else res.tag
        counts[state] += 1
        if state in ("failure", "error"):
            text = (res.get("message") or "") + "\n" + (res.text or "")
            groups[signature(text)].append((tc.get("classname", ""), tc.get("name", "")))
    print(dict(counts))
    for sig, tests in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:a.top]:
        files = collections.Counter(c.rsplit(".", 1)[0] for c, _ in tests)
        print(f"{len(tests):5d}  {sig}")
        print("       " + ", ".join(f"{f.split('.')[-1]}({n})" for f, n in files.most_common(6)))
    if a.json:
        json.dump({k: v for k, v in groups.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
