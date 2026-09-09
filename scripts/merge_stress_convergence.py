"""Merge the sharded stress-test convergence runs into one file per truncation.

stress_3omega_convergence.py writes data/stress_3omega_convergence_nh<nh><tag>.json
when several W3_LIST ranges are run in parallel.  This collects the shards,
sorts by comb gain, and writes data/stress_3omega_convergence_nh<nh>.json.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

for nh in (7, 9):
    shards = sorted(glob.glob(os.path.join(
        DATA, f"stress_3omega_convergence_nh{nh}_*.json")))
    rows, meta = [], None
    for f in shards:
        d = json.load(open(f))
        meta = meta or d
        rows.extend(d["rows"])
    if not rows:
        continue
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r["eta3"]):
        if r["w3"] in seen:
            continue
        seen.add(r["w3"]); uniq.append(r)
    out = os.path.join(DATA, f"stress_3omega_convergence_nh{nh}.json")
    json.dump(dict(nh=nh, windows=meta["windows"], beta1=meta["beta1"],
                   rows=uniq), open(out, "w"), indent=2)
    print(f"nh={nh}: {len(uniq)} members from {len(shards)} shards -> {out}")
