"""All birth branches of the 2-DOF network, not just the one a seed finds.

Both `make_fig_prl_design.py` (panel c) and the first version of
`inverse_design_map.py` trace the birth locus by Newton warm-started from the
previous solution.  That follows ONE branch.  Where the locus has more than
one, the sweep silently keeps whichever branch its starting point happened to
sit on, and reports the others as absent.

That is what produced the "no birth" band in Fig. 3(c): sweeping outward from
w2 = 0.995 and w2 = 1.005, the continuation loses the branch near degeneracy
and the gap between the two runs was read as physical.  Marching finely from
both sides instead finds a birth at every w2 across that band.

This script determines the branch structure without warm starts: at each w2 it
seeds Newton from a grid of (Omega, kappa) and collects the distinct converged
births.  Slower, but it cannot miss a branch for want of a starting point.

Output: data/inverse_design_branches.csv  (one row per distinct birth)
Run with the project venv:  ../.venv/bin/python inverse_design_branches.py
"""
from __future__ import annotations

import os

import numpy as np

from inverse_design_map import G, f, df_dOm, a_Omega, solve_birth, BETA_SIGN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

OM_SEEDS = np.arange(0.75, 1.85, 0.05)
KAP_SEEDS = np.array([0.02, 0.05, 0.09, 0.15, 0.22, 0.30, 0.40, 0.55, 0.75])
TOL_SAME = 2e-3            # births closer than this in (Om, kappa) are one


def births_at(w2):
    found = []
    for om0 in OM_SEEDS:
        for k0 in KAP_SEEDS:
            got = solve_birth(w2, (float(om0), float(k0)))
            if got is None:
                continue
            Om, kap, aO = got
            if any(abs(Om - o) < TOL_SAME and abs(kap - k) < TOL_SAME
                   for o, k, _ in found):
                continue
            found.append((Om, kap, aO))
    return sorted(found, key=lambda t: t[0])


def main() -> None:
    w2s = np.arange(0.60, 1.601, 0.01)
    rows = []
    for w2 in w2s:
        bs = births_at(float(w2))
        for Om, kap, aO in bs:
            rows.append(dict(w2=float(w2), Om=Om, kappa=kap, a_Om=aO,
                             typ="beaks" if aO > 0 else "lips"))

    counts = {}
    for w2 in w2s:
        n = sum(1 for r in rows if abs(r["w2"] - w2) < 1e-9)
        counts[round(float(w2), 3)] = n

    nz = [w for w, n in counts.items() if n == 0]
    multi = [w for w, n in counts.items() if n > 1]
    print(f"w2 grid: {len(w2s)} values, {len(rows)} births found")
    print(f"  w2 with NO birth:        {len(nz)}"
          + (f"  {nz}" if nz and len(nz) < 25 else ""))
    print(f"  w2 with MORE than one:   {len(multi)}"
          + (f"  [{min(multi):.2f}, {max(multi):.2f}]" if multi else ""))

    beaks = [r for r in rows if r["typ"] == "beaks"]
    lips = [r for r in rows if r["typ"] == "lips"]
    for name, sub in (("beaks", beaks), ("lips", lips)):
        if not sub:
            continue
        w = np.array([r["w2"] for r in sub])
        o = np.array([r["Om"] for r in sub])
        k = np.array([r["kappa"] for r in sub])
        print(f"  {name:>5}: n={len(sub):4d}  w2 [{w.min():.2f}, {w.max():.2f}]"
              f"  Om* [{o.min():.4f}, {o.max():.4f}]"
              f"  kappa* [{k.min():.4f}, {k.max():.4f}]")

    print("\n  sample across the band Fig. 3(c) marks as 'no birth':")
    for w2t in (0.99, 1.00, 1.01, 1.02, 1.03, 1.05, 1.07):
        bs = [r for r in rows if abs(r["w2"] - w2t) < 1e-9]
        if not bs:
            print(f"    w2={w2t:.2f}: none")
        for r in bs:
            print(f"    w2={w2t:.2f}: Om*={r['Om']:.5f}  kappa*={r['kappa']:.5f}"
                  f"  a_Om={r['a_Om']:+.4g}  {r['typ']}")

    out = os.path.join(DATA, "inverse_design_branches.csv")
    with open(out, "w") as fh:
        fh.write("w2,Om_star,kappa_star,a_Omega,type\n")
        for r in rows:
            fh.write(f"{r['w2']:.4f},{r['Om']:.9f},{r['kappa']:.9f},"
                     f"{r['a_Om']:.6e},{r['typ']}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
