"""Resolve the one non-pair birth event of the validity-boundary ensemble.

`ensemble_validity_boundary.py` walks a coupling grid of NK = 300 points over
[0, 1.5], so one step is 5.02e-3 in kappa. Across its 100 networks it records
236 birth events, of which 235 add exactly two cusps to the locus and one adds
four. A +4 event is either

  (a) two ordinary pair births falling inside one coupling step, or
  (b) a degenerate/multiple birth at a single pole,

and (b) would be a non-generic and much stronger claim, so it must not be
asserted on the strength of a grid that cannot separate the two.

This script locates the event by re-walking the ensemble under the same seed,
then subdivides that one grid step 400-fold and reports every transition inside
it.

Result: the event is in network #86 (N = 6), between kappa = 0.637124 and
0.642140. Refined, it splits into two ordinary pair births,

    kappa = 0.6391179 : count 1 -> 3, new pair at Omega = 2.58861, 2.58998
    kappa = 0.6410744 : count 3 -> 5, new pair at Omega = 2.01219, 2.01275

separated by 1.96e-3 in kappa -- well inside one unrefined step -- and organized
by different poles, near Omega = 2.59 and 2.01. So case (a): every birth in the
ensemble is a clean pair once the coupling grid resolves it.
"""
from __future__ import annotations

import numpy as np

from ensemble_unified import cusp_freqs, random_network
from ensemble_validity_boundary import NS, N_PER_N, SEED

NK = 300
SUBDIV = 400


def main():
    rng = np.random.default_rng(SEED)
    idx = 0
    found = []
    for N in NS:
        for _ in range(N_PER_N):
            w, ze, K0, L = random_network(rng, N)
            idx += 1
            ks = np.linspace(0.0, 1.5, NK)
            lo, hi = 0.5, w.max() + 0.6
            prev, _, _, _ = cusp_freqs(K0, ze, N, lo, hi)
            for j, k in enumerate(ks[1:], start=1):
                cur, _, _, _ = cusp_freqs(K0 + k * L, ze, N, lo, hi)
                if len(cur) - len(prev) >= 4:
                    found.append((idx, N, j, float(ks[j - 1]), float(k),
                                  w, ze, K0, L, lo, hi,
                                  [float(o) for o in prev],
                                  [float(o) for o in cur]))
                prev = cur

    if not found:
        print("no birth event of size >= +4 in the ensemble")
        return

    for (net, N, j, k_lo, k_hi, w, ze, K0, L, lo, hi, before, after) in found:
        print(f"net #{net}  N={N}  step {j}  "
              f"kappa {k_lo:.6f} -> {k_hi:.6f}  (step {k_hi - k_lo:.2e})")
        print(f"  cusps before: {np.round(before, 5)}")
        print(f"  cusps after : {np.round(after, 5)}")
        print(f"  refining that step into {SUBDIV} sub-steps:")
        sub = np.linspace(k_lo, k_hi, SUBDIV + 1)
        prev = None
        trans = 0
        for k in sub:
            cur, _, _, _ = cusp_freqs(K0 + k * L, ze, N, lo, hi)
            if prev is not None and len(cur) != len(prev):
                trans += 1
                print(f"    kappa={k:.7f}  count {len(prev)} -> {len(cur)}  "
                      f"cusps={np.round(cur, 5)}")
            prev = cur
        verdict = ("two ordinary pair births inside one step"
                   if trans >= 2 else
                   "UNRESOLVED at this refinement -- do not call it degenerate")
        print(f"  => {trans} transitions inside one grid step: {verdict}")


if __name__ == "__main__":
    main()
