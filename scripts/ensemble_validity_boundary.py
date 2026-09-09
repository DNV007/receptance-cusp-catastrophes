"""Where the background-sign type criterion stops working.

The Letter reports the *counting* result on a small ensemble: no network
exceeds P-1 rungs, every birth is a clean cusp pair. This script asks the
different question the long paper needs -- for what backgrounds is the
one-evaluation rule

    sign a_Omega  ~=  -sign Re G_a(omega_b)                        (typecrit)

actually reliable? -- and answers it quantitatively rather than anecdotally.

The rule is derived by taking the background G_a real and slowly varying across
the active pole, so it should degrade with the background PHASE and not with a
marginal |Re G_a|. That is the hypothesis under test. We bin every born cusp by

    psi = |Im G_a| / |Re G_a|

at the active pole and report the accuracy of (typecrit) in each bin, together
with where the failures sit in the |Re G_a| distribution.

Ensemble: N = 3..6, N_PER_N realizations each, drawn by the same rule as
`ensemble_unified.py` (on-site frequencies uniform on [0.6, 2.4] sorted with the
driven coordinate at unity, damping ratios uniform on [0.01, 0.04], each pair
coupled with probability 0.7 and weight uniform on [0.3, 1.0]) but from an
INDEPENDENT seed and over a wider range of N, so the poles crowd and the
backgrounds are more strongly complex than in the Letter's N = 3, 4 set.

NB: this is ONE ensemble, and it carries BOTH results the long paper reports at
N = 3..6 -- the ladder counts of its Sec. VI and the phase-resolved validity
boundary of its Sec. VI E.  They are not independent tests of each other and the
manuscript must not describe them as such.  The seed is independent of
`ensemble_unified.py` (the Letter's 24-network N = 3, 4 set) and of nothing else.

This script therefore reports both ladder measures.  A rung is a COEXISTING cusp
pair, so ladder height = (max cusp count - 1) // 2; the count of increase events
is a different and wrong measure, printed only to size the artifact.

Cost is dominated by the frequency scan, not the matrix size, so the run is
roughly linear in the number of networks (~3 s each).
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

from ensemble_unified import (DEDUP, NK, cusp_freqs, f_of, random_network,
                              visible)

SEED = 20260811          # independent of ensemble_unified.SEED (20260718)
N_PER_N = 25
NS = (3, 4, 5, 6)
PSI_EDGES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, np.inf)


def born_cusps(N, w, ze, K0, L):
    """Walk the coupling grid, returning one record per newly born cusp.

    Also returns the two LADDER measures, which are not the same thing and must
    not be confused (see ensemble_unified.py's header):

      max_count   the largest cusp count seen anywhere on the grid.  A rung is
                  a COEXISTING cusp pair, so the ladder height of the network
                  is (max_count - 1) // 2.  This is the measure to quote.
      pairs       the number of times the count stepped up by two.  The cusp
                  count is NOT monotonic in kappa -- pairs annihilate and
                  re-form -- so this OVERSTATES the height and can manufacture
                  apparent violations of the P-1 bound.  Kept only to report
                  the size of that artifact.

    `nonmono` records whether the count ever fell, which is what makes the two
    differ.
    """
    lo, hi = 0.5, w.max() + 0.6
    ks = np.linspace(0.0, 1.5, NK)
    prev, _, _, _ = cusp_freqs(K0, ze, N, lo, hi)
    recs, events, pairs = [], [], 0
    max_count, nonmono = len(prev), False
    for k in ks[1:]:
        cur, wm, zm, res = cusp_freqs(K0 + k * L, ze, N, lo, hi)
        max_count = max(max_count, len(cur))
        if len(cur) < len(prev):
            nonmono = True
        if len(cur) > len(prev):
            size = len(cur) - len(prev)
            # Same guard as ensemble_unified: a cusp that MOVED more than DEDUP
            # in one step would otherwise be miscounted as new. The count jump
            # is the reliable quantity, so keep exactly `size` candidates.
            dist = sorted(((min([abs(o - q) for q in prev], default=9.0), o)
                           for o in cur), reverse=True)
            new = [o for d, o in dist[:size] if d > DEDUP]
            events.append(size)
            if size == 2:
                pairs += 1
            for Om in new:
                h = 1e-4
                aO = 0.5 * (f_of(Om + h, wm, zm, res)
                            - 2 * f_of(Om, wm, zm, res)
                            + f_of(Om - h, wm, zm, res)) / h ** 2
                act = int(np.argmin(np.abs(Om - wm)))
                Ga = np.sum([res[m] / (wm[m] ** 2 - Om ** 2
                                       + 2j * zm[m] * Om)
                             for m in range(N) if m != act])
                recs.append(dict(N=N, kappa=float(k), Om=float(Om),
                                 aO=float(aO), reGa=float(Ga.real),
                                 imGa=float(Ga.imag),
                                 psi=abs(Ga.imag) / abs(Ga.real),
                                 ok=bool(np.sign(aO) == -np.sign(Ga.real))))
        prev = cur
    return recs, events, pairs, (max_count - 1) // 2, nonmono


def main():
    rng = np.random.default_rng(SEED)
    recs, n_nets, n_over, n_dark, n_events = [], 0, 0, 0, []
    n_over_ev = n_full = n_nonmono = 0
    per_N = {}
    for N in NS:
        stats = dict(nets=0, over=0, over_ev=0, dark=0, cusps=0, ok=0,
                     full=0, nonmono=0)
        for _ in range(N_PER_N):
            w, ze, K0, L = random_network(rng, N)
            ks = np.linspace(0.0, 1.5, NK)
            P = visible(K0, L, ks[1:])
            r, ev, pairs, rungs, nonmono = born_cusps(N, w, ze, K0, L)
            recs += r
            n_events += ev
            n_nets += 1
            stats["nets"] += 1
            stats["dark"] += P < N
            n_dark += P < N
            # The bound is on COEXISTING pairs; `pairs` is the event tally and
            # is reported only to size the artifact it produces.
            over = rungs > P - 1
            over_ev = pairs > P - 1
            stats["over"] += over
            stats["over_ev"] += over_ev
            n_over += over
            n_over_ev += over_ev
            stats["full"] += rungs == P - 1
            n_full += rungs == P - 1
            stats["nonmono"] += nonmono
            n_nonmono += nonmono
            stats["cusps"] += len(r)
            stats["ok"] += sum(x["ok"] for x in r)
        per_N[N] = stats
        print(f"  N={N}: {stats['nets']} nets, {stats['dark']} with a dark mode, "
              f"{stats['over']} exceeding P-1 rungs "
              f"({stats['over_ev']} by the event tally), "
              f"{stats['full']} realizing the full ladder, "
              f"{stats['nonmono']} non-monotonic, {stats['cusps']} born cusps, "
              f"rule holds at {stats['ok']}/{stats['cusps']}", flush=True)

    out = pathlib.Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    with (out / "ensemble_validity_boundary.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)
    print(f"\n  wrote data/ensemble_validity_boundary.csv ({len(recs)} rows)")

    psi = np.array([r["psi"] for r in recs])
    ok = np.array([r["ok"] for r in recs])
    reGa = np.abs([r["reGa"] for r in recs])

    print("\n" + "=" * 68)
    print(f"ENSEMBLE: {n_nets} networks, N = {min(NS)}..{max(NS)}, seed {SEED}")
    print("=" * 68)
    sizes = {}
    for e in n_events:
        sizes[e] = sizes.get(e, 0) + 1
    print(f"  birth events: {len(n_events)}   size distribution: "
          + ", ".join(f"+{s}x{c}" for s, c in sorted(sizes.items())))
    print(f"  born cusps:   {len(recs)}")
    print(f"  networks with a dark mode (P < N): {n_dark}/{n_nets}")
    print(f"  criterion holds overall:           {ok.sum()}/{len(ok)} "
          f"({100*ok.mean():.1f}%)")

    print("\n  LADDER HEIGHT  (a rung is a COEXISTING cusp pair)")
    print(f"  networks exceeding P-1 rungs:      {n_over}/{n_nets}")
    print(f"  full (P-1)-rung ladder realized:   {n_full}/{n_nets}")
    print(f"  cusp count non-monotonic in kappa: {n_nonmono}/{n_nets}")
    print(f"  -- counting increase events instead would report "
          f"{n_over_ev}/{n_nets} exceeding the bound, which is the artifact of "
          f"that non-monotonicity, not a violation.")

    print("\n  ACCURACY OF (typecrit) vs BACKGROUND PHASE")
    print(f"  {'psi = |Im Ga|/|Re Ga|':>26} {'n':>6} {'holds':>7} {'accuracy':>10}")
    for a, b in zip(PSI_EDGES[:-1], PSI_EDGES[1:]):
        m = (psi >= a) & (psi < b)
        if not m.any():
            continue
        lab = f"[{a:.2f}, {b:.2f})" if np.isfinite(b) else f"[{a:.2f}, inf)"
        print(f"  {lab:>26} {m.sum():6d} {ok[m].sum():7d} "
              f"{100*ok[m].mean():9.1f}%")

    print("\n  WHICH DROPPED TERM?  (typecrit keeps only Re G_a)")
    imGa = np.array([r["imGa"] for r in recs])
    reGa_s = np.array([r["reGa"] for r in recs])
    # Keeping Im G_a in the derivation of (typecrit) replaces Re G_a by
    # Re G_a - sqrt3 Im G_a; that is the ONLY effect of a constant complex
    # background, so it flips the verdict exactly when psi > 1/sqrt3 with
    # Re and Im of the same sign.  Failures it does not repair belong to the
    # other approximation, a background held constant across the pole.
    ok2 = np.sign([r["aO"] for r in recs]) == -np.sign(reGa_s
                                                      - np.sqrt(3) * imGa)
    hi = psi > 1 / np.sqrt(3)
    print(f"  sign a = -sign Re G_a               : {ok.sum()}/{len(ok)}")
    print(f"  sign a = -sign(Re G_a - sqrt3 Im G_a): {ok2.sum()}/{len(ok2)}")
    print(f"  cusps with psi > 1/sqrt3 = {1/np.sqrt(3):.3f}: {hi.sum()}, "
          f"of which the stated rule fails {(hi & ~ok).sum()} and the "
          f"kept-Im form fails {(hi & ~ok2).sum()}")
    print(f"  failures with psi <= 1/sqrt3 (constant-background term): "
          f"{(~ok & ~hi).sum()}, psi in "
          f"[{psi[~ok & ~hi].min():.2f}, {psi[~ok & ~hi].max():.2f}]")

    print("\n  IS IT PHASE, OR A MARGINAL Re G_a?")
    bad = ~ok
    if bad.any():
        print(f"  failures: {bad.sum()}")
        print(f"    psi   at failures: median {np.median(psi[bad]):.3f}   "
              f"vs {np.median(psi):.3f} over all cusps")
        pct = [100 * (reGa < v).mean() for v in reGa[bad]]
        print(f"    |Re Ga| at failures sits at percentile "
              f"{np.min(pct):.0f}-{np.max(pct):.0f} of the ensemble "
              f"(marginality would push this toward 0)")
        print(f"    median |Re Ga| ratio failures/all: "
              f"{np.median(reGa[bad])/np.median(reGa):.2f}")
    else:
        print("  no failures")


if __name__ == "__main__":
    main()
