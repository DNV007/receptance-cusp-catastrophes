"""Sensitivity of the 100-network ensemble result to seed, coupling grid, and
cusp-association tolerance.

`ensemble_validity_boundary.py` reports the long paper's ensemble numbers on
ONE draw: seed 20260811, 300 coupling values on [0, 1.5], and a frequency
tolerance DEDUP = 5e-3 for calling a root on the next step "the same cusp".
A referee is entitled to ask whether the two headline statements

    (i)  no realization exceeds P-1 coexisting additional pairs,
    (ii) sign a_Omega ~= -sign Re G_a(Omega) holds at ~97% of born cusps,

are properties of the mechanism or of those three numerical choices.  This
script re-runs the same ensemble construction under

    baseline    seed 20260811, NK = 300, DEDUP = 5e-3   (reproduces the paper)
    seed        seed 20260908, NK = 300, DEDUP = 5e-3
    grid        seed 20260811, NK = 600, DEDUP = 5e-3
    assoc       seed 20260811, NK = 300, DEDUP = 2e-3

and prints the same summary for each.  Only the counting bound and the rule
accuracy are meant to be stable; the raw event and cusp tallies are not, since
a denser grid resolves births that a coarse one lumps together.

Result (2026-09-08), quoted in the long paper's Appendix on the ensemble:

    config      seed   NK   DEDUP | >P-1  full  events  cusps       rule
  baseline  20260811  300   5e-03 |    0    23     236    474  462/474  97.5%
      seed  20260908  300   5e-03 |    0    31     243    492  489/492  99.4%
      grid  20260811  600   5e-03 |    0    23     236    472  464/472  98.3%
     assoc  20260811  300   2e-03 |    0    23     236    474  462/474  97.5%

The P-1 bound is never violated and the rule accuracy stays in 97.5-99.4%.
Event and cusp tallies shift with the grid, which is expected: a denser grid
resolves births that a coarser one lumps together.

Run:  JAX not required;  .venv/bin/python scripts/ensemble_seed_grid_sensitivity.py
Cost: ~25 min for all four configurations on one core.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ensemble_validity_boundary as evb
from ensemble_unified import random_network

CONFIGS = (
    ("baseline", 20260811, 300, 5e-3),
    ("seed",     20260908, 300, 5e-3),
    ("grid",     20260811, 600, 5e-3),
    ("assoc",    20260811, 300, 2e-3),
)


def run(seed, nk, dedup):
    evb.NK, evb.DEDUP = nk, dedup
    rng = np.random.default_rng(seed)
    nets = over = full = dark = nonmono = cusps = ok = 0
    events = []
    for N in evb.NS:
        for _ in range(evb.N_PER_N):
            w, ze, K0, L = random_network(rng, N)
            ks = np.linspace(0.0, 1.5, nk)
            P = evb.visible(K0, L, ks[1:])
            r, ev, _pairs, rungs, nm = evb.born_cusps(N, w, ze, K0, L)
            nets += 1
            dark += P < N
            over += rungs > P - 1
            full += rungs == P - 1
            nonmono += nm
            cusps += len(r)
            ok += sum(x["ok"] for x in r)
            events += ev
    return dict(nets=nets, dark=dark, over=over, full=full, nonmono=nonmono,
                events=len(events), cusps=cusps, ok=ok)


def main():
    print(f"{'config':>9} {'seed':>9} {'NK':>5} {'DEDUP':>7} | {'nets':>5} "
          f"{'dark':>5} {'>P-1':>5} {'full':>5} {'nonmono':>8} {'events':>7} "
          f"{'cusps':>6} {'rule':>12}")
    for name, seed, nk, dedup in CONFIGS:
        s = run(seed, nk, dedup)
        acc = 100 * s["ok"] / s["cusps"] if s["cusps"] else float("nan")
        print(f"{name:>9} {seed:>9} {nk:>5} {dedup:>7.0e} | {s['nets']:>5} "
              f"{s['dark']:>5} {s['over']:>5} {s['full']:>5} {s['nonmono']:>8} "
              f"{s['events']:>7} {s['cusps']:>6} "
              f"{s['ok']:>4}/{s['cusps']:<4} {acc:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
