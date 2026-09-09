"""Convergence of the directly located projection fold kappa*_HB.

The bisection detects a narrowing frequency gap on a discrete grid, so the
quoted bracket of 1.1e-5 is the width to which the binary test was resolved,
not by itself evidence that the answer is insensitive to how the test is set
up.  Since the figure appears in the abstract, that insensitivity is
demonstrated here rather than estimated.

Four knobs are varied independently about a baseline:

    n_om   frequency samples across the scan window (gap detection threshold)
    pad    scan-window width as a fraction of the closed-form pair separation
    ds     arclength step of the continuation that supplies the fold count
    nh     harmonic truncation

Each determination is warm-started from a bracket around the known answer, so
five bisection steps suffice to reach 1e-5.

Output: data/projfold_convergence.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, F_folds, K_2dof  # noqa: E402
from recompute_beaks_table import closed_form_cusps, Fc_of, Z1, Z2, BETA1  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")


def gap_width(S, kappa, n_om, pad, ds):
    cf = [c for c in closed_form_cusps(kappa) if c > 1.2]
    if len(cf) >= 2:
        lo_cf, hi_cf = cf[0], cf[1]
        w = max(hi_cf - lo_cf, 4e-3)
        lo, hi = lo_cf - pad * w, hi_cf + pad * w
        F_max = 3.0 * max(Fc_of(lo_cf, kappa), Fc_of(hi_cf, kappa))
    else:
        c0 = 1.3010
        lo, hi = c0 - 0.016, c0 + 0.016
        F_max = 3.0 * max(Fc_of(c0, kappa) or 1.0, 1.0)
    oms = np.linspace(lo, hi, n_om)
    bist = np.array([len(F_folds(S, K_2dof(kappa), float(o), ds=ds,
                                 F_max=F_max)) >= 2 for o in oms])
    if not bist.any():
        return 0.0
    idx = np.where(bist)[0]
    inner = bist[idx[0]:idx[-1] + 1]
    if inner.all():
        return 0.0
    runs, cur = [], []
    for j, b in enumerate(inner):
        if not b:
            cur.append(j)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return 0.0
    return (len(max(runs, key=len)) + 1) * (oms[1] - oms[0])


def locate(n_om=60, pad=0.35, ds=0.01, nh=3,
           k_lo=0.11920, k_hi=0.11940, tol=1e-5):
    S = make_solver(2, nh, 256, [Z1, Z2], BETA1)
    if gap_width(S, k_lo, n_om, pad, ds) > 0:
        k_lo -= 0.0004
    if gap_width(S, k_hi, n_om, pad, ds) == 0:
        k_hi += 0.0004
    while k_hi - k_lo > tol:
        m = 0.5 * (k_lo + k_hi)
        if gap_width(S, m, n_om, pad, ds) > 0:
            k_hi = m
        else:
            k_lo = m
    return 0.5 * (k_lo + k_hi), k_hi - k_lo


BASE = dict(n_om=60, pad=0.35, ds=0.01, nh=3)
CASES = [("baseline", {}),
         ("n_om 40", dict(n_om=40)),
         ("n_om 90", dict(n_om=90)),
         ("pad 0.25", dict(pad=0.25)),
         ("pad 0.50", dict(pad=0.50)),
         ("ds 0.005", dict(ds=0.005)),
         ("N_HARM 4", dict(nh=4))]

if __name__ == "__main__":
    rows = []
    ref = None
    print(f"{'case':>10s} {'kappa*_HB':>11s} {'bracket':>9s} "
          f"{'vs baseline':>12s} {'wall':>7s}")
    for name, over in CASES:
        kw = dict(BASE); kw.update(over)
        t0 = time.time()
        k, br = locate(**kw)
        if ref is None:
            ref = k
        rows.append(dict(case=name, **kw, kstar=k, bracket=br, dev=k - ref))
        print(f"{name:>10s} {k:11.6f} {br:9.1e} {k-ref:+12.1e} "
              f"{time.time()-t0:6.0f}s", flush=True)
        out = os.path.join(DATA, "projfold_convergence.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    sp = max(r["kstar"] for r in rows) - min(r["kstar"] for r in rows)
    print(f"\nspread across all settings: {sp:.1e}")
