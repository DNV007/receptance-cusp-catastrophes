"""Is there a device parameter set where the beaks is actually observable?

Three requirements pull against one another, and all three are governed by the
same quantity, the auxiliary detuning D2 at the birth:

  resolvability   the cusp splitting must exceed the auxiliary linewidth k2;
  accessibility   both born sub-wedges must contain stable coexistence;
  cleanliness     n2/n1 = J^2/|D2|^2 sets how far the coupler cross-Kerr is
                  amplified, hence the tolerance chi/|K| that must be met.

Raising k2 damps the auxiliary mode and should help accessibility, but it
directly harms resolvability and raises J*.  Increasing the mode separation
moves the birth but also changes |D2|.  This scans (k2, f2-f1) and reports all
three quantities so the trade-off can be read off rather than argued.

Output: data/kerr_tradeoff.csv
"""
from __future__ import annotations

import csv
import os

import numpy as np

import kerr_inverse_design as kid
import kerr_stability_map as ksm

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
NSCAN = 8000          # frequency samples inside cusps(); 80001 is far finer
                      # than this search needs and dominates the runtime


def usable_fraction(J, df, pair, side, n_om=10, n_F=9):
    w = pair[1] - pair[0]
    if side < 0:
        lo, hi = pair[0] - 0.5 * w, pair[0] - 0.03 * w
    else:
        lo, hi = pair[1] + 0.03 * w, pair[1] + 0.5 * w
    rows = []
    for fd in np.linspace(lo, hi, n_om):
        f = ksm.folds_at(fd, J, df)
        if f is None:
            continue
        F1, F2 = f
        for t in np.linspace(0.08, 0.92, n_F):
            code, _ = ksm.classify_point(fd, J, F1 + t * (F2 - F1), df)
            rows.append(code)
    bis = [c for c in rows if c >= 0]
    if not bis:
        return float("nan")
    return len([c for c in bis if c == 1]) / len(bis)


def evaluate(k2, df, J_mult=2.0):
    kid.K2 = k2                       # module-level linewidth used by rho_sigma
    ksm.DF = df
    Js = kid.birth_J(df, n=NSCAN)
    if Js is None:
        return None
    J = J_mult * Js
    pair = kid.pair_near(J, df, n=NSCAN)
    if pair is None:
        return None
    split = pair[1] - pair[0]
    fd_b = 0.5 * pair[0] + 0.5 * pair[1]
    D2 = df - fd_b + 1j * k2 / 2
    ratio = J**2 / abs(D2) ** 2
    coef = J**2 * (1 / abs(D2) ** 2 + 1 / D2**2)
    tol = 0.05 * abs(kid.K_KERR) / abs(coef.imag) / abs(kid.K_KERR)
    n1 = kid.occupation(fd_b, J, df)
    return dict(k2=k2, df=df, Jstar=Js, J=J, split=split,
                split_over_k2=split / k2,
                low=usable_fraction(J, df, pair, -1),
                high=usable_fraction(J, df, pair, +1),
                n1=n1, n2_over_n1=ratio, chi_tol=tol)


if __name__ == "__main__":
    rows = []
    print(f"{'k2':>6s} {'df':>7s} {'J*':>7s} {'split':>7s} {'split/k2':>9s} "
          f"{'lower':>7s} {'upper':>7s} {'n2/n1':>8s} {'chi_tol':>9s}")
    for k2 in (0.25, 0.5, 1.0, 2.0, 4.0):
        for df in (-20.0, -35.0, -50.0, -80.0):
            try:
                r = evaluate(k2, df)
            except Exception:
                r = None
            if r is None:
                print(f"{k2:6.2f} {df:7.1f}   no clean birth")
                continue
            rows.append(r)
            print(f"{k2:6.2f} {df:7.1f} {r['Jstar']:7.3f} {r['split']:7.3f} "
                  f"{r['split_over_k2']:9.2f} "
                  f"{100*r['low']:6.0f}% {100*r['high']:6.0f}% "
                  f"{r['n2_over_n1']:8.1f} {r['chi_tol']:9.1e}", flush=True)
    out = os.path.join(DATA, "kerr_tradeoff.csv")
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote", out)
