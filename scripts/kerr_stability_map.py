"""Where in the born sectors are both outer branches actually stable?

Existence of three steady states is not coexistence of two stable ones.  This
maps the drive plane on each side of the born cusp pair and asks, at every
point, whether the reduction admits three states AND the outer two are
dynamically stable in the rotating frame.

Reports, for each sub-wedge, the fraction of the bistable area that is
usable, and the character of the instability where it is not.

Output: data/kerr_stability_map.csv
"""
from __future__ import annotations

import csv
import os

import numpy as np

import kerr_inverse_design as kid
import kerr_stability as ks

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
DF = -50.0
K = kid.K_KERR


def folds_at(fd, J, df=DF):
    rho, sig = kid.rho_sigma(fd, J, df)
    disc = rho * rho - 3 * sig * sig
    if disc <= 0 or K * rho >= 0:
        return None
    sd = np.sqrt(disc)
    ys = [(-2 * rho + sd) / (3 * K), (-2 * rho - sd) / (3 * K)]
    if min(ys) <= 0:
        return None
    Fs = sorted(np.sqrt(y * ((rho + K * y) ** 2 + sig ** 2)) for y in ys)
    return Fs[0], Fs[1]


def classify_point(fd, J, F, df=DF):
    """(-1) not bistable, (0) bistable but an outer branch unstable, (1) usable."""
    b = ks.branches(fd, J, F, df)
    if len(b) < 3:
        return -1, None
    ev_lo = np.linalg.eigvals(ks.jacobian(b[0][1], fd, J, F, df))
    ev_hi = np.linalg.eigvals(ks.jacobian(b[2][1], fd, J, F, df))
    lo_ok = ev_lo.real.max() < 0
    hi_ok = ev_hi.real.max() < 0
    if lo_ok and hi_ok:
        return 1, None
    bad = ev_lo if not lo_ok else ev_hi
    k = int(np.argmax(bad.real))
    return 0, ("Hopf" if abs(bad[k].imag) > 1e-6 else "real")


def scan(J, lo, hi, n_om=26, n_F=22):
    rows = []
    for fd in np.linspace(lo, hi, n_om):
        f = folds_at(fd, J)
        if f is None:
            continue
        F1, F2 = f
        for t in np.linspace(0.06, 0.94, n_F):
            F = F1 + t * (F2 - F1)
            code, kind = classify_point(fd, J, F)
            rows.append(dict(J=J, fd=fd, F=F, code=code, kind=kind or ""))
    return rows


if __name__ == "__main__":
    J = 5.0
    pair = kid.pair_near(J, DF)
    print(f"J = {J} MHz, born pair at {pair[0]:.4f} / {pair[1]:.4f} MHz\n")

    allrows = []
    for tag, lo, hi in (("lower sub-wedge", pair[0] - 0.60, pair[0] - 0.02),
                        ("upper sub-wedge", pair[1] + 0.02, pair[1] + 0.60)):
        rows = scan(J, lo, hi)
        allrows += rows
        bis = [r for r in rows if r["code"] >= 0]
        good = [r for r in rows if r["code"] == 1]
        kinds = {}
        for r in rows:
            if r["code"] == 0:
                kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        frac = 100.0 * len(good) / max(1, len(bis))
        print(f"{tag}: {len(bis)} bistable samples, {len(good)} with both "
              f"outer branches stable ({frac:.0f}%)")
        if kinds:
            print(f"    instability character: {kinds}")

    out = os.path.join(DATA, "kerr_stability_map.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["J", "fd", "F", "code", "kind"])
        w.writeheader()
        for r in allrows:
            w.writerow(r)
    print("\nwrote", out)
