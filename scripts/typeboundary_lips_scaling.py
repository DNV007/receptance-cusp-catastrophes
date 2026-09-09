"""Lips birth at the type boundary: isolation, orientation, and the sqrt(mu) law.

Follow-up to typeboundary_hb_verify.py, which had two defects at the lips point
(omega_2 = 1.06, kappa* = 0.0592217, a_Omega = -18.59):

 1. Its window [1.035, 1.085] stopped below the third physical cusp at 1.10167,
    so it never showed the lens was SEPARATED from the upper band -- "isolated
    lens" was asserted on a truncated scan.  Here the window runs to 1.130.
 2. The lens is present BELOW kappa* and gone above, the opposite orientation
    to the omega_2 = 0.95 lips case of Fig. 2b.  That is the sign of b_kappa,
    not a contradiction, and the object that vanishes is continuous with the
    lower sub-wedge left by the beaks birth at kappa = 0.0410 -- so it is a
    lips-type ANNIHILATION, not a birth "from an empty neighbourhood".

The upgrade: the classifier's own scaling law, tested where it is weakest.
Across a lips point the pair separation obeys Delta Omega = c* sqrt(mu) with
mu = kappa* - kappa here.  Measuring it at |a_Omega| = 18.6, seventy times
below the benchmark's +1315, tests far more than presence/absence.

Result (2026-09-08):

  ISOLATION, wide window at kappa = 0.054222
    predicted physical cusps  1.05512, 1.06583, 1.10167
    full HB   2 bands  [1.0568, 1.0647]  [1.1023, 1.1300]
    -> the lens IS separated; the earlier "isolated" claim now has data.

  sqrt(mu) LAW           mu      pred dOm   HB dOm   c*_pred   c*_HB
                    5.0e-03      0.01071   0.01018    0.1515   0.1439
                    2.0e-03      0.00648   0.00677    0.1449   0.1514
                    1.0e-03      0.00452   0.00515    0.1428   0.1628
                    5.0e-04      0.00317   0.00422    0.1418   0.1886

c*_pred is stable to 7% across the range; c*_HB agrees to 5% at the two larger
separations then drifts up 30% as mu falls.  That drift is the TIP-DETECTION
BIAS -- the scan reports the last frequency at which the continuation still
resolves two folds, an outward bound on the tip, and a fixed absolute bias is
a growing fraction of a shrinking lens.  Do NOT report this as a clean
verification of the law: it SUPERSEDED by typeboundary_lips_threshold.py: the
residual changes sign and grows 3.6x, which no fixed estimator bias produces.
It is a displaced THRESHOLD -- fit c*sqrt(mu+delta), c=0.1381, delta=4.34e-4,
so the HB oval closes at kappa*_pred + delta = 0.05966, confirmed directly to
lie in [0.0595, 0.0597].

b_kappa at the four relevant points (sign selects which side the folds are on):
    Fig. 2b lips  w2=0.95  a_Om = -94.2   b_kap = +1.315  -> oval ABOVE kappa*
    near-bdy lips w2=1.06  a_Om = -18.6   b_kap = -0.092  -> oval BELOW kappa*
    near-bdy beaks w2=1.06 a_Om = +40.0   b_kap = -0.796
    benchmark      w2=1.25 a_Om = +1315   b_kap = -8.689

Run:  .venv/bin/python scripts/typeboundary_lips_scaling.py
Cost: ~40 min on one core at NH = 5.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cusp_tip_intercept import make_solver, tongue_width

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
BETA1 = 0.147
NH, NT = 5, 256
W2, KSTAR = 1.06, 0.0592217
GAM = np.array([Z1, Z2])


def K_of(kappa):
    return np.array([[W1 ** 2 + kappa, -kappa], [-kappa, W2 ** 2 + kappa]])


def G(Om, kap):
    Z = np.array([[1.0 - Om ** 2 + 2j * GAM[0] * Om + kap, -kap],
                  [-kap, W2 ** 2 - Om ** 2 + 2j * GAM[1] * Om + kap]], complex)
    return np.linalg.inv(Z)[0, 0]


def f(Om, kap):
    inv = 1.0 / G(Om, kap)
    return inv.real ** 2 - 3 * inv.imag ** 2


def predicted_cusps(kap, lo=1.00, hi=1.13, n=6000):
    oms = np.linspace(lo, hi, n)
    fv = np.array([f(o, kap) for o in oms])
    r = [brentq(lambda x: f(x, kap), oms[i], oms[i + 1], xtol=1e-13)
         for i in range(n - 1) if fv[i] * fv[i + 1] < 0]
    return [x for x in r if (1.0 / G(x, kap)).real < 0]


def bands(S, kappa, lo, hi, n):
    rows = []
    for om in np.linspace(lo, hi, n):
        w = tongue_width(S, K_of(kappa), float(om), F_max=3.0)
        rows.append((float(om), float(w) if np.isfinite(w) else float("nan")))
    live = [np.isfinite(w) and w > 0 for _, w in rows]
    runs, i = [], 0
    while i < len(live):
        if live[i]:
            j = i
            while j < len(live) and live[j]:
                j += 1
            runs.append((rows[i][0], rows[j - 1][0]))
            i = j
        else:
            i += 1
    return rows, runs


def main():
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    lines = []

    print("=" * 72)
    print("1. ISOLATION -- wide window at kappa = 0.054222 (mu = 5.0e-3)")
    print("=" * 72, flush=True)
    cf = predicted_cusps(0.054222)
    print(f"   predicted physical cusps: " + ", ".join(f"{c:.5f}" for c in cf))
    rows, runs = bands(S, 0.054222, 1.035, 1.130, 49)
    print(f"   full HB: {len(runs)} band(s): "
          + ", ".join(f"[{a:.4f}, {b:.4f}]" for a, b in runs), flush=True)
    for o, w in rows:
        lines.append(f"isolation,0.054222,{o:.6f},{w:.6e}")

    print()
    print("=" * 72)
    print("2. sqrt(mu) LAW across the lips point,  a_Omega = -18.59")
    print("=" * 72)
    print(f"   {'mu':>9} {'kappa':>10} {'pred dOm':>10} {'HB dOm':>10} "
          f"{'c*_pred':>9} {'c*_HB':>9}", flush=True)
    for mu in (5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4):
        kap = KSTAR - mu
        cf = predicted_cusps(kap)
        pair = [c for c in cf if 1.03 < c < 1.09]
        if len(pair) < 2:
            print(f"   mu={mu:.1e}: no predicted pair in window ({cf})")
            continue
        p_lo, p_hi = min(pair), max(pair)
        d_pred = p_hi - p_lo
        pad = 0.45 * d_pred
        rows, runs = bands(S, kap, p_lo - pad, p_hi + pad, 41)
        for o, w in rows:
            lines.append(f"scaling_{mu:.0e},{kap:.6f},{o:.6f},{w:.6e}")
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            d_hb = b - a
            print(f"   {mu:9.1e} {kap:10.6f} {d_pred:10.5f} {d_hb:10.5f} "
                  f"{d_pred/np.sqrt(mu):9.4f} {d_hb/np.sqrt(mu):9.4f}", flush=True)
        else:
            print(f"   {mu:9.1e} {kap:10.6f} {d_pred:10.5f} {'none':>10}",
                  flush=True)

    out = os.path.join(DATA, "typeboundary_lips_scaling.csv")
    with open(out, "w") as fh:
        fh.write("case,kappa,Omega,tongue_width\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
