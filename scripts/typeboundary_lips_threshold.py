"""Where does the full-HB lips oval actually close?

The sqrt(mu) fit of typeboundary_lips_scaling.py gives
Delta Omega_HB = c sqrt(mu + delta) with c = 0.1381, delta = 4.34e-4 and
mu = kappa*_pred - kappa, kappa*_pred = 0.0592217.  The HB width therefore
vanishes at kappa*_pred + delta = 0.059655, ABOVE the first-harmonic
prediction: the organizer places this birth 0.73% low in kappa.

This is a direct test of that sign.  If kappa*_HB > kappa*_pred the oval must
still be present at and above kappa*_pred, where the first-harmonic receptance
predicts no cusps at all.  If instead kappa*_HB = kappa*_pred - delta, the oval
must already be gone by 0.058788 -- but it is observed 0.00422 wide at
0.058722, four times the 1.1e-3 that threshold would allow.

Scanned on one fixed window so widths are directly comparable.

Result (2026-09-08), window [1.050, 1.072], 61 points, step 3.67e-4:

    kappa       HB band            width     fit c*sqrt(mu+delta)
    0.0592217   [1.0599, 1.0625]   0.00257   0.00288
    0.0595      [1.0606, 1.0617]   0.00110   0.00172
    0.0597      none               0         0
    0.0599      none               0         0
    0.0601      none               0         0

The oval closes in [0.0595, 0.0597], bracketing the fitted 0.059655.  It is
still present AT kappa*_pred = 0.0592217, where the first-harmonic receptance
predicts no cusps at all.  So kappa*_HB > kappa*_pred: the organizer places
this near-boundary birth about 0.7% LOW in kappa, against 0.30% at the
benchmark.  The alternative sign (kappa*_HB = kappa*_pred - delta = 0.05879)
is refuted at four separate couplings.

Location degrades while the local topology AND its sqrt(mu) separation law
survive, at seventy times smaller curvature -- the paper's own thesis,
measured where it is weakest.

Run:  .venv/bin/python scripts/typeboundary_lips_threshold.py
Cost: ~25 min on one core at NH = 5.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cusp_tip_intercept import make_solver, tongue_width

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
BETA1 = 0.147
NH, NT = 5, 256
W2, KPRED = 1.06, 0.0592217
C_FIT, DELTA_FIT = 0.1381, 4.3375e-4

KAPPAS = (0.0592217, 0.0595, 0.0597, 0.0599, 0.0601)
OM_LO, OM_HI, NOM = 1.050, 1.072, 61


def K_of(kappa):
    return np.array([[W1 ** 2 + kappa, -kappa], [-kappa, W2 ** 2 + kappa]])


def main():
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    step = (OM_HI - OM_LO) / (NOM - 1)
    print(f"window [{OM_LO}, {OM_HI}], {NOM} points, step {step:.2e}")
    print(f"{'kappa':>10} {'HB band':>22} {'width':>10} {'fit width':>10}")
    lines = []
    for kap in KAPPAS:
        rows = []
        for om in np.linspace(OM_LO, OM_HI, NOM):
            w = tongue_width(S, K_of(kap), float(om), F_max=3.0)
            rows.append((float(om), float(w) if np.isfinite(w) else float("nan")))
            lines.append(f"{kap:.6f},{om:.6f},{rows[-1][1]:.6e}")
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
        mu_eff = KPRED + DELTA_FIT - kap
        fit = C_FIT * np.sqrt(mu_eff) if mu_eff > 0 else 0.0
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            print(f"{kap:10.6f} {f'[{a:.4f}, {b:.4f}]':>22} {b-a:10.5f} "
                  f"{fit:10.5f}", flush=True)
        else:
            print(f"{kap:10.6f} {'none':>22} {0.0:10.5f} {fit:10.5f}", flush=True)
    out = os.path.join(DATA, "typeboundary_lips_threshold.csv")
    with open(out, "w") as fh:
        fh.write("kappa,Omega,tongue_width\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
