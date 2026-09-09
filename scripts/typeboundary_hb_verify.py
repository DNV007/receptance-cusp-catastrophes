"""Full-HB test of the beaks/lips classifier near the type boundary.

Away from the boundary the classifier is not under stress: at omega_2 = 1.25
the benchmark birth has a_Omega = +1315.  The fragile place is the fold of the
birth locus, where a_Omega passes through zero and the two topologies coexist
over Om* in [1.061, 1.074].  This script tests the classifier there.

At omega_2 = 1.06 the family carries two births with OPPOSITE predicted type
and |a_Omega| roughly thirty times smaller than the benchmark:

    beaks  Om* = 1.078504  kappa* = 0.0410008  a_Omega = +40.03
    lips   Om* = 1.061086  kappa* = 0.0592217  a_Omega = -18.59

both at eps ~ 0.1 and eta_3 ~ 3e-3, i.e. deep inside the regime where the
fundamental-harmonic organizer is quantitative -- so a failure here would be a
failure of the CLASSIFIER, not of the first-harmonic reduction.  That
separation is the point of choosing this operating point.

Predicted from the linear receptance alone (physical cusps, arg G = -150 deg):

    kappa = 0.036001  ->  1 cusp                              (before beaks)
    kappa = 0.046001  ->  3 cusps: 1.04997, 1.07028, 1.09093  (pair born)
    kappa = 0.054222  ->  3 cusps: 1.05512, 1.06583, 1.10167  (lips oval up)
    kappa = 0.064222  ->  1 cusp                              (oval gone)

What this CAN decide: whether the classifier assigns the correct topology on
both sides of a boundary it places inside a window narrower than full-HB
resolution.  What it CANNOT decide, and what is therefore not claimed: whether
the third-harmonic correction displaces that boundary.  The End Matter puts
the flip window at ~2e-6 in kappa, far under the ~1e-6 bisection floor and the
~1e-3 frequency scan step.

Result (2026-09-08), quoted in the Letter's End Matter:

    kappa = 0.036001  1 band  [1.0460, 1.1100]              connected
    kappa = 0.046001  2 bands [1.0500, 1.0700] [1.0920, 1.1100]   GAP OPENED
    kappa = 0.054222  1 band  [1.0550, 1.0650]              isolated lens
    kappa = 0.064222  no bistability in the window          lens gone

Both predicted topologies are reproduced, and the band edges match the cusps
predicted from the linear receptance alone to within the scan step (2.0e-3 for
the beaks window, 1.25e-3 for the lips window):

    beaks  predicted 1.07028 / 1.09093   measured 1.0700 / 1.0920
    lips   predicted 1.05512 / 1.06583   measured 1.0550 / 1.0650

So the classifier is correct on both sides of the boundary at |a_Omega| = 40
and 18.6, thirty to seventy times below the benchmark's +1315.

Run:  .venv/bin/python scripts/typeboundary_hb_verify.py
Cost: ~35 min on one core at NH = 5.
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
W2 = 1.06

CASES = [
    dict(tag="beaks", Om_t=1.078504, kap_star=0.0410008, a_Om=+40.033,
         kappas=(0.036001, 0.046001), om=(1.030, 1.110), n=41),
    dict(tag="lips", Om_t=1.061086, kap_star=0.0592217, a_Om=-18.589,
         kappas=(0.054222, 0.064222), om=(1.035, 1.085), n=41),
]


def K_of(kappa, w2=W2):
    return np.array([[W1 ** 2 + kappa, -kappa], [-kappa, w2 ** 2 + kappa]])


def scan(S, kappa, lo, hi, n):
    out = []
    for om in np.linspace(lo, hi, n):
        w = tongue_width(S, K_of(kappa), float(om), F_max=3.0)
        out.append((float(om), float(w) if np.isfinite(w) else float("nan")))
    return out


def structure(rows):
    live = [np.isfinite(w) and w > 0 for _, w in rows]
    if not any(live):
        return "no bistability in the window", []
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
    desc = f"{len(runs)} bistable band(s): " + ", ".join(
        f"[{a:.4f}, {b:.4f}]" for a, b in runs)
    return desc, runs


def main():
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    lines = []
    for c in CASES:
        print("=" * 72)
        print(f"{c['tag'].upper()}  omega_2={W2}  kappa*={c['kap_star']:.7f}  "
              f"a_Omega={c['a_Om']:+.2f}  Om*={c['Om_t']:.6f}")
        print("=" * 72, flush=True)
        for kap in c["kappas"]:
            rows = scan(S, kap, c["om"][0], c["om"][1], c["n"])
            desc, runs = structure(rows)
            side = "below" if kap < c["kap_star"] else "above"
            print(f"  kappa={kap:.6f} ({side} kappa*): {desc}", flush=True)
            for o, w in rows:
                lines.append(f"{c['tag']},{W2},{kap:.6f},{side},{o:.6f},{w:.6e}")
    out = os.path.join(DATA, "typeboundary_hb_verify.csv")
    with open(out, "w") as fh:
        fh.write("case,w2,kappa,side,Omega,tongue_width\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
