"""Full-HB validation of a lips birth at moderate anharmonicity.

The lips case in Sec. V.C uses omega_2 = 0.60, where epsilon ~ 0.7 and full HB
reproduces bistability only near the lower born cusp rather than across the
whole closed-form lens.  lips_moderate_eps.py shows epsilon falls monotonically
as the absorber moves toward the drive; omega_2 = 0.95 gives epsilon = 0.30 to
0.41, inside the stated validity window, while remaining a genuine lips
(a_Omega < 0, Re G_a > 0, no bistability below threshold).

This script checks, in the original equations of motion:
  1. below the predicted birth, no bistability anywhere near the lens
  2. above it, an isolated bistable region exists
  3. that region spans BOTH born cusps, i.e. the full lens is realized
     (the point on which the omega_2 = 0.60 case fails)
  4. the located cusp tips against the closed-form pair

For a lips the cusps face inward: bistability lives at Omega > Omega_- for the
left tip and Omega < Omega_+ for the right, so the intercept sides are the
reverse of the beaks case.

Output: data/lips_hb_moderate.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, cusp_tip, F_folds  # noqa: E402
from lips_moderate_eps import (  # noqa: E402
    cusps as cf_cusps, eps_of, Fc_of, find_birth, Z1, Z2, BETA1,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

W2 = 0.95
NH, NT = 7, 512          # eps ~ 0.4: keep the harmonics the manuscript uses for lips


def K_of(kappa, w2=W2):
    return np.array([[1.0 + kappa, -kappa], [-kappa, w2**2 + kappa]])


def scan_bistability(S, kappa, lo, hi, n=45, F_max=3.0):
    """Return the Omega values in [lo,hi] carrying >=2 folds."""
    hits = []
    for om in np.linspace(lo, hi, n):
        f = F_folds(S, K_of(kappa), float(om), ds=0.01, F_max=F_max)
        if len(f) >= 2:
            hits.append((float(om), f[0], f[-1]))
    return hits


if __name__ == "__main__":
    kstar, new, _ = find_birth(W2)
    new = sorted(new)[:2]
    (Om_l, rho_l), (Om_r, rho_r) = new
    print(f"closed form: kappa* = {kstar:.4f}, pair at "
          f"{Om_l:.5f} / {Om_r:.5f}")
    print(f"  eps = {eps_of(rho_l):.3f} / {eps_of(rho_r):.3f}, "
          f"F_c = {Fc_of(rho_l):.4f} / {Fc_of(rho_r):.4f}")

    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    rows = []

    # The three published panels of Fig. 3b.  These were originally derived as
    # kstar +/- offsets from the modal birth; they are pinned here so the
    # figure keeps its panels now that inv_G is the exact receptance.  They
    # still straddle the exact birth kappa* = 0.29976 as labelled:
    # 0.2632 < kappa*, 0.3032 just above, 0.3332 well above.
    PANELS = (0.26320802005012534, 0.30320802005012537, 0.33320802005012534)
    for kappa in PANELS:
        cs = [c for c in cf_cusps(kappa, W2) if 0.9 < c[0] < 1.15]
        if len(cs) >= 2:
            ol, orr = cs[0][0], cs[1][0]
            width = orr - ol
            lo, hi = ol - 0.4 * width, orr + 0.4 * width
        else:
            ol = orr = float("nan")
            lo, hi = Om_l - 0.03, Om_r + 0.03
        F_max = 3.0 * max(Fc_of(rho_l), Fc_of(rho_r))
        hits = scan_bistability(S, kappa, lo, hi, F_max=F_max)
        tag = "below" if kappa < kstar else "above"
        if not hits:
            print(f"\nkappa={kappa:.4f} ({tag}): NO bistability in "
                  f"[{lo:.4f},{hi:.4f}]  <- lens absent, as predicted below "
                  f"threshold" if tag == "below" else "  <- UNEXPECTED")
            rows.append(dict(kappa=kappa, hb_lo="", hb_hi="", cf_lo=ol,
                             cf_hi=orr, n_hits=0, tip_lo="", tip_hi=""))
            continue
        hb_lo, hb_hi = hits[0][0], hits[-1][0]
        print(f"\nkappa={kappa:.4f} ({tag}): bistable over "
              f"[{hb_lo:.4f},{hb_hi:.4f}] ({len(hits)} of the scanned "
              f"frequencies); closed-form pair [{ol:.4f},{orr:.4f}]")

        tip_lo = tip_hi = float("nan")
        if kappa > kstar and np.isfinite(ol):
            w = orr - ol
            kw = dict(span=min(0.4 * w, 0.02), inner=max(0.0004, 0.04 * w),
                      n_pts=10, ds=0.01, F_max=F_max)
            # lips: cusps face inward
            L = cusp_tip(S, K_of(kappa), ol, side=+1, **kw)
            R = cusp_tip(S, K_of(kappa), orr, side=-1, **kw)
            if L:
                tip_lo = L["Omega_tip"]
            if R:
                tip_hi = R["Omega_tip"]
            print(f"   HB tips: {tip_lo:.5f} / {tip_hi:.5f}  "
                  f"(closed form {ol:.5f} / {orr:.5f})")
        rows.append(dict(kappa=kappa, hb_lo=hb_lo, hb_hi=hb_hi, cf_lo=ol,
                         cf_hi=orr, n_hits=len(hits), tip_lo=tip_lo,
                         tip_hi=tip_hi))

    out = os.path.join(DATA, "lips_hb_moderate.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote", out)
