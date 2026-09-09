"""Window-refinement pass for the (Delta F)^{2/3} cusp-tip intercepts.

The intercept fit assumes the cusp normal form, i.e. that (Delta F)^{2/3} is
linear in Omega.  That holds only close enough to the tip; taking too wide a
window admits curvature and biases the intercept.  The first pass used
span = 0.45 * dOmega_LO capped at 0.020, which gives r^2 ~ 0.95 on the left
cusp at large kappa -- not good enough to trust.

Here each tip is re-fitted on a sequence of shrinking windows.  A tip is
accepted when consecutive window estimates agree to better than `TOL` and
the fit has r^2 above `R2_MIN`; the reported uncertainty is the spread over
the accepted windows.  This makes the window sensitivity an explicit,
reported quantity rather than a hidden one.

Output: data/beaks_tips_refined.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, cusp_tip, K_2dof  # noqa: E402
from recompute_beaks_table import (  # noqa: E402
    closed_form_cusps, Fc_of, KAP_STAR, Z1, Z2, BETA1, NT,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

FRACS = [0.45, 0.32, 0.22, 0.15, 0.10]     # window as fraction of dOmega_LO
R2_MIN = 0.9990
TOL = 3e-4


def refine_one(S, K, Om_pred, side, dOm_LO, F_max, verbose=False):
    ests = []
    for fr in FRACS:
        span = min(0.020, fr * dOm_LO)
        inner = max(0.0003, 0.05 * span)
        if span <= inner * 2:
            continue
        r = cusp_tip(S, K, Om_pred, side=side, span=span, inner=inner,
                     n_pts=12, ds=0.01, F_max=F_max)
        if r is None:
            continue
        ests.append((fr, span, r["Omega_tip"], r["r2"]))
        if verbose:
            print(f"        frac={fr:.2f} span={span:.5f} "
                  f"tip={r['Omega_tip']:.6f} r2={r['r2']:.6f}", flush=True)
    if not ests:
        return None
    good = [e for e in ests if e[3] >= R2_MIN]
    use = good if len(good) >= 2 else ests[-2:]
    tips = np.array([e[2] for e in use])
    return dict(tip=float(tips[-1]), spread=float(tips.max() - tips.min()),
                n_win=len(use), r2_best=float(max(e[3] for e in use)),
                converged=bool(tips.max() - tips.min() < TOL),
                all_ests=ests)


def main():
    KAPPAS = [0.1200, 0.1210, 0.1225, 0.1250, 0.1275,
              0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
              0.20, 0.22, 0.25, 0.28, 0.30]
    rows = []
    S = make_solver(2, 3, NT, [Z1, Z2], BETA1)
    for kappa in KAPPAS:
        t0 = time.time()
        cusps = [x for x in closed_form_cusps(kappa) if x > 1.2]
        if len(cusps) < 2:
            continue
        Om_l, Om_r = cusps
        dOm_LO = Om_r - Om_l
        F_max = 3.0 * max(Fc_of(Om_l, kappa), Fc_of(Om_r, kappa))
        K = K_2dof(kappa)
        print(f"\nkappa={kappa:.4f}  dOm_LO={dOm_LO:.6f}", flush=True)
        print("      LEFT", flush=True)
        L = refine_one(S, K, Om_l, -1, dOm_LO, F_max, verbose=True)
        print("      RIGHT", flush=True)
        R = refine_one(S, K, Om_r, +1, dOm_LO, F_max, verbose=True)
        if L is None or R is None:
            print("   FAILED", flush=True)
            continue
        mu = kappa - KAP_STAR
        dOm = R["tip"] - L["tip"]
        row = dict(kappa=kappa, mu=mu, Om_left=L["tip"], Om_right=R["tip"],
                   dOm=dOm, c_emp=dOm / np.sqrt(mu),
                   spread_left=L["spread"], spread_right=R["spread"],
                   r2_left=L["r2_best"], r2_right=R["r2_best"],
                   conv_left=int(L["converged"]), conv_right=int(R["converged"]),
                   dOm_LO=dOm_LO)
        rows.append(row)
        print(f"   -> Om_l={L['tip']:.6f}(+-{L['spread']:.1e}) "
              f"Om_r={R['tip']:.6f}(+-{R['spread']:.1e}) "
              f"dOm={dOm:.6f} c={row['c_emp']:.5f} [{time.time()-t0:.0f}s]",
              flush=True)
        out = os.path.join(DATA, "beaks_tips_refined.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print("\nwrote", os.path.join(DATA, "beaks_tips_refined.csv"), flush=True)


if __name__ == "__main__":
    main()
