"""N_HARM convergence study using the SAME window-refined estimator as Table I.

The first version of this study (recompute_beaks_table.py, mode "nharm") used
the fixed first-pass fit window, span = min(0.020, 0.45 * dOmega_LO), because
the window-refinement pass did not exist yet.  Table I was later regenerated
with refinement, and the two tables then disagreed by ~6% in dOmega at
kappa = 0.15 -- not a truncation effect at all, but two different estimators.

This script reruns the truncation study through refine_beaks_window.refine_one,
so the N_HARM = 3 rows reproduce Table I exactly and the table genuinely
certifies the measurement it accompanies.

Output: data/beaks_nharm_refined.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, K_2dof  # noqa: E402
from recompute_beaks_table import (  # noqa: E402
    closed_form_cusps, Fc_of, KAP_STAR, Z1, Z2, BETA1, NT,
)
from refine_beaks_window import refine_one  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

KAPPAS = (0.13, 0.15, 0.20)
NHARMS = (2, 3, 4, 5)


def main():
    rows = []
    for kappa in KAPPAS:
        cusps = [x for x in closed_form_cusps(kappa) if x > 1.2]
        if len(cusps) < 2:
            continue
        Om_l, Om_r = cusps
        dOm_LO = Om_r - Om_l
        F_max = 3.0 * max(Fc_of(Om_l, kappa), Fc_of(Om_r, kappa))
        K = K_2dof(kappa)
        for nh in NHARMS:
            t0 = time.time()
            S = make_solver(2, nh, NT, [Z1, Z2], BETA1)
            L = refine_one(S, K, Om_l, -1, dOm_LO, F_max)
            R = refine_one(S, K, Om_r, +1, dOm_LO, F_max)
            if L is None or R is None:
                print(f"  kappa={kappa} nh={nh}: FAILED", flush=True)
                continue
            dOm = R["tip"] - L["tip"]
            mu = kappa - KAP_STAR
            rows.append(dict(kappa=kappa, nh=nh, mu=mu,
                             Om_left=L["tip"], Om_right=R["tip"], dOm=dOm,
                             c_emp=dOm / np.sqrt(mu),
                             spread_left=L["spread"], spread_right=R["spread"]))
            print(f"  kappa={kappa:.2f} nh={nh}: Om_l={L['tip']:.6f} "
                  f"Om_r={R['tip']:.6f} dOm={dOm:.6f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            out = os.path.join(DATA, "beaks_nharm_refined.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    print("\nwrote", os.path.join(DATA, "beaks_nharm_refined.csv"), flush=True)


if __name__ == "__main__":
    main()
