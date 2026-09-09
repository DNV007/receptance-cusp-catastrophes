"""Compute full-HB fold forcings F_SN(Omega) (turning points of the arclength
continuation) for the beaks (hub) and lips (2-DOF) bistability maps, below/at/
above each birth. Saved to CSV for the 2x3 map figure. Closed-form fold curves
are computed in the plot script (cheap)."""
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndof_hb import build_K
from cusp_tip_intercept import make_solver, F_folds

ROOT = Path(__file__).parent.parent
BETA1 = 0.147
C3 = 0.75 * BETA1


def K_hub(om, k):
    N = len(om)
    K = np.diag(np.asarray(om, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += k; K[j, j] += k
        K[0, j] -= k; K[j, 0] -= k
    return K


def rs(om, ze, Om, k, Kf):
    N = len(om)
    Z = Kf(om, k) - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in ze])
    inv = 1.0 / np.linalg.solve(Z, np.eye(N)[:, 0])[0]
    return inv.real, inv.imag


def cf_Fc(om, ze, Om, k, Kf):
    r, s = rs(om, ze, Om, k, Kf)
    if r >= 0:
        return None
    yc = -2 * r / (3 * C3)
    F2 = yc * ((r + C3 * yc) ** 2 + s ** 2)
    return np.sqrt(F2) if F2 > 0 else None


# Fold forcings come from cusp_tip_intercept.F_folds, which interpolates the
# tangent's F-component to zero rather than taking the midpoint of a bracketing
# pair.  That distinction matters here: near the lips birth the tongue is only
# ~5e-4 wide in F, comparable to the old ds = 0.008 midpoint error, which made
# the measured widths meaningless (and about 2x too large).


CONFIGS = [
    # label, om, ze, Kbuilder, kappas, om_band, nh, n_om
    # The lips lens is only ~0.02 wide in Omega just above its birth, so it
    # needs a finer frequency grid than the broad beaks band.
    ("beaks", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02], K_hub,
     [0.05, 0.10, 0.14], (1.11, 1.28), 5, 22),
    # omega_2 = 0.95 keeps the anharmonicity at the born cusps to
    # eps = 0.30-0.41, inside the validity window of Sec. II.E, while staying
    # firmly on the lips side.  The earlier omega_2 = 0.60 choice sat at
    # eps ~ 0.7, where full HB realizes only part of the predicted lens
    # (Sec. V.D).  kappa* = 0.2832, so these bracket the birth.
    ("lips", [1.0, 0.95], [0.015, 0.02], build_K,
     [0.263, 0.303, 0.333], (1.00, 1.09), 7, 60),
]

rows = []
for (lab, om, ze, Kf, kappas, band, nh, n_om) in CONFIGS:
    N = len(om)
    oms = np.linspace(band[0], band[1], n_om)
    S = make_solver(N, nh, 512, ze, BETA1)
    for k in kappas:
        K = Kf(om, k)
        # Give the continuation generous headroom in F: at the far edge of the
        # lens the folds sit well above the closed-form cusp forcing, and the
        # old 1.7*Fc ceiling truncated the right half of the lips lens.
        Fc = max([cf_Fc(om, ze, Om, k, Kf) or 0.0 for Om in oms] + [0.5])
        Fmax = 3.0 * Fc
        for Om in oms:
            for Ff in F_folds(S, K, float(Om), ds=0.01, F_max=Fmax):
                rows.append([lab, f"{k:.3f}", f"{Om:.5f}", f"{Ff:.6f}"])
        print(f"{lab} kappa={k}: done (F_max={Fmax:.2f})", flush=True)

with (ROOT / "data" / "hb_bistability_maps.csv").open("w") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "kappa", "omega", "F_fold"])
    w.writerows(rows)
print(f"wrote {len(rows)} fold points to data/hb_bistability_maps.csv")
