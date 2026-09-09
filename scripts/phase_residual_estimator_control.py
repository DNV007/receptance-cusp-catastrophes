"""Control test: is the Fig. 2(b) hardening phase residual estimator bias or a
real deformation of the phase contour?

The (Delta F)^{2/3} tongue-width intercept locates a cusp tip; evaluating
arg G there (phase_contour_validation.py) gives a residual against -150 deg
that is one-signed and grows with coupling.  That residual is either
(i) a genuine higher-comb deformation of the contour, or (ii) a fit-window
systematic of the intercept estimator.  This script separates them.

METHOD -- identical estimator (refine_beaks_window.refine_one, shrinking
windows), two models differing ONLY in harmonic count:

    nh = 1 : the reduced single-harmonic (describing-function) model.  Its
             cusp sits at arg G = -150 deg EXACTLY -- the fundamental balance
             IS rho^2 = 3 sigma^2 in the SAME physical-damping receptance
             G_2dof used below.  So any arg G residual at the located tip is
             PURE ESTIMATOR BIAS; the physics contributes zero.

    nh = 3 : the multi-harmonic model (reused from beaks_tips_refined.csv,
             the file phase_contour_validation.py already evaluates).  Its
             residual is estimator bias PLUS any real contour deformation.

If nh=1 reproduces the sign, magnitude and coupling-monotonicity of nh=3, the
residual is the estimator, not the contour.

Beaks (hardening, w2=1.25) family only -- the one with the open residual.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, K_2dof  # noqa: E402
from refine_beaks_window import refine_one  # noqa: E402
from recompute_beaks_table import (  # noqa: E402
    closed_form_cusps, Fc_of, Z1, Z2, BETA1, NT,
)

W1, W2 = 1.0, 1.25
TARGET = -150.0
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

# same coupling grid as beaks_tips_refined.csv
KAPPAS = [0.1200, 0.1210, 0.1225, 0.1250, 0.1275,
          0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
          0.20, 0.22, 0.25, 0.28, 0.30]

# a few kappa to re-run at nh=3 as a harness check against the stored CSV
SPOTCHECK_NH3 = [0.1200, 0.16, 0.30]


def G_2dof(Om, kappa):
    """Physical-damping driving-point receptance -- the SAME one the HB solver
    integrates and the SAME one phase_contour_validation.py evaluates."""
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, W2**2 + kappa]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def arg_G(Om, kappa):
    return float(np.degrees(np.angle(G_2dof(Om, kappa))))


def refine_family(nh, kappas):
    """Locate both beaks tips at each kappa with the refine_one estimator."""
    S = make_solver(2, nh, NT, [Z1, Z2], BETA1)
    rows = []
    for kappa in kappas:
        cusps = [x for x in closed_form_cusps(kappa) if x > 1.2]
        if len(cusps) < 2:
            continue
        Om_l, Om_r = cusps
        dOm_LO = Om_r - Om_l
        F_max = 3.0 * max(Fc_of(Om_l, kappa), Fc_of(Om_r, kappa))
        K = K_2dof(kappa)
        t0 = time.time()
        L = refine_one(S, K, Om_l, -1, dOm_LO, F_max)
        R = refine_one(S, K, Om_r, +1, dOm_LO, F_max)
        for tag, res in (("left", L), ("right", R)):
            if res is None or not res["converged"]:
                print(f"  nh={nh} kappa={kappa:.4f} {tag:>5}: not converged",
                      flush=True)
                continue
            om = res["tip"]
            rows.append(dict(nh=nh, kappa=kappa, side=tag, om_tip=om,
                             resid=arg_G(om, kappa) - TARGET,
                             spread=res["spread"], r2=res["r2_best"]))
            print(f"  nh={nh} kappa={kappa:.4f} {tag:>5}: Om={om:.6f}  "
                  f"resid={rows[-1]['resid']:+.3f} deg  "
                  f"(spread {res['spread']:.1e}) [{time.time()-t0:.0f}s]",
                  flush=True)
    return rows


def nh3_from_csv():
    """arg G residuals at the stored (published) nh=3 tips."""
    path = os.path.join(DATA, "beaks_tips_refined.csv")
    raw = np.genfromtxt(path, delimiter=",", names=True)
    rows = []
    for r in np.atleast_1d(raw):
        for side, key, conv in (("left", "Om_left", "conv_left"),
                                ("right", "Om_right", "conv_right")):
            om = float(r[key])
            if not np.isfinite(om) or not int(r[conv]):
                continue
            rows.append(dict(nh=3, kappa=float(r["kappa"]), side=side,
                             om_tip=om, resid=arg_G(om, float(r["kappa"])) - TARGET,
                             spread=float("nan"), r2=float("nan")))
    return rows


def stats(rows, label):
    res = np.array([r["resid"] for r in rows])
    one_signed = bool(np.all(res > 0) or np.all(res < 0))
    # monotone in kappa? sort by kappa, check right-tip residual trend
    rt = sorted([r for r in rows if r["side"] == "right"], key=lambda r: r["kappa"])
    trend = np.polyfit([r["kappa"] for r in rt], [r["resid"] for r in rt], 1)[0] \
        if len(rt) >= 2 else float("nan")
    print(f"\n  {label}:  n={len(rows)}  mean {res.mean():+.3f}  "
          f"median {np.median(res):+.3f}  max|.| {np.abs(res).max():.3f} deg")
    print(f"    one-signed: {one_signed}   d(resid_right)/dkappa: {trend:+.2f} deg")
    return res


def main():
    print("=" * 72)
    print("ESTIMATOR-BIAS CONTROL (beaks, w2=1.25, refine_one, target -150 deg)")
    print("=" * 72)

    print("\n-- nh=3 harness spot-check vs stored beaks_tips_refined.csv --")
    chk = refine_family(3, SPOTCHECK_NH3)
    csv3 = {(r["kappa"], r["side"]): r for r in nh3_from_csv()}
    for r in chk:
        key = (round(r["kappa"], 4), r["side"])
        match = next((v for k, v in csv3.items()
                      if abs(k[0] - r["kappa"]) < 1e-6 and k[1] == r["side"]), None)
        if match:
            print(f"    kappa={r['kappa']:.4f} {r['side']:>5}: fresh "
                  f"{r['om_tip']:.6f} vs stored {match['om_tip']:.6f}  "
                  f"(dOm={r['om_tip']-match['om_tip']:+.1e})")

    print("\n-- nh=1 reduced model (cusp on contour EXACTLY; residual = bias) --")
    r1 = refine_family(1, KAPPAS)

    print("\n-- nh=3 published tips (bias + any real deformation) --")
    r3 = nh3_from_csv()

    print("\n" + "=" * 72)
    a = stats(r1, "nh=1 (pure estimator bias)")
    b = stats(r3, "nh=3 (published, bias+physics)")
    if a.size and b.size:
        print(f"\n  median-residual ratio nh=3 / nh=1: "
              f"{np.median(b)/np.median(a):.2f}")
        print("  ratio ~1 and same sign  =>  residual is the estimator, "
              "not the contour.")

    out = os.path.join(DATA, "phase_residual_estimator_control.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["nh", "kappa", "side", "om_tip",
                                           "resid", "spread", "r2"])
        w.writeheader()
        for r in r1 + r3:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
