"""Direct test of the phase-contour law against HB-located cusps.

THE POINT
---------
Eq. (5) of the Letter says cusps sit at arg G = -150 deg for a hardening
cubic.  Every figure so far shows that contour drawn from the receptance and
compares *derived* quantities (tongue widths, A3 slaving) against HB.  None
evaluates arg G at a cusp that was located WITHOUT using the phase condition.

That distinction matters.  The random-network ensemble
(ensemble_unified.py) finds cusps by root-finding on
f = Re(1/G)^2 - 3 Im(1/G)^2, which IS the phase condition rearranged;
evaluating arg G at those roots returns -150 deg to machine precision by
construction and validates nothing.

Here the cusp locations come from full multi-harmonic balance in the original
equations of motion, via the (Delta F)^{2/3} tongue-width intercept of
cusp_tip_intercept.py, which never solves at the tip and never references f.
We then evaluate arg G there from the LINEAR receptance alone.  Agreement is
a real test; disagreement is a real failure.

ERROR MODEL
-----------
The intercept estimator carries a fit-window systematic of order 1e-4..1e-3
in Omega (End Matter).  Near a pole arg G turns fast, so that Omega error
propagates into phase as

    sigma_phase = |d(arg G)/d Omega| * sigma_Omega .

A raw phase residual is therefore not interpretable on its own; the residual
normalized by the propagated Omega uncertainty is.  Both are reported.

Inputs (already on disk, no HB re-run needed):
    data/beaks_tips_refined.csv   w2 = 1.25, 16 couplings x 2 tips
    data/lips_hb_moderate.csv     w2 = 0.95, lips tips
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
TARGET = -150.0

# Omega uncertainty of the intercept estimator (End Matter: fit-window
# systematic spans 2-8e-5 at the baseline, degrading away from it).
SIG_OM = 3e-4


def G_2dof(Om: float, kappa: float, w2: float) -> complex:
    """Driving-point receptance at the nonlinear coordinate."""
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def arg_G(Om: float, kappa: float, w2: float) -> float:
    return float(np.degrees(np.angle(G_2dof(Om, kappa, w2))))


def dargG_dOm(Om: float, kappa: float, w2: float, h: float = 1e-6) -> float:
    """Phase sensitivity, deg per unit Omega, by central difference."""
    return (arg_G(Om + h, kappa, w2) - arg_G(Om - h, kappa, w2)) / (2 * h)


def collect() -> list[dict]:
    rows: list[dict] = []

    # ---- beaks family, w2 = 1.25 -----------------------------------------
    path = os.path.join(DATA, "beaks_tips_refined.csv")
    raw = np.genfromtxt(path, delimiter=",", names=True)
    for r in np.atleast_1d(raw):
        for side in ("left", "right"):
            om = float(r[f"Om_{side}"])
            if not np.isfinite(om):
                continue
            if not int(r[f"conv_{side}"]):        # estimator did not converge
                continue
            rows.append(dict(family="beaks", w2=1.25, kappa=float(r["kappa"]),
                             side=side, Om=om, estimator="tip",
                             sig_Om=SIG_OM))

    # ---- lips family, w2 = 0.95 ------------------------------------------
    path = os.path.join(DATA, "lips_hb_moderate.csv")
    raw = np.genfromtxt(path, delimiter=",", names=True)
    for r in np.atleast_1d(raw):
        hb_lo, hb_hi = float(r["hb_lo"]), float(r["hb_hi"])
        for side in ("lo", "hi"):
            om = float(r[f"tip_{side}"])
            if not np.isfinite(om):
                continue
            # A located cusp must lie within the frequency band in which full
            # HB actually finds bistability.  On the lips upper tips the
            # (dF)^{2/3} extrapolation runs past the last bistable frequency
            # -- the scan covers that range and finds nothing there -- so
            # those points are extrapolation artifacts, not cusps, and cannot
            # be used to test the contour.  eps ~ 0.65 there, outside the
            # leading-order window.
            if np.isfinite(hb_lo) and np.isfinite(hb_hi):
                tol = 0.02 * (hb_hi - hb_lo)
                if not (hb_lo - tol <= om <= hb_hi + tol):
                    print(f"  [excluded] lips kappa={float(r['kappa']):.4f} "
                          f"{side} tip {om:.5f} outside HB bistable band "
                          f"[{hb_lo:.5f},{hb_hi:.5f}]")
                    continue
            rows.append(dict(family="lips", w2=0.95, kappa=float(r["kappa"]),
                             side=side, Om=om, estimator="tip", sig_Om=SIG_OM))

    # The lips family is instead localized by the edges of the bistable band
    # that HB actually resolves.  That is a coarser measurement -- its
    # uncertainty is one scan step, not the tip estimator's fit-window
    # systematic -- so these points show WHERE the cusps sit (panel a) but are
    # not used for the precision residual (panel b).
    for r in np.atleast_1d(raw):
        hb_lo, hb_hi = float(r["hb_lo"]), float(r["hb_hi"])
        cf_lo, cf_hi = float(r["cf_lo"]), float(r["cf_hi"])
        if not all(np.isfinite(v) for v in (hb_lo, hb_hi, cf_lo, cf_hi)):
            continue
        w = cf_hi - cf_lo
        step = (2.0 * 0.4 * w + w) / 44.0        # 45-point scan over the window
        for side, om in (("lo", hb_lo), ("hi", hb_hi)):
            rows.append(dict(family="lips_band", w2=0.95,
                             kappa=float(r["kappa"]), side=side, Om=om,
                             estimator="band", sig_Om=step))
    return rows


def main() -> None:
    rows = collect()
    for d in rows:
        d["argG"] = arg_G(d["Om"], d["kappa"], d["w2"])
        d["resid"] = d["argG"] - TARGET
        d["slope"] = dargG_dOm(d["Om"], d["kappa"], d["w2"])
        d["sig"] = abs(d["slope"]) * d.get("sig_Om", SIG_OM)
        d["z"] = d["resid"] / d["sig"] if d["sig"] > 0 else np.nan
        # Omega offset that would close the phase residual exactly
        d["dOm_equiv"] = d["resid"] / d["slope"] if d["slope"] != 0 else np.nan

    print("=" * 78)
    print("arg G AT HB-LOCATED CUSP TIPS   (prediction: -150.000 deg)")
    print("=" * 78)
    print(f"{'family':>6} {'w2':>5} {'kappa':>7} {'side':>5} {'Om_HB':>10} "
          f"{'argG':>9} {'resid':>8} {'dargG/dOm':>10} {'dOm_equiv':>10}")
    for d in rows:
        print(f"{d['family']:>6} {d['w2']:>5.2f} {d['kappa']:>7.4f} "
              f"{d['side']:>5} {d['Om']:>10.6f} {d['argG']:>9.3f} "
              f"{d['resid']:>+8.3f} {d['slope']:>10.1f} {d['dOm_equiv']:>+10.2e}")

    prec = [d for d in rows if d["estimator"] == "tip"]
    res = np.array([d["resid"] for d in prec])
    dom = np.array([d["dOm_equiv"] for d in prec])
    print("\n" + "-" * 78)
    print(f"n = {len(prec)} tip-located cusps "
          f"({sum(d['family'] == 'beaks' for d in rows)} beaks, "
          f"{sum(d['family'] == 'lips' for d in prec)} lips); "
          f"plus {sum(d['estimator'] == 'band' for d in rows)} band-edge points")
    print(f"  phase residual:  mean {res.mean():+.3f} deg   "
          f"median {np.median(res):+.3f}   max|.| {np.abs(res).max():.3f}")
    print(f"  equivalent Omega offset: median {np.median(np.abs(dom)):.2e}, "
          f"max {np.abs(dom).max():.2e}")
    print(f"  (estimator's own Omega systematic is ~{SIG_OM:.0e})")

    for fam in ("beaks", "lips"):
        sub = np.array([d["resid"] for d in rows if d["family"] == fam])
        if sub.size:
            print(f"  {fam:>5}: mean {sub.mean():+.3f} deg, "
                  f"max|.| {np.abs(sub).max():.3f} deg, n = {sub.size}")

    # Dynamic range of the test.  A tolerance of 0.44 deg is only meaningful
    # against the phase the band actually traverses, and against the local
    # slope that converts degrees into a frequency displacement.  Without
    # this null the residual could be a trivial pass.
    beaks = [d for d in rows if d["family"] == "beaks"]
    if beaks:
        ks = sorted({d["kappa"] for d in beaks})
        oms = np.linspace(min(d["Om"] for d in beaks),
                          max(d["Om"] for d in beaks), 2001)
        spans, allv = [], []
        for k in ks:
            a = np.array([arg_G(o, k, 1.25) for o in oms])
            spans.append(a.max() - a.min())
            allv.append(a)
        allv = np.concatenate(allv)
        sl = np.abs([d["slope"] for d in beaks])
        worst = np.abs([d["resid"] for d in beaks]).max()
        print("\n" + "-" * 78)
        print("DYNAMIC RANGE OF THE PHASE TEST (hardening family)")
        print(f"  band Omega = {oms[0]:.4f}-{oms[-1]:.4f}, "
              f"kappa = {min(ks):.4f}-{max(ks):.4f} (x{max(ks)/min(ks):.2f})")
        print(f"  arg G span at fixed kappa: {min(spans):.1f} deg "
              f"(kappa={ks[int(np.argmin(spans))]:.4f}) to {max(spans):.1f} deg "
              f"(kappa={ks[int(np.argmax(spans))]:.4f}); "
              f"{allv.max() - allv.min():.1f} deg over the whole set")
        print(f"  worst residual {worst:.3f} deg = "
              f"{100 * worst / min(spans):.1f}% of the smallest span")
        print(f"  |d argG/dOm| at the located cusps: "
              f"{sl.min():.0f} to {sl.max():.0f} deg per unit Omega")
        print(f"  so {worst:.2f} deg corresponds to dOmega = "
              f"{worst / sl.max():.2e} to {worst / sl.min():.2e}")

    out = os.path.join(DATA, "phase_contour_validation.csv")
    with open(out, "w") as fh:
        fh.write("family,w2,kappa,side,Om_HB,argG,resid_deg,dargG_dOm,dOm_equiv\n")
        for d in rows:
            fh.write(f"{d['family']},{d['w2']},{d['kappa']},{d['side']},"
                     f"{d['Om']:.9f},{d['argG']:.6f},{d['resid']:.6f},"
                     f"{d['slope']:.4f},{d['dOm_equiv']:.6e}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
