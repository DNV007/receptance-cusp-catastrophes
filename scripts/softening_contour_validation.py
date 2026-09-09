"""Softening branch of the phase-contour law, tested against full HB.

Equation (5) of the Letter has two branches: arg G = -/+150 deg for a
hardening cubic and -/+30 deg for a softening one.  Everything else in the
Letter runs at beta1 = +0.147, so the softening half of the claim -- which
appears in the abstract -- has no numerical support at all.  This supplies it.

The test is the same one phase_contour_validation.py runs for the hardening
branch, and it is independent in the same way: cusp tips come from full
multi-harmonic balance in the original equations of motion, located as the
(Delta F)^{2/3} tongue-width intercept, an estimator that never solves at the
tip and never evaluates f = rho^2 - 3 sigma^2.  arg G is then read off the
LINEAR receptance there.

GEOMETRY.  For a softening cubic the physical branch requires
rho = Re(1/G) > 0, so the organizers sit BELOW the modes and the bistable
tongue lies at Omega < Omega_c: cusp_tip is called with side = -1.  This is
the opposite of the hardening case and is itself a check on the sign rule --
if the sign convention were wrong, no bistability would be found on this side
at all.

Run with the project venv, which has jax and scipy together:
    ../.venv/bin/python softening_contour_validation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, cusp_tip, K_2dof  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
BETA1 = -0.147                 # SOFTENING
NH, NT = 3, 256
TARGET = -30.0                 # deg, the softening branch of Eq. (5)

KAPPAS = [0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]
W2 = 1.25


def G_2dof(Om: float, kappa: float, w2: float = W2) -> complex:
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def f_of(Om: float, kappa: float) -> float:
    inv = 1.0 / G_2dof(Om, kappa)
    return inv.real**2 - 3 * inv.imag**2


def _bisect(fn, a, b, n=200):
    fa = fn(a)
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = fn(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def predict_soft_cusp(kappa: float):
    """Softening cusp from the receptance: f = 0 on the rho > 0 branch."""
    oms = np.linspace(0.6, 1.6, 4000)
    fv = np.array([f_of(o, kappa) for o in oms])
    for i in range(len(oms) - 1):
        if fv[i] * fv[i + 1] < 0:
            r = _bisect(lambda x: f_of(x, kappa), oms[i], oms[i + 1])
            if (1.0 / G_2dof(r, kappa)).real > 0:
                return float(r)
    return None


def main() -> None:
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    rows = []
    for kappa in KAPPAS:
        om_pred = predict_soft_cusp(kappa)
        if om_pred is None:
            print(f"kappa={kappa:.3f}: no softening cusp predicted")
            continue
        res = cusp_tip(S, K_2dof(kappa), om_pred, side=-1,
                       span=0.020, inner=0.002, n_pts=10, F_max=3.0)
        if res is None:
            print(f"kappa={kappa:.3f}: HB tip not resolved")
            continue
        om_hb = res["Omega_tip"]
        ang = float(np.degrees(np.angle(G_2dof(om_hb, kappa))))
        rows.append(dict(kappa=kappa, Om_pred=om_pred, Om_hb=om_hb,
                         argG=ang, resid=ang - TARGET, r2=res["r2"],
                         n=res["n_pts"]))
        print(f"kappa={kappa:.3f}  Om_pred={om_pred:.6f}  Om_HB={om_hb:.6f}  "
              f"argG={ang:+.3f}  resid={ang - TARGET:+.3f}  r2={res['r2']:.5f}",
              flush=True)

    if not rows:
        print("no points")
        return

    r = np.array([d["resid"] for d in rows])
    print("\n" + "-" * 70)
    print(f"n = {len(rows)} HB-located softening cusps, beta1 = {BETA1}")
    print(f"  prediction: arg G = {TARGET:+.1f} deg")
    print(f"  residual: mean {r.mean():+.3f}  median {np.median(r):+.3f}  "
          f"max|.| {np.abs(r).max():.3f} deg")

    out = os.path.join(DATA, "softening_contour_validation.csv")
    with open(out, "w") as fh:
        fh.write("kappa,Om_pred,Om_hb,argG,resid_deg,r2,n_pts\n")
        for d in rows:
            fh.write(f"{d['kappa']},{d['Om_pred']:.9f},{d['Om_hb']:.9f},"
                     f"{d['argG']:.6f},{d['resid']:.6f},{d['r2']:.6f},"
                     f"{d['n']}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
