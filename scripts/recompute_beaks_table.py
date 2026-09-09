"""Recompute the upper-pair cusp tips (manuscript Table II) by the
(Delta F)^{2/3} intercept method, plus an N_HARM convergence study.

Replaces locate_lips_cusps_augmented.py, which located the tip as the Omega
where Newton on the augmented system fails to converge -- a conditioning
boundary that drifts with the harmonic truncation (up to 5e-3 between
N_HARM = 2 and 3, against a quoted 1e-7 bracket).

Outputs
  data/beaks_tips_intercept.csv   extended kappa grid, converged truncation
  data/beaks_nharm_study.csv      tip vs N_HARM at three couplings
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, cusp_tip, K_2dof  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.02
BETA1 = 0.147
CVAL = 0.75 * BETA1
KAP_STAR = 0.1183654          # reduced-model beaks point
NT = 256


# ------------------------------------------------------------ closed form
def _modal(kappa):
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, W2**2 + kappa]])
    ev, evec = np.linalg.eigh(K)
    Cm = np.diag([2 * Z1, 2 * Z2])
    return (ev[0], ev[1], evec[0, 0], evec[0, 1],
            0.5 * (evec[:, 0] @ Cm @ evec[:, 0]),
            0.5 * (evec[:, 1] @ Cm @ evec[:, 1]))


def transfer(Om, kappa):
    wa2, wb2, pa1, pb1, za, zb = _modal(kappa)
    G = pa1**2 / (wa2 - Om**2 + 2j * za * Om) + \
        pb1**2 / (wb2 - Om**2 + 2j * zb * Om)
    inv = 1.0 / G
    return inv.real, inv.imag


def closed_form_cusps(kappa, lo=0.7, hi=1.7, n=60001):
    oms = np.linspace(lo, hi, n)
    a = np.empty(n); b = np.empty(n)
    for i, o in enumerate(oms):
        a[i], b[i] = transfer(o, kappa)
    f = a**2 - 3 * b**2
    out = []
    for i in np.where(np.diff(np.sign(f)))[0]:
        x0, x1, y0, y1 = oms[i], oms[i + 1], f[i], f[i + 1]
        Om = x0 - y0 * (x1 - x0) / (y1 - y0)
        if transfer(Om, kappa)[0] < 0:
            out.append(float(Om))
    return sorted(out)


def Fc_of(Om, kappa):
    rho, _ = transfer(Om, kappa)
    return float(np.sqrt(-8.0 * rho**3 / (27.0 * CVAL)))


# ------------------------------------------------------------ one coupling
def upper_pair_tips(kappa, nh, verbose=False):
    cusps = closed_form_cusps(kappa)
    upper = [x for x in cusps if x > 1.2]
    if len(upper) < 2:
        return None
    Om_l, Om_r = upper[0], upper[1]
    dOm_LO = Om_r - Om_l
    Fc_l, Fc_r = Fc_of(Om_l, kappa), Fc_of(Om_r, kappa)

    # Windows must not reach across the gap into the other cusp.
    span = min(0.020, 0.45 * dOm_LO)
    inner = max(0.0004, 0.03 * dOm_LO)
    F_max = 3.0 * max(Fc_l, Fc_r)
    S = make_solver(2, nh, NT, [Z1, Z2], BETA1)
    K = K_2dof(kappa)

    kw = dict(span=span, inner=inner, n_pts=12, verbose=verbose,
              ds=0.01, F_max=F_max)
    left = cusp_tip(S, K, Om_l, side=-1, **kw)
    right = cusp_tip(S, K, Om_r, side=+1, **kw)
    if left is None or right is None:
        return None
    return dict(
        kappa=kappa, nh=nh, mu=kappa - KAP_STAR,
        Om_left=left["Omega_tip"], Om_right=right["Omega_tip"],
        r2_left=left["r2"], r2_right=right["r2"],
        Om_left_LO=Om_l, Om_right_LO=Om_r, dOm_LO=dOm_LO,
        dOm=right["Omega_tip"] - left["Omega_tip"],
        span=span, inner=inner,
    )


def _write(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print("wrote", path, flush=True)


FIELDS = ["kappa", "nh", "mu", "Om_left", "Om_right", "dOm", "c_emp",
          "r2_left", "r2_right", "Om_left_LO", "Om_right_LO", "dOm_LO"]


def finish(r):
    r["c_emp"] = r["dOm"] / np.sqrt(r["mu"]) if r["mu"] > 0 else float("nan")
    return r


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # ---------------------------------------------- N_HARM convergence study
    if mode in ("all", "nharm"):
        print("\n===== N_HARM convergence study =====", flush=True)
        rows = []
        for kappa in (0.13, 0.15, 0.20):
            for nh in (2, 3, 4, 5):
                t0 = time.time()
                r = upper_pair_tips(kappa, nh)
                if r is None:
                    print(f"  kappa={kappa} nh={nh}: FAILED", flush=True)
                    continue
                r = finish(r)
                rows.append(r)
                print(f"  kappa={kappa:.3f} nh={nh}: "
                      f"Om_l={r['Om_left']:.6f} Om_r={r['Om_right']:.6f} "
                      f"dOm={r['dOm']:.6f} c={r['c_emp']:.5f} "
                      f"r2=({r['r2_left']:.5f},{r['r2_right']:.5f}) "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        _write(os.path.join(DATA, "beaks_nharm_study.csv"), rows, FIELDS)

    # ---------------------------------------------------- extended kappa grid
    if mode in ("all", "grid"):
        print("\n===== extended kappa grid (nh=3) =====", flush=True)
        KAPPAS = [0.1200, 0.1210, 0.1225, 0.1250, 0.1275,
                  0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
                  0.20, 0.22, 0.25, 0.28, 0.30]
        rows = []
        for kappa in KAPPAS:
            t0 = time.time()
            r = upper_pair_tips(kappa, 3)
            if r is None:
                print(f"  kappa={kappa}: FAILED", flush=True)
                continue
            r = finish(r)
            rows.append(r)
            print(f"  kappa={kappa:.4f} mu={r['mu']:.5f}: "
                  f"Om_l={r['Om_left']:.6f} Om_r={r['Om_right']:.6f} "
                  f"dOm={r['dOm']:.6f} c={r['c_emp']:.5f} "
                  f"r2=({r['r2_left']:.5f},{r['r2_right']:.5f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            _write(os.path.join(DATA, "beaks_tips_intercept.csv"), rows, FIELDS)
