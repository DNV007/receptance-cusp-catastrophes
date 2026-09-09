"""Confirm inverse-designed networks in full multi-harmonic balance.

inverse_design_targets.py prescribes a birth frequency and returns a network
(w2, kappa*) plus a predicted topology.  Nothing so far checks that the
network actually does what was ordered: the design step lives entirely in the
linear receptance, and the prediction is about the nonlinear response.

Two designs are taken exactly as the solver returned them -- not tuned, not
hand-picked -- and the bistable tongue width is scanned in full harmonic
balance on the original equations of motion, below and above the designed
coupling.  The predictions are crisp and opposite:

  BEAKS (a_Omega > 0), target Om* = 1.20:
      kappa < kappa*  ->  bistable band CONNECTED through Om*
      kappa > kappa*  ->  GAP straddling Om*, bistability on both sides

  LIPS (a_Omega < 0), target Om* = 0.95:
      kappa < kappa*  ->  NO bistability anywhere near Om*
      kappa > kappa*  ->  isolated LENS containing Om*

If the receptance construction were decorative, these would not come out in
the sign the classifier assigns before any nonlinear computation is done.

Run with the project venv:  ../.venv/bin/python inverse_design_verify.py
"""
from __future__ import annotations

import os

import numpy as np

from cusp_tip_intercept import make_solver, tongue_width

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
BETA1 = 0.147
NH, NT = 3, 256

# straight from data/inverse_design_targets.csv, unmodified
DESIGNS = [
    dict(tag="beaks", Om_t=1.20, w2=1.15958, kap=0.08683, a_Om=+488.5,
         dk=0.030, span=0.075),
    dict(tag="lips", Om_t=0.95, w2=0.80132, kap=0.39204, a_Om=-179.0,
         dk=0.030, span=0.055),
]


def K_of(kappa: float, w2: float) -> np.ndarray:
    return np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])


def scan(S, w2, kappa, om_lo, om_hi, n=21, F_max=3.0):
    oms = np.linspace(om_lo, om_hi, n)
    K = K_of(kappa, w2)
    out = []
    for om in oms:
        w = tongue_width(S, K, float(om), F_max=F_max)
        out.append((float(om), float(w) if np.isfinite(w) else float("nan")))
    return out


def summarize(rows, Om_t):
    live = [(o, w) for o, w in rows if np.isfinite(w) and w > 0]
    if not live:
        return "no bistability in the window", None
    lo, hi = live[0][0], live[-1][0]
    # is there an interior gap?
    flags = [np.isfinite(w) and w > 0 for _, w in rows]
    gaps = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            j = i
            while j < len(flags) and not flags[j]:
                j += 1
            if 0 < i and j < len(flags):        # interior only
                gaps.append((rows[i][0], rows[j - 1][0]))
            i = j
        else:
            i += 1
    at_target = min(rows, key=lambda r: abs(r[0] - Om_t))
    desc = f"bistable over [{lo:.4f}, {hi:.4f}]"
    if gaps:
        desc += "; interior gap(s) " + ", ".join(
            f"[{a:.4f}, {b:.4f}]" for a, b in gaps)
    else:
        desc += "; connected"
    tgt = at_target[1]
    desc += f"; at Om*={Om_t:.3f} width = " + (
        "none" if not np.isfinite(tgt) or tgt <= 0 else f"{tgt:.3e}")
    return desc, gaps


def main() -> None:
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    lines = []
    for d in DESIGNS:
        Om_t, w2, kap = d["Om_t"], d["w2"], d["kap"]
        lo, hi = Om_t - d["span"], Om_t + d["span"]
        print("=" * 74)
        print(f"DESIGN: target Om* = {Om_t}  ->  w2 = {w2}, kappa* = {kap}")
        print(f"        predicted a_Omega = {d['a_Om']:+.1f}  ->  {d['tag']}")
        print("=" * 74)
        for tag, kappa in (("below", kap - d["dk"]), ("above", kap + d["dk"])):
            rows = scan(S, w2, kappa, lo, hi)
            desc, gaps = summarize(rows, Om_t)
            print(f"  kappa = {kappa:.5f} ({tag} kappa*): {desc}", flush=True)
            for o, w in rows:
                lines.append(f"{d['tag']},{Om_t},{w2},{kappa:.6f},{tag},"
                             f"{o:.6f},{w:.6e}")
        print()

    out = os.path.join(DATA, "inverse_design_verify.csv")
    with open(out, "w") as fh:
        fh.write("design,Om_target,w2,kappa,side,Omega,tongue_width\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
