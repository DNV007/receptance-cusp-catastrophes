"""Cusp 3/2 closure law, shown compensated.

The previous version was a raw log-log plot of Delta F against Omega - Omega_c
over seven decades.  On those axes the analytic curve and the slope-3/2 guide
lay exactly on top of one another and the full-HB points occupied one corner,
so the panel demonstrated little beyond a straight line being straight, and
most of it was empty.

Here the 3/2 power is divided out:

    y = Delta F_tongue / (Omega - Omega_c)^{3/2},

which is constant if and only if the closure exponent is exactly 3/2.  Any
departure appears as a slope, so flatness of the closed form across five
decades is a far stronger statement than collinearity on a log-log plot, and
the higher-order bending away from the tip becomes visible rather than hidden.

The full-HB tongue widths are recomputed here with the validated F_folds
routine rather than read from data/cusp_scaling_law.csv, whose widths sit on a
coarse forcing grid and whose stored Omega - Omega_c column assumes an older
tip estimate (1.06793) inconsistent with the intercept value now reported.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st
from cusp_tip_intercept import make_solver, tongue_width, K_2dof

ROOT = Path(__file__).parent.parent

w1, w2, z1, z2 = 1.0, 1.25, 0.015, 0.02
kappa, beta1 = 0.10, 0.147
K = np.array([[w1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
e, V = np.linalg.eigh(K)
wa, wb = np.sqrt(e[0]), np.sqrt(e[1])
Cm = np.diag([2 * z1, 2 * z2])
za = 0.5 * (V[:, 0] @ Cm @ V[:, 0])
zb = 0.5 * (V[:, 1] @ Cm @ V[:, 1])
phi_a, phi_b = V[0, 0], V[0, 1]
c = 0.75 * beta1


def ab(Om):
    Da = wa**2 - Om**2 + 2j * za * Om
    Db = wb**2 - Om**2 + 2j * zb * Om
    inv = 1.0 / (phi_a**2 / Da + phi_b**2 / Db)
    return inv.real, inv.imag


Om_c = brentq(lambda O: (lambda r, s: r**2 - 3 * s**2)(*ab(O)), 1.04, 1.08)


def folds_at(Om):
    a, b = ab(Om)
    disc = 4 * c**2 * (a**2 - 3 * b**2)
    if disc < 0:
        return np.nan
    sd = np.sqrt(disc)
    y1 = (-4 * c * a + sd) / (6 * c**2)
    y2 = (-4 * c * a - sd) / (6 * c**2)
    F1 = np.sqrt(y1 * ((a + c * y1) ** 2 + b**2))
    F2 = np.sqrt(y2 * ((a + c * y2) ** 2 + b**2))
    return abs(F1 - F2)


eps_an = np.logspace(-7, -1.75, 400)
dF_an = np.array([folds_at(Om_c + t) for t in eps_an])

# full HB, measured with the same routine used for the cusp tips
OM_C_HB = 1.067319          # (Delta F)^{2/3} intercept, Sec. IV.B
S = make_solver(2, 3, 256, [z1, z2], beta1)
Kb = K_2dof(kappa)
eps_hb = np.logspace(np.log10(1.2e-3), np.log10(1.6e-2), 16)
dF_hb = np.array([tongue_width(S, Kb, float(OM_C_HB + t), ds=0.004, F_max=0.25)
                  for t in eps_hb])
ok = np.isfinite(dF_hb) & (dF_hb > 0)
eps_hb, dF_hb = eps_hb[ok], dF_hb[ok]

# Normalise by the small-detuning plateau, so the vertical axis reads directly
# as a fractional departure from the exact 3/2 law and the panel is filled by
# the 1-3% structure rather than by empty space above zero.
plateau = float(np.nanmedian((dF_an / eps_an ** 1.5)[eps_an < 1e-4]))

fig, ax = plt.subplots(figsize=(3.4, 2.55))
ax.semilogx(eps_an, dF_an / eps_an ** 1.5 / plateau, "-", color=st.C_ANALYTIC,
            lw=1.3, label="modal closed form")
ax.semilogx(eps_hb, dF_hb / eps_hb ** 1.5 / plateau, "o", ms=3.4, mfc=st.C_HB,
            mec="white", mew=0.5, ls="", label="full HB")
ax.axhline(1.0, color=st.GREY, ls="--", lw=0.8, zorder=0)
ax.text(1.3e-7, 1.002, r"exact $3/2$", fontsize=6.8, color=st.GREY,
        va="bottom")

ax.set_xlabel(r"$\Omega - \Omega_c$")
ax.set_ylabel(r"$\Delta F_{\rm tongue}\,/\,C\,(\Omega-\Omega_c)^{3/2}$")
ax.set_xlim(1e-7, 3e-2)
ax.set_ylim(0.972, 1.052)
st.legend(ax, loc="lower left")
st.tight(fig)
st.save(fig, ROOT / "figures" / "fig2_cusp_scaling.pdf")
print(f"  Omega_c(analytic) = {Om_c:.6f}, plateau = {plateau:.4f}")
print(f"  n_HB = {len(eps_hb)}, HB/plateau in "
      f"[{(dF_hb/eps_hb**1.5/plateau).min():.3f}, "
      f"{(dF_hb/eps_hb**1.5/plateau).max():.3f}]")
