"""Bistability wedge in (Omega, F) at the baseline.

Full-HB fold curves closing at the cusp tip, with the closed-form 2-DOF
prediction overlaid.

Two defects in the previous version are fixed here.  The analytic wedge was
traced over Omega in [1.04, Omega_c), i.e. BELOW the tip, whereas the
hardening-Duffing tongue opens ABOVE it; `folds()` therefore returned None at
every sampled frequency and the shaded region and fold lines were silently
absent from the figure even though both appeared in the legend.  The axes were
also fixed to a range roughly twice the extent of the data, leaving most of
the panel empty.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import _prx_style as st

ROOT = Path(__file__).parent.parent

with (ROOT / "data" / "cusp_both_folds.csv").open() as fh:
    rows = [r for r in csv.DictReader(fh)]


def branch(name):
    r = sorted((x for x in rows if x["branch"] == name),
               key=lambda x: float(x["omega"]))
    return (np.array([float(x["omega"]) for x in r]),
            np.array([float(x["F_fold"]) for x in r]))


om1, F1 = branch("SN1")
om2, F2 = branch("SN2")

# closed-form 2-DOF folds at the baseline
w1, w2, z1, z2, kappa, beta1 = 1.0, 1.25, 0.015, 0.02, 0.10, 0.147
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


def folds(Om):
    a, b = ab(Om)
    disc = 4 * c**2 * (a**2 - 3 * b**2)
    if disc < 0:
        return None
    sd = np.sqrt(disc)
    y1 = (-4 * c * a + sd) / (6 * c**2)
    y2 = (-4 * c * a - sd) / (6 * c**2)
    if y1 <= 0 or y2 <= 0:
        return None
    F1 = np.sqrt(y1 * ((a + c * y1) ** 2 + b**2))
    F2 = np.sqrt(y2 * ((a + c * y2) ** 2 + b**2))
    return min(F1, F2), max(F1, F2)


Om_c = brentq(lambda O: (lambda r, s: r**2 - 3 * s**2)(*ab(O)), 1.04, 1.08)
a_c, _ = ab(Om_c)
F_c = np.sqrt((-2 * a_c / (3 * c)) * (4 * a_c**2 / 9))

# the wedge opens ABOVE the tip
Om_hi = 1.0805
Oms = np.linspace(Om_c + 1e-7, Om_hi, 600)
F_lo, F_hi = [], []
for O in Oms:
    r = folds(O)
    F_lo.append(np.nan if r is None else r[0])
    F_hi.append(np.nan if r is None else r[1])
F_lo, F_hi = np.array(F_lo), np.array(F_hi)

fig, ax = plt.subplots(figsize=(3.4, 2.55))
ax.fill_between(Oms, F_lo, F_hi, color=st.C_ANALYTIC, alpha=0.18, lw=0,
                label="bistable (modal closed form)")
ax.plot(Oms, F_lo, "-", color=st.C_ANALYTIC, lw=1.1)
ax.plot(Oms, F_hi, "-", color=st.C_ANALYTIC, lw=1.1, label="modal closed-form folds")
ax.plot(om1, F1, "o", color=st.C_HB, mfc="none", mew=0.9, ms=3.2,
        label="full HB, lower fold")
ax.plot(om2, F2, "s", color=st.C_HB, mfc="none", mew=0.9, ms=3.2,
        label="full HB, upper fold")
ax.plot([Om_c], [F_c], marker="*", ms=9, color="black", mec="white", mew=0.5,
        ls="", zorder=6, label=r"cusp tip $(\Omega_c,F_c)$")

ax.set_xlabel(r"drive frequency $\Omega$")
ax.set_ylabel(r"drive amplitude $F$")
ax.set_xlim(1.0615, Om_hi)
ax.set_ylim(0.0212, 0.0358)
# five 4-decimal labels run together at this width; use four 3-decimal ones
ax.set_xticks([1.065, 1.070, 1.075, 1.080])
ax.set_xticklabels(["1.065", "1.070", "1.075", "1.080"])
st.legend(ax, loc="upper left")
st.tight(fig)
st.save(fig, ROOT / "figures" / "fig2_wedge.pdf")
