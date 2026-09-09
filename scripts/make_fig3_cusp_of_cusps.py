"""Fig. 5: cusp-pair separation at the beaks point, 2-DOF closed form vs full HB.

Rebuilt to use data/beaks_tips_refined.csv -- upper-pair tips located by the
(Delta F)^{2/3} intercept (scripts/cusp_tip_intercept.py) with the fit window
refined until the intercept stabilises.  The previous version of this figure
took its HB points from lips_scaling_hb.csv, a coarse jump-detection scan on a
5e-3 grid in F, and never drew the next-order curve its caption described.

Panel (a) shows Delta Omega against mu; on that scale every candidate law
looks alike.  Panel (b) divides out the square root and is where the
comparison actually lives: it plots the prefactor c = Delta Omega / sqrt(mu)
against the analytic c0 + c1 mu.

mu is referred to kappa*_HB = 0.119322, the projection fold located directly
in the full HB system by topological bisection (Sec. IV.C).  That
determination assumes no exponent, which matters here: extrapolating the
square-root law through the two couplings nearest the birth instead returns a
threshold low by 2.4e-4, and referring the figure to it would have made the
near-threshold points appear to sit on the prediction when they do not.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _prx_style as st  # noqa: F401  (applies rcParams on import)

ROOT = Path(__file__).parent.parent

KAPPA_STAR_MODEL = 0.1183654      # reduced-model beaks point
KAPPA_STAR_HB = 0.119322          # directly located, Eq. (kstar_hb)
# Analytic LO and finite-mu coefficients, from normal_form_cstar.py.
# C1 was 0.30906 here until 2026-08-15 -- the superseded value, wrong by 0.95%,
# which this figure's dashed LO+1 curve was still being drawn from while
# Eq. (cstar_finite_mu), tab:beaks_hb and the caption all carried 0.31199.
C0, C1 = 0.16216, 0.31199

with open(ROOT / "data" / "beaks_tips_refined.csv") as fh:
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
rows.sort(key=lambda r: r["kappa"])

kappa = np.array([r["kappa"] for r in rows])
dOm_hb = np.array([r["dOm"] for r in rows])
dOm_2d = np.array([r["dOm_LO"] for r in rows])    # closed-form locus

# Each dataset is referred to its OWN threshold: the closed-form locus to the
# reduced-model kappa*, the HB tips to the measured kappa*_HB.  Mixing them
# would put a spurious divergence into c_* at small mu.
mu = kappa - KAPPA_STAR_HB
mu_2d = kappa - KAPPA_STAR_MODEL
c_hb = dOm_hb / np.sqrt(mu)
c_2d = dOm_2d / np.sqrt(mu_2d)

mu_ref = np.linspace(1e-5, mu.max() * 1.05, 400)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.75))

# The closed-form locus and the HB tips very nearly coincide, so equal-sized
# markers hide one another entirely.  Draw the prediction as a large open ring
# first and the measurement as a smaller solid square on top, so both read.
MK_2D = dict(marker="o", ls="", ms=5.4, mfc="none", mew=0.9,
             color=st.C_ANALYTIC)
MK_HB = dict(marker="s", ls="", ms=2.9, mfc=st.C_HB, mew=0.5,
             mec="white", color=st.C_HB)

# ---------------------------------------------------------------- panel (a)
ax.plot(mu_ref, C0 * np.sqrt(mu_ref), "-", color=st.C_ANALYTIC, lw=1.2,
        label=r"LO: $0.16216\,\sqrt{\mu}$")
ax.plot(mu_ref, (C0 + C1 * mu_ref) * np.sqrt(mu_ref), "--",
        color=st.C_ANALYTIC, lw=1.2, label=r"LO$+1$")
ax.plot(mu_2d, dOm_2d, label="closed-form locus", **MK_2D)
ax.plot(mu, dOm_hb, label="full HB", **MK_HB)
ax.set_xlabel(r"$\mu$ (each referred to its own $\kappa^*$)")
ax.set_ylabel(r"$\Delta\Omega$")
ax.set_xlim(0, mu.max() * 1.05)
ax.set_ylim(0, None)
st.legend(ax, loc="upper left")
st.panel(ax, "a")

# ---------------------------------------------------------------- panel (b)
bx.plot(mu_ref, np.full_like(mu_ref, C0), "-", color=st.C_ANALYTIC, lw=1.2,
        label="LO")
bx.plot(mu_ref, C0 + C1 * mu_ref, "--", color=st.C_ANALYTIC, lw=1.2,
        label=r"LO$+1$")
bx.plot(mu_2d, c_2d, label="closed-form locus", **MK_2D)
bx.plot(mu, c_hb, label="full HB", **MK_HB)
bx.set_xlabel(r"$\mu$ (each referred to its own $\kappa^*$)")
bx.set_ylabel(r"$c_* = \Delta\Omega/\sqrt{\mu}$")
bx.set_xlim(0, mu.max() * 1.05)
bx.set_ylim(0.157, None)
st.legend(bx, loc="upper left")
st.panel(bx, "b")

st.tight(fig)
st.save(fig, ROOT / "figures" / "fig3_cusp_of_cusps.pdf")
print(f"  n_HB = {len(mu)}, mu in [{mu.min():.5f}, {mu.max():.5f}]")
print(f"  c_HB in [{c_hb.min():.4f}, {c_hb.max():.4f}]")
resid = (c_hb - (C0 + C1 * mu)) / (C0 + C1 * mu)
print(f"  residual vs LO+1: {resid.min():+.2%} to {resid.max():+.2%}")
