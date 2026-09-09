"""Cusp locus in (Omega, kappa) at beta_1 = 0.147.

Fixes to the previous version: the unframed legend sat directly on the lower
branch and read as if the curve were struck through; the baseline marker was
labelled "v8 baseline", an internal repository tag with no meaning to a
reader; and the axes extended well beyond the traced locus.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _prx_style as st

ROOT = Path(__file__).parent.parent

with (ROOT / "data" / "two_dof_cusp_locus.csv").open() as fh:
    rows = [r for r in csv.DictReader(fh)
            if abs(float(r["beta_1"]) - 0.147) < 1e-3]


def branch(b):
    r = sorted((x for x in rows if int(float(x["branch"])) == b),
               key=lambda x: float(x["kappa"]))
    return (np.array([float(x["omega_cusp"]) for x in r]),
            np.array([float(x["kappa"]) for x in r]))


OMEGA_STAR, KAPPA_STAR = 1.30100, 0.118365

fig, ax = plt.subplots(figsize=(3.4, 2.55))

om0, k0 = branch(0)
om1, k1 = branch(1)
om2, k2 = branch(2)

ax.plot(om0, k0, "-", color=st.C_ANALYTIC, label="lower cusp")
ax.plot(om1, k1, "-", color=st.C_HB, label="upper pair")
ax.plot(om2, k2, "-", color=st.C_HB)

ax.plot([OMEGA_STAR], [KAPPA_STAR], marker="*", ms=9, color="black",
        mec="white", mew=0.5, ls="", zorder=6,
        label=r"birth $(\Omega^*,\kappa^*)$")
ax.plot([1.0672], [0.10], marker="o", ms=4.5, color="black", mfc="white",
        mew=0.9, ls="", zorder=6, label="baseline cusp")

ax.set_xlabel(r"cusp frequency $\Omega_c$")
ax.set_ylabel(r"coupling $\kappa$")
ax.set_xlim(1.03, 1.45)
ax.set_ylim(0.04, 0.30)
# The lower branch runs up the left edge and the upper pair up the right, so
# the only region free of data is the lower middle.
st.legend(ax, loc="lower center", bbox_to_anchor=(0.52, 0.0), ncol=2,
          columnspacing=1.0)
st.tight(fig)
st.save(fig, ROOT / "figures" / "fig1_cusp_locus.pdf")
