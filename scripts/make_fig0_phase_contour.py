"""Fig. 1 (conceptual): the receptance-phase construction.

The paper's logic is

    G(Omega, kappa)  ->  arg G = const contour  ->  stationary point of that
    contour  ->  beaks or lips,

and until now no figure showed it; the opening figure was a conventional
Duffing bistability wedge, which validates the solver rather than displaying
the idea.  This figure carries the construction left to right.

(a) arg G(Omega) at three couplings, against the fixed level -150 deg.  Below
    the birth the phase never reaches the level; at kappa* it is tangent to
    it; above, it crosses twice and two cusps exist.  The cusp-pair birth is
    a tangency of the linear receptance phase with a fixed level.
(b) The same statement in the (Omega, kappa) plane: the locus arg G = -150
    deg is the cusp locus, and the birth is where it folds under projection
    onto kappa.
(c) The local critical set 9 du^2 - a_Om dOm^2 = b_kap dkap.  Indefinite
    (a_Om > 0) gives two branches that reconnect, leaving a gap in Omega:
    beaks.  Definite (a_Om < 0) gives an isolated oval: lips.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st  # noqa: F401
from recompute_beaks_table import transfer, closed_form_cusps, KAP_STAR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL = -150.0


def argG(Om, kap):
    rho, sig = transfer(Om, kap)
    return np.degrees(np.angle(1.0 / (rho + 1j * sig)))


fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6),
                         gridspec_kw={"width_ratios": [1.18, 1.06, 1.0, 0.60]})

# ------------------------------------------------------------------ (a)
ax = axes[0]
oms = np.linspace(1.283, 1.352, 700)
styles = [(0.100, st.C_BEFORE, r"$\kappa<\kappa^*$", 1.3010, -166.5, "top"),
          (KAP_STAR, st.C_AT, r"$\kappa=\kappa^*$", 1.3265, -158.0, "top"),
          (0.150, st.C_AFTER, r"$\kappa>\kappa^*$", 1.3175, -134.0, "bottom")]
for kap, col, lab, tx, ty, va in styles:
    ax.plot(oms, [argG(o, kap) for o in oms], "-", color=col, lw=1.4)
    # inline labels: a three-entry legend had to sit on the curves
    ax.text(tx, ty, lab, color=col, fontsize=7, ha="center", va=va)
ax.axhline(LEVEL, color=st.GREY, ls="--", lw=0.9, zorder=1)
ax.text(1.3505, LEVEL - 1.2, r"$-150^\circ$", fontsize=6.8, color=st.GREY,
        ha="right", va="top")
cs_star = [c for c in closed_form_cusps(KAP_STAR) if c > 1.2]
if cs_star:
    ax.plot([np.mean(cs_star)], [LEVEL], "*", ms=9, color=st.C_AT,
            mec="white", mew=0.6, zorder=6)
for c in [c for c in closed_form_cusps(0.150) if c > 1.2]:
    ax.plot([c], [LEVEL], "o", ms=4.0, color=st.C_AFTER, mec="white", mew=0.6,
            zorder=6)
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\arg G$ (deg)")
ax.set_ylim(-172, -130)
ax.set_xlim(oms[0], oms[-1])
st.panel(ax, "a")

# ------------------------------------------------------------------ (b)
bx = axes[1]
kaps = np.linspace(KAP_STAR - 0.004, 0.175, 160)
lo, hi = [], []
for k in kaps:
    # coarser scan than the default: this is a locus curve for display, and
    # each root is still polished by brentq inside closed_form_cusps.
    cs = [c for c in closed_form_cusps(k, lo=1.24, hi=1.40, n=4000)
          if c > 1.2]
    lo.append(cs[0] if len(cs) >= 2 else np.nan)
    hi.append(cs[1] if len(cs) >= 2 else np.nan)
bx.plot(lo, kaps, "-", color=st.C_AT, lw=1.4)
bx.plot(hi, kaps, "-", color=st.C_AT, lw=1.4)
bx.plot([np.nanmin(lo)], [KAP_STAR], "*", ms=9, color=st.C_AT,
        mec="white", mew=0.6, zorder=6)
bx.annotate(r"$\partial_\Omega f=0$", xy=(np.nanmin(lo), KAP_STAR),
            xytext=(1.316, 0.1245), fontsize=7, color=st.GREY,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=st.GREY,
                            shrinkA=1, shrinkB=3))
bx.text(1.348, 0.169, "two cusps", fontsize=7, ha="right", color=st.C_AT)
bx.text(1.348, 0.1195, "none", fontsize=7, ha="right", color=st.GREY)
bx.set_xlabel(r"$\Omega$")
bx.set_ylabel(r"coupling $\kappa$")
bx.set_xlim(1.283, 1.352)
bx.set_ylim(kaps[0], kaps[-1])
st.panel(bx, "b")

# ------------------------------------------------------------------ (c)
cx = axes[2]
d = np.linspace(-1.35, 1.35, 800)
# Colours track panel (a): light = before the birth, dark = at it, red = after.
STAGES = ((0.55, st.C_BEFORE), (0.0, st.C_AT), (-0.55, st.C_AFTER))


def _runs(mask):
    """Contiguous True runs, so disjoint branches are not joined by a chord."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)


for s, col in STAGES:
    # indefinite form, a_Omega > 0:  9 du^2 - dOm^2 = s
    u2 = (s + d ** 2) / 9.0
    for r in _runs(u2 >= 0):
        cx.plot(d[r], np.sqrt(u2[r]), color=col, lw=1.3)
        cx.plot(d[r], -np.sqrt(u2[r]), color=col, lw=1.3)
cx.axhline(0, color="0.90", lw=0.6, zorder=0)
cx.set_xlabel(r"$\delta\Omega$")
cx.set_ylabel(r"$\delta u$")
cx.set_xlim(-1.38, 1.38)
cx.set_ylim(-0.66, 0.66)
st.panel(cx, "c")
# the descriptive sentence lives in the caption; only the discriminating
# sign is marked on the axes, so nothing sits on the curves
cx.text(0.04, 0.07, r"$a_\Omega>0$", transform=cx.transAxes, fontsize=7.5)

# ------------------------------------------------------------------ (d)
# The definite case had been an inset over panel (c), where its white
# background occluded one of the branches; it gets its own narrow panel.
ix = axes[3]
for s, col in STAGES:
    # definite form, a_Omega < 0:  9 du^2 + dOm^2 = -s
    u2 = (-s - d ** 2) / 9.0
    for r in _runs(u2 >= 0):
        if r.size > 1:
            ix.plot(d[r], np.sqrt(u2[r]), color=col, lw=1.2)
            ix.plot(d[r], -np.sqrt(u2[r]), color=col, lw=1.2)
        else:
            ix.plot(d[r], [0.0], ".", color=col, ms=3.2)
ix.axhline(0, color="0.90", lw=0.6, zorder=0)
ix.set_xlim(-1.05, 1.05)
ix.set_ylim(-0.66, 0.66)
ix.set_xlabel(r"$\delta\Omega$")
ix.set_yticklabels([])
st.panel(ix, "d")
ix.text(0.07, 0.07, r"$a_\Omega<0$", transform=ix.transAxes, fontsize=7.5)

st.tight(fig)
st.save(fig, os.path.join(ROOT, "figures", "fig0_phase_contour.pdf"))
