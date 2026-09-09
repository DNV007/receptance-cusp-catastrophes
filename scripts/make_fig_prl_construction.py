"""PRL Fig. 1 -- the construction, left to right.

    network  ->  G sampled on the harmonic comb  ->  arg G = const contour
             ->  tangency of that contour  ->  beaks or lips.

Built from make_fig0_phase_contour.py, with one panel added at the front:
the reduction itself, which is what makes the rest mean anything.

TWO DIFFERENCES FROM fig0_phase_contour.py:

  * SELF-CONTAINED. The original imports recompute_beaks_table, which pulls
    in cusp_tip_intercept -> jax, while also needing scipy.brentq. This
    environment cannot import jax and scipy together (see the project memory
    on the user-site numpy/scipy clash), so the receptance and the root
    finding are reimplemented here in numpy alone.

  * EXACT RECEPTANCE e_1^T Z^{-1} e_1 throughout, not the modal
    (proportional-damping) form the original uses. Fig. 2 is already exact,
    and mixing the two forms across figures of one Letter is what produced
    the tab:typecriterion inconsistency in the long paper. The beaks point
    therefore sits at (Om*, kap*) = (1.30088, 0.11896) rather than the modal
    (1.30100, 0.11837).

Output: figures/fig_prl_construction.pdf
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL = -150.0
W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.02
CMAT = np.diag([2 * Z1, 2 * Z2])
KAP_STAR = 0.11896          # exact-receptance beaks point
OM_STAR = 1.30088


def Gof(Om, kap):
    K = np.array([[W1 ** 2 + kap, -kap], [-kap, W2 ** 2 + kap]])
    Z = K - Om ** 2 * np.eye(2) + 1j * Om * CMAT
    return np.linalg.inv(Z)[0, 0]


def argG(Om, kap):
    return np.degrees(np.angle(Gof(Om, kap)))


def fdisc(Om, kap):
    inv = 1.0 / Gof(Om, kap)
    return inv.real ** 2 - 3.0 * inv.imag ** 2


def cusps(kap, lo=1.24, hi=1.40, n=4000):
    """Roots of f with the physical branch rho < 0, by scan + bisection."""
    xs = np.linspace(lo, hi, n)
    vs = np.array([fdisc(x, kap) for x in xs])
    out = []
    for i in np.where(np.diff(np.sign(vs)))[0]:
        a, b = xs[i], xs[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(fdisc(a, kap)) == np.sign(fdisc(m, kap)):
                a = m
            else:
                b = m
        r = 0.5 * (a + b)
        if (1.0 / Gof(r, kap)).real < 0:
            out.append(r)
    return sorted(out)


fig = plt.figure(figsize=(7.0, 3.9))
gs = GridSpec(2, 4, figure=fig, height_ratios=[0.82, 1.0],
              width_ratios=[1.18, 1.06, 1.0, 0.60])
axa = fig.add_subplot(gs[0, :])

# ---------------------------------------------------------------- (a) comb
# Panel (a) is about the REDUCTION and uses the lower-cusp operating point
# (kappa = 0.10, Om_c = 1.0673), where the fundamental sits squarely on a
# pole and the comb structure is legible; it is also the benchmark quoted in
# the text. Panels (b)-(e) are about the CONSTRUCTION and use the beaks
# birth. The birth happens to sit between an antiresonance at 1.2967 and a
# pole at 1.3060, which nearly cancel, so |G| there is only 1.67 and the
# comb ratio is 4.2e-2 rather than 7.3e-3 -- the ratio is a property of the
# operating point, not a constant, and the caption says so.
KAP_A, OM_A = 0.10, 1.0673142
oms = np.logspace(np.log10(0.45), np.log10(7.5), 1400)
axa.semilogy(oms, [abs(Gof(o, KAP_A)) for o in oms], "-",
             color=st.GREY, lw=1.2)
for n, col in [(1, st.C_AT), (3, st.C_HB), (5, st.C_HB)]:
    x = n * OM_A
    if x > oms[-1]:
        continue
    y = abs(Gof(x, KAP_A))
    axa.plot([x, x], [1e-4, y], "-", color=col, lw=0.9,
             alpha=0.55 if n > 1 else 1.0)
    axa.plot([x], [y], "o", ms=4.2, mfc=col, mec="white", mew=0.6, zorder=6)
    axa.text(x, y * 2.6, rf"$G({'' if n == 1 else n}\Omega)$", fontsize=7,
             color=col, ha="center")
axa.set_xscale("log")
axa.set_xlim(0.45, 7.5)
axa.set_ylim(1e-3, 1e2)
axa.set_xlabel(r"$\Omega$")
axa.set_ylabel(r"$|G|$")
axa.set_xticks([0.5, 1, 2, 3, 5, 7])
axa.set_xticklabels(["0.5", "1", "2", "3", "5", "7"])
rat = abs(Gof(3 * OM_A, KAP_A) / Gof(OM_A, KAP_A))
axa.text(0.975, 0.90,
         rf"$|G(3\Omega)/G(\Omega)|={rat*1e3:.1f}\times10^{{-3}}$",
         transform=axa.transAxes, fontsize=7, ha="right", va="top",
         color=st.C_HB)
st.panel(axa, "a")

# network cartoon, inset at the left of (a)
ins = axa.inset_axes([0.030, 0.13, 0.24, 0.62])
ins.set_axis_off()
ins.set_xlim(-0.2, 3.4); ins.set_ylim(-0.75, 0.95)
pos = [(0.55, 0.0), (1.65, 0.42), (1.65, -0.42), (2.75, 0.0)]
for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    ins.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], "-",
             color="0.55", lw=0.8, zorder=1)
for k, (x, y) in enumerate(pos):
    ins.plot([x], [y], "o", ms=7 if k == 0 else 5.5,
             mfc=st.C_AT if k == 0 else "white", mec="0.35", mew=0.8, zorder=3)
ins.annotate("", xy=(0.42, 0.0), xytext=(-0.15, 0.0),
             arrowprops=dict(arrowstyle="-|>", lw=0.9, color=st.C_HB))
ins.text(0.55, -0.52, r"$\beta_1 x_1^3$", fontsize=6.5, color=st.C_AT,
         ha="center")
ins.text(1.7, 0.80, "linear", fontsize=6.5, color="0.45", ha="center")

# ---------------------------------------------------------------- (b) phase
bx = fig.add_subplot(gs[1, 0])
oms_b = np.linspace(1.265, 1.345, 700)
styles = [(0.100, st.C_BEFORE, r"$\kappa<\kappa^*$", 1.283, -167.5, "top"),
          (KAP_STAR, st.C_AT, r"$\kappa=\kappa^*$", 1.322, -159.5, "top"),
          (0.150, st.C_AFTER, r"$\kappa>\kappa^*$", 1.310, -134.0, "bottom")]
for kap, col, lab, tx, ty, va in styles:
    bx.plot(oms_b, [argG(o, kap) for o in oms_b], "-", color=col, lw=1.4)
    bx.text(tx, ty, lab, color=col, fontsize=7, ha="center", va=va)
bx.axhline(LEVEL, color=st.GREY, ls="--", lw=0.9, zorder=1)
bx.text(1.3435, LEVEL - 1.2, r"$-150^\circ$", fontsize=6.8, color=st.GREY,
        ha="right", va="top")
bx.plot([OM_STAR], [LEVEL], "*", ms=9, color=st.C_AT, mec="white", mew=0.6,
        zorder=6)
for c in cusps(0.150):
    bx.plot([c], [LEVEL], "o", ms=4.0, color=st.C_AFTER, mec="white", mew=0.6,
            zorder=6)
bx.set_xlabel(r"$\Omega$")
bx.set_ylabel(r"$\arg G$ (deg)")
bx.set_ylim(-172, -130)
bx.set_xlim(oms_b[0], oms_b[-1])
st.panel(bx, "b")

# ---------------------------------------------------------------- (c) locus
cx = fig.add_subplot(gs[1, 1])
kaps = np.linspace(KAP_STAR - 0.004, 0.175, 150)
lo, hi = [], []
for k in kaps:
    cs = cusps(k)
    lo.append(cs[0] if len(cs) >= 2 else np.nan)
    hi.append(cs[-1] if len(cs) >= 2 else np.nan)
cx.plot(lo, kaps, "-", color=st.C_AT, lw=1.4)
cx.plot(hi, kaps, "-", color=st.C_AT, lw=1.4)
cx.plot([OM_STAR], [KAP_STAR], "*", ms=9, color=st.C_AT, mec="white", mew=0.6,
        zorder=6)
cx.annotate(r"$\partial_\Omega f=0$", xy=(OM_STAR, KAP_STAR),
            xytext=(1.318, 0.1265), fontsize=7, color=st.GREY,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=st.GREY,
                            shrinkA=1, shrinkB=3))
cx.text(1.352, 0.170, "two cusps", fontsize=7, ha="right", color=st.C_AT)
cx.text(1.352, 0.1205, "none", fontsize=7, ha="right", color=st.GREY)
cx.set_xlabel(r"$\Omega$")
cx.set_ylabel(r"coupling $\kappa$")
cx.set_xlim(1.265, 1.355)
cx.set_ylim(kaps[0], kaps[-1])
st.panel(cx, "c")

# ------------------------------------------------- (d),(e) critical sets
d = np.linspace(-1.35, 1.35, 800)
STAGES = ((0.55, st.C_BEFORE), (0.0, st.C_AT), (-0.55, st.C_AFTER))


def _runs(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)


dx = fig.add_subplot(gs[1, 2])
for s, col in STAGES:
    u2 = (s + d ** 2) / 9.0            # indefinite: a_Omega > 0
    for r in _runs(u2 >= 0):
        dx.plot(d[r], np.sqrt(u2[r]), color=col, lw=1.3)
        dx.plot(d[r], -np.sqrt(u2[r]), color=col, lw=1.3)
dx.axhline(0, color="0.90", lw=0.6, zorder=0)
dx.set_xlabel(r"$\delta\Omega$")
dx.set_ylabel(r"$\delta u$")
dx.set_xlim(-1.38, 1.38); dx.set_ylim(-0.66, 0.66)
dx.text(0.04, 0.07, r"$a_\Omega>0$", transform=dx.transAxes, fontsize=7.5)
dx.text(0.5, 0.93, "beaks", transform=dx.transAxes, fontsize=7.5,
        ha="center", va="top", color=st.C_HB)
st.panel(dx, "d")

ex = fig.add_subplot(gs[1, 3])
for s, col in STAGES:
    u2 = (-s - d ** 2) / 9.0           # definite: a_Omega < 0
    for r in _runs(u2 >= 0):
        if r.size > 1:
            ex.plot(d[r], np.sqrt(u2[r]), color=col, lw=1.2)
            ex.plot(d[r], -np.sqrt(u2[r]), color=col, lw=1.2)
        else:
            ex.plot(d[r], [0.0], ".", color=col, ms=3.2)
ex.axhline(0, color="0.90", lw=0.6, zorder=0)
ex.set_xlim(-1.05, 1.05); ex.set_ylim(-0.66, 0.66)
ex.set_xlabel(r"$\delta\Omega$")
ex.set_yticklabels([])
ex.text(0.07, 0.07, r"$a_\Omega<0$", transform=ex.transAxes, fontsize=7.5)
ex.text(0.5, 0.93, "lips", transform=ex.transAxes, fontsize=7.5,
        ha="center", va="top", color=st.C_ANALYTIC)
st.panel(ex, "e")

fig.tight_layout(pad=0.3, w_pad=0.7, h_pad=0.9)
out = os.path.join(ROOT, "figures", "fig_prl_construction.pdf")
plt.savefig(out)
print("wrote", out)
print(f"  beaks point (exact G): Om*={OM_STAR}, kappa*={KAP_STAR}")
print(f"  |G(3Om)/G(Om)| at the birth = {rat:.4e}")
print(f"  cusps at kappa=0.150: {['%.5f' % c for c in cusps(0.150)]}")
