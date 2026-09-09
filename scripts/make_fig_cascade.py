"""Fig (mechanism): linear response governs the ladder of cusp-pair births.
Two stacked panels sharing the Omega axis, for a 4-resonator hub:
 (a) receptance |G(Om)| = |[Z^-1]_11| at a coupling above all births, poles marked;
 (b) cusp locus (Om where rho^2=3 sigma^2, rho<0) vs kappa; each satellite pole
     spawns a cusp pair at its onset kappa*, the count climbing 1->3->5->7.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import _prx_style as st

ROOT = Path(__file__).parent.parent
OMS = [1.0, 1.25, 1.5, 1.75]
ZS = [0.015, 0.02, 0.02, 0.02]
N = len(OMS)
e1 = np.eye(N)[:, 0]


def K_hub(kappa):
    K = np.diag(np.asarray(OMS, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += kappa; K[j, j] += kappa
        K[0, j] -= kappa; K[j, 0] -= kappa
    return K


def G11(Om, kappa):
    Z = K_hub(kappa) - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in ZS])
    return np.linalg.solve(Z, e1)[0]


def f(Om, kappa):
    inv = 1.0 / G11(Om, kappa)
    return inv.real ** 2 - 3 * inv.imag ** 2


def rho(Om, kappa):
    return (1.0 / G11(Om, kappa)).real


def poles(kappa):
    return np.sqrt(np.sort(np.linalg.eigvalsh(K_hub(kappa))))


def cusp_points(kappa, lo=0.9, hi=2.2, n=3000):
    oms = np.linspace(lo, hi, n)
    fv = np.array([f(o, kappa) for o in oms])
    pts = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            o = brentq(lambda x: f(x, kappa), oms[i], oms[i + 1], xtol=1e-9)
            if rho(o, kappa) < 0:
                pts.append(o)
    return pts


# ---- compute cusp locus over kappa
KDISP = 0.32                       # display coupling (above all births)
ks = np.linspace(0.02, 0.34, 110)
locus = [(k, o) for k in ks for o in cusp_points(k)]
lk = np.array([p[0] for p in locus]); lo = np.array([p[1] for p in locus])

# detect birth kappas (count jumps) for star markers + count annotation
counts = np.array([len(cusp_points(k)) for k in ks])
births = []
for j in range(len(ks) - 1):
    if counts[j + 1] > counts[j]:
        klo, khi, c0 = ks[j], ks[j + 1], counts[j]
        for _ in range(36):
            m = 0.5 * (klo + khi)
            if len(cusp_points(m)) > c0: khi = m
            else: klo = m
        kstar = 0.5 * (klo + khi)
        below = cusp_points(kstar - 6e-4)
        above = cusp_points(kstar + 6e-4)
        newpts = [o for o in above
                  if min([abs(o - b) for b in below], default=9) > 8e-3]
        births.append((kstar, newpts))

# ---- plot
fig, (axr, axc) = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.45]})
OM_LO, OM_HI = 0.95, 2.05
oms = np.linspace(OM_LO, OM_HI, 1400)
axr.semilogy(oms, np.abs([G11(o, KDISP) for o in oms]), color="#333333",
             lw=1.1)
for p in poles(KDISP):
    axr.axvline(p, color=st.LIGHTGREY, ls=":", lw=0.7)
axr.set_ylabel(r"$|G(\Omega)|$")
st.panel(axr, "a")
axr.text(0.99, 0.93, f"hub $N={N}$, $\\kappa={KDISP}$",
         transform=axr.transAxes, va="top", ha="right", fontsize=7)

# The cusp condition is a statement about arg G, not |G|: overlay the phase
# and the -150 deg level that selects the cusps, so this panel shows the
# quantity the locus below is actually a level set of.
axp = axr.twinx()
argG = np.degrees(np.angle([G11(o, KDISP) for o in oms]))
axp.plot(oms, argG, color=st.C_HB, lw=1.0)
axp.axhline(-150.0, color=st.C_HB, ls="--", lw=0.8)
axp.set_ylabel(r"$\arg G$ (deg)", color=st.C_HB)
axp.tick_params(axis="y", colors=st.C_HB)
axp.set_ylim(-205, 25)
axp.set_yticks([-180, -150, -90, 0])
axp.spines["right"].set_color(st.C_HB)

axc.plot(lo, lk, ".", ms=1.6, color=st.C_ANALYTIC, rasterized=True)
for kstar, newpts in births:
    for o in newpts:
        axc.plot(o, kstar, marker="*", ms=7.5, color="black", mec="white",
                 mew=0.4, ls="", zorder=6)
st.panel(axc, "b")
# cusp count, anchored to the right frame rather than floating past it
for c, ky in [(1, 0.045), (3, 0.125), (5, 0.198), (7, 0.295)]:
    if ky < ks.max():
        axc.text(0.985, ky, f"{c}", transform=axc.get_yaxis_transform(),
                 fontsize=7.5, color=st.GREY, va="center", ha="right")
axc.text(0.985, 0.325, "cusps", transform=axc.get_yaxis_transform(),
         fontsize=6.8, color=st.GREY, va="center", ha="right", style="italic")
axc.set_xlabel(r"frequency $\Omega$")
axc.set_ylabel(r"coupling $\kappa$")
axc.set_xlim(OM_LO, OM_HI)
axc.set_ylim(0.02, 0.34)

st.tight(fig)
out = ROOT / "figures" / "fig_cascade.pdf"
plt.savefig(out)
print(f"wrote {out}; births at kappa*={[round(b[0],3) for b in births]}, "
      f"counts {counts.min()}..{counts.max()}")
