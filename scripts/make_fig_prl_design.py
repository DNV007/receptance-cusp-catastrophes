"""PRL Fig. 2 -- linear response as a design variable.

Reuses the verified fig_maps content (full-HB bistability maps below/at/above
a birth, beaks on top, lips below; data/hb_bistability_maps.csv) and adds a
right-hand DESIGN panel spanning both rows.

The design panel is the inverse step. For the 2-DOF family it sweeps the
auxiliary-mode frequency w2, solves the birth condition

    f = 0,  d_Om f = 0,     f = rho^2 - 3 sigma^2,  rho + i sigma = 1/G

on the EXACT receptance for (Om*, kappa*), and plots the classifier
a_Om = (1/2) d2f/dOm2 against w2. Its sign flips as the auxiliary pole
crosses the driven resonance: pole below -> lips, pole above -> beaks. So the
topology is chosen before any nonlinear computation is attempted.

TWO DELIBERATE CHOICES, both from the 2026-07-21 audit:

  * Everything here is computed from the EXACT receptance e_1^T Z^{-1} e_1,
    not the modal (proportional-damping) closed form. The long paper's
    tab:typecriterion mixes the two: its baseline reproduces exactly from the
    modal form (kappa* = 0.11837, a_Om = +1370 at w2 = 1.25), but the
    w2 = 0.95 row does not close under either form (quoted kappa* = 0.283 vs
    modal 0.28085; quoted a_Om = -103 vs modal -110.2, exact -94.2). The
    classification is robust -- every variant gives a_Om < 0 there by a wide
    margin -- but the row needs regenerating before its numbers are quoted.
    This script therefore marks the lips case WITHOUT quoting kappa*.

  * The marked beaks case is a hub, not a member of the 2-DOF sweep, so it is
    annotated on the a_Om > 0 branch rather than pinned to a w2 value.

Output: figures/fig_prl_design.pdf
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import _prx_style as st

ROOT = Path(__file__).parent.parent
BETA1 = 0.147
C3 = 0.75 * BETA1
W1 = 1.0
ZET = (0.015, 0.02)

with (ROOT / "data" / "hb_bistability_maps.csv").open() as fh:
    _rows = list(csv.DictReader(fh))


# ------------------------------------------------------------ maps (reused)
def hb_widths(config, k):
    by_om = {}
    for r in _rows:
        if r["config"] == config and abs(float(r["kappa"]) - k) < 1e-6:
            by_om.setdefault(float(r["omega"]), []).append(float(r["F_fold"]))
    oms = sorted(o for o, v in by_om.items() if len(v) >= 2)
    return (np.array(oms),
            np.array([max(by_om[o]) - min(by_om[o]) for o in oms]))


def K_hub(om, k):
    N = len(om); K = np.diag(np.asarray(om, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += k; K[j, j] += k; K[0, j] -= k; K[j, 0] -= k
    return K


def K_chain(om, k):
    N = len(om); K = np.diag(np.asarray(om, float) ** 2).copy()
    for i in range(N - 1):
        K[i, i] += k; K[i + 1, i + 1] += k; K[i, i + 1] -= k; K[i + 1, i] -= k
    return K


def rs(om, ze, Om, k, Kf):
    N = len(om)
    Z = Kf(om, k) - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in ze])
    inv = 1.0 / np.linalg.solve(Z, np.eye(N)[:, 0])[0]
    return inv.real, inv.imag


def cf_region(om, ze, k, Kf, band, n=600):
    oms = np.linspace(band[0], band[1], n)
    Flo = np.full(n, np.nan); Fhi = np.full(n, np.nan)
    for i, Om in enumerate(oms):
        r, s = rs(om, ze, Om, k, Kf)
        if r < 0 and r ** 2 > 3 * s ** 2:
            d = np.sqrt(r ** 2 - 3 * s ** 2)
            Fs = []
            for y in [(-2 * r + d) / (3 * C3), (-2 * r - d) / (3 * C3)]:
                if y > 0:
                    F2 = y * ((r + C3 * y) ** 2 + s ** 2)
                    if F2 > 0:
                        Fs.append(np.sqrt(F2))
            if len(Fs) == 2:
                Flo[i], Fhi[i] = min(Fs), max(Fs)
    return oms, Flo, Fhi


# ------------------------------------------------------- design panel (new)
def f_disc(Om, w2, k):
    Z = (K_chain([W1, w2], k) - Om ** 2 * np.eye(2)
         + 1j * Om * np.diag([2 * ZET[0], 2 * ZET[1]]))
    inv = 1.0 / np.linalg.inv(Z)[0, 0]
    return inv.real ** 2 - 3.0 * inv.imag ** 2


def dO(Om, w2, k, h=1e-6):
    return (f_disc(Om + h, w2, k) - f_disc(Om - h, w2, k)) / (2 * h)


def a_omega(Om, w2, k, h=1e-4):
    """a_Om = (1/2) d2f/dOm2, matching the long paper's definition."""
    return 0.5 * (f_disc(Om + h, w2, k) - 2 * f_disc(Om, w2, k)
                  + f_disc(Om - h, w2, k)) / h ** 2


def birth(w2, guess):
    v = np.array(guess, float)
    for _ in range(300):
        Om, k = v
        r = np.array([f_disc(Om, w2, k), dO(Om, w2, k)])
        if np.linalg.norm(r) < 1e-14:
            break
        J = np.zeros((2, 2))
        for j, h in enumerate([1e-7, 1e-7]):
            vp, vm = v.copy(), v.copy(); vp[j] += h; vm[j] -= h
            J[0, j] = (f_disc(vp[0], w2, vp[1]) - f_disc(vm[0], w2, vm[1])) / (2 * h)
            J[1, j] = (dO(vp[0], w2, vp[1]) - dO(vm[0], w2, vm[1])) / (2 * h)
        try:
            v = v + np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None
    Om, k = v
    if not (np.isfinite(Om) and np.isfinite(k)) or k <= 0 or k > 1.5:
        return None
    if abs(f_disc(Om, w2, k)) + abs(dO(Om, w2, k)) > 1e-8:
        return None
    # physical branch for a hardening cubic: beta1 * rho < 0
    Z = (K_chain([W1, w2], k) - Om ** 2 * np.eye(2)
         + 1j * Om * np.diag([2 * ZET[0], 2 * ZET[1]]))
    if (1.0 / np.linalg.inv(Z)[0, 0]).real >= 0:
        return None
    return Om, k, a_omega(Om, w2, k)


ROWS = [
    ("beaks", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02], K_hub,
     [0.05, 0.10, 0.14], (1.11, 1.28)),
    ("lips", [1.0, 0.95], [0.015, 0.02], K_chain,
     [0.263, 0.303, 0.333], (1.00, 1.09)),
]
LABELS = {0.05: r"$\kappa<\kappa^\ast$", 0.10: r"$\kappa\gtrsim\kappa^\ast$",
          0.14: r"$\kappa>\kappa^\ast$", 0.263: r"$\kappa<\kappa^\ast$",
          0.303: r"$\kappa\gtrsim\kappa^\ast$", 0.333: r"$\kappa>\kappa^\ast$"}

fig = plt.figure(figsize=(7.0, 4.0))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.42])
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                 for r in range(2)])
axd = fig.add_subplot(gs[:, 3])

_tops = {}
for ri, (lab, om, ze, Kf, kappas, band) in enumerate(ROWS):
    for ci, k in enumerate(kappas):
        _, Flo, Fhi = cf_region(om, ze, k, Kf, band)
        dF = Fhi - Flo
        _, hdF = hb_widths(lab, k)
        _tops[(ri, ci)] = float(np.nanmax([
            np.nanmax(dF) if np.any(~np.isnan(dF)) else 0.0,
            hdF.max() if hdF.size else 0.0]))
_rowmax = {ri: max(_tops[(ri, c)] for c in range(3)) for ri in (0, 1)}

for ri, (lab, om, ze, Kf, kappas, band) in enumerate(ROWS):
    for ci, k in enumerate(kappas):
        ax = axes[ri, ci]
        oms, Flo, Fhi = cf_region(om, ze, k, Kf, band)
        dF = Fhi - Flo
        ax.fill_between(oms, 0.0, dF, where=~np.isnan(dF),
                        color=st.C_ANALYTIC, alpha=0.25, lw=0,
                        label="from $G$")
        how, hdF = hb_widths(lab, k)
        ax.plot(how, hdF, "o", ms=2.4, mfc=st.C_HB, mec="white", mew=0.4,
                ls="", label="full HB")
        ax.set_xlim(*band)
        ax.set_ylim(0, (_tops[(ri, ci)] or _rowmax[ri]) * 1.30)
        ax.text(0.04, 0.93, f"{LABELS[k]}\n$\\kappa={k}$",
                transform=ax.transAxes, va="top", fontsize=7)
        if len(how) == 0 and not np.any(~np.isnan(dF)):
            ax.text(0.5, 0.45, "no bistability", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="0.45",
                    style="italic")
        if ci == 0:
            ax.set_ylabel(r"tongue width $\Delta F$")
        if ri == 1:
            ax.set_xlabel(r"frequency $\Omega$")
st.legend(axes[0, 2], loc="upper right", fontsize=6.5, handlelength=1.0)

# ---- design panel: the chart, read Om* -> the network -------------------
# REPLACES the earlier a_Omega-vs-w2 sweep. Two reasons:
#
#   * That sweep warm-started from w2 = 0.995 and 1.005 and swept outward.
#     It loses the branch near degeneracy, and the gap between the two runs
#     was labelled "no birth". It is not: seeding Newton from a grid at each
#     w2 (inverse_design_branches.py) finds a birth at EVERY w2 in
#     [0.60, 1.60]. What actually happens near w2 ~ 1.05 is that the lips and
#     beaks branches overlap and a_Omega passes through zero, so the
#     leading-order classification degenerates -- the birth does not vanish.
#
#   * The Letter's claim is the backward direction. Plotting the network
#     against the target frequency states it directly: pick Om*, read off
#     (w2, kappa*).
BR = ROOT / "data" / "inverse_design_branches.csv"
_rows = []
with open(BR) as fh:
    for r in csv.DictReader(fh):
        _rows.append(dict(w2=float(r["w2"]), Om=float(r["Om_star"]),
                          kappa=float(r["kappa_star"]),
                          a_Om=float(r["a_Omega"]), typ=r["type"]))
_rows = [r for r in _rows if 0.80 <= r["Om"] <= 1.62]
_rows.sort(key=lambda r: r["Om"])
_lips = [r for r in _rows if r["typ"] == "lips"]
_beaks = [r for r in _rows if r["typ"] == "beaks"]

# verified in full HB, straight from inverse_design_verify.py
VERIFIED = [dict(Om=1.20, w2=1.15958, kap=0.08683, typ="beaks"),
            dict(Om=0.95, w2=0.80132, kap=0.39204, typ="lips")]

for _sub, _col in ((_lips, st.C_ANALYTIC), (_beaks, st.C_HB)):
    if not _sub:
        continue
    _o = np.array([r["Om"] for r in _sub])
    axd.plot(_o, [r["w2"] for r in _sub], "o", ms=2.0, color=_col, mew=0,
             zorder=3)
    axd.plot(_o, [r["kappa"] for r in _sub], "^", ms=2.0, color=_col, mew=0,
             alpha=0.75, zorder=3)

# the two topologies overlap rather than leaving a gap; shade the overlap
_lo_hi = max(r["Om"] for r in _lips)
_hi_lo = min(r["Om"] for r in _beaks)
axd.axvspan(0.80, min(_lo_hi, _hi_lo), color=st.C_ANALYTIC, alpha=0.07, lw=0)
axd.axvspan(max(_lo_hi, _hi_lo), 1.62, color=st.C_HB, alpha=0.07, lw=0)
axd.axvspan(min(_lo_hi, _hi_lo), max(_lo_hi, _hi_lo), color="0.45",
            alpha=0.16, lw=0)

for _v in VERIFIED:
    _c = st.C_HB if _v["typ"] == "beaks" else st.C_ANALYTIC
    axd.annotate("", xy=(_v["Om"], _v["kap"]), xytext=(_v["Om"], _v["w2"]),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=_c, alpha=0.55))
    axd.plot([_v["Om"]], [_v["w2"]], "o", ms=6.0, mfc="none", mec=_c, mew=1.3,
             zorder=6)
    axd.plot([_v["Om"]], [_v["kap"]], "^", ms=6.0, mfc="none", mec=_c, mew=1.3,
             zorder=6)

axd.set_xlim(0.80, 1.62)
axd.set_ylim(0.0, 1.62)
axd.set_xlabel(r"target birth frequency $\Omega^\ast$")
axd.set_ylabel(r"network that delivers it")
axd.text(0.88, 1.50, "lips", color=st.C_ANALYTIC, fontsize=7.5, ha="center")
axd.text(1.45, 1.50, "beaks", color=st.C_HB, fontsize=7.5, ha="center")
axd.text(1.60, 1.06, r"$\omega_2$", fontsize=7, color="0.35", ha="right")
axd.text(1.60, 0.145, r"$\kappa^\ast$", fontsize=7, color="0.35", ha="right")

st.panel(axes[0, 0], "a")
st.panel(axes[1, 0], "b")
st.panel(axd, "c")

fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.7)
fig.subplots_adjust(left=0.115)
for ri, lab in ((0, "beaks\n(hub)"), (1, "lips\n(2-DOF)")):
    box = axes[ri, 0].get_position()
    fig.text(0.008, 0.5 * (box.y0 + box.y1), lab, rotation=90,
             va="center", ha="left", fontweight="bold", fontsize=7.5)

out = ROOT / "figures" / "fig_prl_design.pdf"
plt.savefig(out)
print(f"wrote {out}")
print(f"  design chart: {len(_rows)} births over Om* in "
      f"[{_rows[0]['Om']:.4f}, {_rows[-1]['Om']:.4f}] "
      f"({len(_lips)} lips, {len(_beaks)} beaks)")
print(f"  lips reach Om* <= {_lo_hi:.4f}; beaks reach down to {_hi_lo:.4f}; "
      f"overlap {max(0.0, _lo_hi - _hi_lo):.4f}")
print(f"  verified in full HB: "
      + ", ".join(f"Om*={v['Om']} -> (w2={v['w2']}, kap={v['kap']}) {v['typ']}"
                  for v in VERIFIED))
