"""PRL figure: the phase-contour law tested against HB-located cusps.

This is the panel the Letter was missing. Every other figure draws the
contour from the receptance and then compares *derived* quantities against
harmonic balance. None evaluates arg G at a cusp located independently of the
phase condition, so the central equation is never tested directly.

Cusp locations here come from full multi-harmonic balance via the
(Delta F)^{2/3} tongue-width intercept (cusp_tip_intercept.py), which never
solves at the tip and never references f = rho^2 - 3 sigma^2. arg G is then
evaluated from the LINEAR receptance alone.

BOTH BRANCHES OF EQ. (5) APPEAR HERE. The hardening runs (beta1 = +0.147)
test the -150 deg level; the softening runs (beta1 = -0.147,
softening_contour_validation.py) test the -30 deg level. Plotting them on one
axis is the point: the sign of beta1 selects between two widely separated
invariant levels and nothing lands in between.

  (a) arg G(Omega) across the coupling families, with the measured cusps.
  (b) Residual from the relevant level against kappa. For the hardening
      family it is one-signed and monotone -- a systematic, not scatter --
      but in equivalent-Omega terms comparable to the intercept estimator's
      own fit-window bias. It is an order of magnitude too large, and
      wrong-signed, to be the third-harmonic correction of Eq. (9).

Run with the project venv:  ../.venv/bin/python make_fig_prl_contour.py
Output: figures/fig_prl_contour.pdf
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st
from phase_contour_validation import collect, arg_G, dargG_dOm, SIG_OM, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "fig_prl_contour.pdf")
SOFT_CSV = os.path.join(ROOT, "data", "softening_contour_validation.csv")
SOFT_TARGET = -30.0


def load_softening():
    if not os.path.exists(SOFT_CSV):
        return []
    raw = np.genfromtxt(SOFT_CSV, delimiter=",", names=True)
    out = []
    for r in np.atleast_1d(raw):
        out.append(dict(kappa=float(r["kappa"]), Om=float(r["Om_hb"]),
                        argG=float(r["argG"]), resid=float(r["resid_deg"])))
    return out


def main():
    rows = collect()
    for d in rows:
        d["argG"] = arg_G(d["Om"], d["kappa"], d["w2"])
        d["resid"] = d["argG"] - TARGET
        d["sig"] = (abs(dargG_dOm(d["Om"], d["kappa"], d["w2"]))
                    * d.get("sig_Om", SIG_OM))

    beaks = [d for d in rows if d["family"] == "beaks"]
    lips = [d for d in rows if d["family"] == "lips"]
    # Band-edge localization of the lips family: shown in (a) as a
    # placement test, kept out of (b) whose y-scale is set by the much
    # finer tip estimator.
    lips_band = [d for d in rows if d["family"] == "lips_band"]
    soft = load_softening()

    # Wide layout: the two frequency regions carry no data between them, so
    # they get their own panels sharing one phase axis. This removes the dead
    # band and, by making the figure wide rather than tall, roughly halves its
    # APS word-equivalent cost (150/aspect + 20 for a single-column figure).
    # Wide rather than tall: same two panels, laid side by side. A
    # single-column figure costs 150/aspect + 20 word-equivalents under the
    # APS rule, so the horizontal layout is roughly half the price of the
    # stacked one for the same content.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.4, 1.95),
        gridspec_kw=dict(width_ratios=[1.30, 1.0], wspace=0.42))

    kaps = sorted({d["kappa"] for d in beaks})
    cmap = plt.cm.Blues(np.linspace(0.30, 0.92, len(kaps)))
    oms = np.linspace(1.28, 1.45, 900)
    for c, kap in zip(cmap, kaps):
        ax1.plot(oms, [arg_G(o, kap, 1.25) for o in oms],
                 color=c, lw=0.55, zorder=1)

    kaps_l = sorted({d["kappa"] for d in lips + lips_band})
    cmap_l = plt.cm.Greens(np.linspace(0.45, 0.85, len(kaps_l)))
    oms_l = np.linspace(1.00, 1.10, 700)
    for c, kap in zip(cmap_l, kaps_l):
        ax1.plot(oms_l, [arg_G(o, kap, 0.95) for o in oms_l],
                 color=c, lw=0.55, zorder=1)

    if soft:
        kaps_s = sorted({d["kappa"] for d in soft})
        cmap_s = plt.cm.Purples(np.linspace(0.40, 0.88, len(kaps_s)))
        oms_s = np.linspace(0.94, 1.075, 800)
        for c, kap in zip(cmap_s, kaps_s):
            ax1.plot(oms_s, [arg_G(o, kap, 1.25) for o in oms_s],
                     color=c, lw=0.55, zorder=1)

    for lvl in (TARGET, SOFT_TARGET):
        ax1.axhline(lvl, color=st.GREY, lw=0.9, ls="--", zorder=2)

    ax1.plot([d["Om"] for d in beaks], [d["argG"] for d in beaks],
             "o", color=st.C_HB, ms=2.7, mew=0.35, mec="white", zorder=4)
    ax1.plot([d["Om"] for d in lips], [d["argG"] for d in lips],
             "^", color=st.GREEN, ms=3.4, mew=0.35, mec="white", zorder=4)
    if lips_band:
        ax1.errorbar([d["Om"] for d in lips_band],
                     [d["argG"] for d in lips_band],
                     yerr=[d["sig"] for d in lips_band],
                     fmt="^", color=st.GREEN, ms=3.4, mew=0.35, mec="white",
                     ecolor=st.GREEN, elinewidth=0.7, capsize=1.4, zorder=4)
    if soft:
        ax1.plot([d["Om"] for d in soft], [d["argG"] for d in soft],
                 "s", color=st.PURPLE, ms=2.9, mew=0.35, mec="white", zorder=4)

    ax1.set_xlim(0.93, 1.46)
    ax1.set_ylim(-176, -6)
    ax1.set_yticks([-150, -120, -90, -60, -30])
    ax1.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4])
    ax1.set_xlabel(r"cusp frequency $\Omega$")
    ax1.set_ylabel(r"$\arg G$ (deg)")
    ax1.text(1.452, SOFT_TARGET, r"$-30^\circ$", ha="right", va="bottom",
             fontsize=6.2, color=st.GREY)
    ax1.text(1.452, TARGET, r"$-150^\circ$", ha="right", va="bottom",
             fontsize=6.2, color=st.GREY)
    # both annotations live in the empty band between the two families
    ax1.text(1.115, -48, r"$\beta_1<0$", fontsize=6.2, color=st.PURPLE,
             ha="left", va="top")
    ax1.text(1.150, -138, r"$\beta_1>0$", fontsize=6.2, color=st.C_HB,
             ha="left", va="bottom")
    st.panel(ax1, "a")

    # ---------------- (b) residual --------------------------------------
    kb = [d["kappa"] for d in beaks]
    rb = [d["resid"] for d in beaks]
    order = np.argsort(kb)
    kb_s = np.array(kb)[order]
    sb_s = np.array([d["sig"] for d in beaks])[order]

    ax2.fill_between(kb_s, -sb_s, sb_s, color=st.LIGHTGREY, alpha=0.55,
                     lw=0, zorder=1, label=r"estimator bias")
    ax2.axhline(0.0, color=st.GREY, lw=0.9, ls="--", zorder=2)
    ax2.plot(kb, rb, "o", color=st.C_HB, ms=2.9, mew=0.4, mec="white",
             zorder=4)
    ax2.plot([d["kappa"] for d in lips], [d["resid"] for d in lips],
             "^", color=st.GREEN, ms=3.6, mew=0.4, mec="white", zorder=4)
    if soft:
        ax2.plot([d["kappa"] for d in soft], [d["resid"] for d in soft],
                 "s", color=st.PURPLE, ms=3.0, mew=0.4, mec="white", zorder=4)

    allr = rb + [d["resid"] for d in lips] + [d["resid"] for d in soft]
    lo, hi = min(allr), max(allr)
    pad = 0.20 * (hi - lo) if hi > lo else 0.5
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_xlabel(r"coupling $\kappa$")
    ax2.set_ylabel(r"residual (deg)")
    ax2.set_xlim(0.02, 0.36)
    ax2.set_xticks([0.1, 0.2, 0.3])
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    st.legend(ax2, loc="upper left", over_data=True)
    st.panel(ax2, "b", dx=0.02)

    st.tight(fig)
    st.save(fig, OUT)

    print(f"  hardening tips: n = {len(beaks) + len(lips)}, "
          f"max|resid| {max(abs(r) for r in rb + [d['resid'] for d in lips]):.3f} deg")
    if lips_band:
        print(f"  lips band edges: n = {len(lips_band)}, "
              f"max|resid| {max(abs(d['resid']) for d in lips_band):.3f} deg "
              f"(sigma {min(d['sig'] for d in lips_band):.2f}-"
              f"{max(d['sig'] for d in lips_band):.2f} deg)")
    if soft:
        sr = [d["resid"] for d in soft]
        print(f"  softening: n = {len(soft)}, "
              f"mean {np.mean(sr):+.3f}, max|resid| {max(abs(r) for r in sr):.3f} deg")


if __name__ == "__main__":
    main()
