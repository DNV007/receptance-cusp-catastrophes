"""Full-HB reconnection sequence and the classifier at its weakest point.

Four panels, all from the original equations of motion at omega_2 = 1.06, where
the family carries two births of OPPOSITE predicted type with |a_Omega| = 40.0
and 18.6 against +1315 at the benchmark -- 33 and 71 times smaller. Both sit at
eps ~ 0.1 and eta_3 ~ 3e-3, so anything that fails here fails as a classifier,
not as a first-harmonic reduction. That separation is the point of the figure.

  (a) the whole reconnection sequence in one parameter sweep:
      connected band -> gap opens -> isolated lens -> nothing;
  (b) where the lens actually closes, against where the receptance says it does;
  (c) isolation on a window wide enough to contain the third cusp;
  (d) the sqrt(mu) separation law and its coefficient at |a_Omega| = 18.6.

Inputs (all archived):
  data/typeboundary_hb_verify.csv       (a)
  data/typeboundary_lips_threshold.csv  (b)
  data/typeboundary_lips_scaling.csv    (c), (d)

Run:  .venv/bin/python scripts/make_fig_typeboundary.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
W2, GAM = 1.06, np.array([0.015, 0.02])
KPRED, C_FIT, DELTA = 0.0592217, 0.1381, 4.3375e-4


def G(Om, kap):
    Z = np.array([[1.0 - Om**2 + 2j*GAM[0]*Om + kap, -kap],
                  [-kap, W2**2 - Om**2 + 2j*GAM[1]*Om + kap]], complex)
    return np.linalg.inv(Z)[0, 0]


def f(Om, kap):
    inv = 1.0/G(Om, kap)
    return inv.real**2 - 3*inv.imag**2


def cusps(kap, lo=1.00, hi=1.13, n=6000):
    oms = np.linspace(lo, hi, n)
    fv = np.array([f(o, kap) for o in oms])
    r = [brentq(lambda x: f(x, kap), oms[i], oms[i+1], xtol=1e-13)
         for i in range(n-1) if fv[i]*fv[i+1] < 0]
    return [x for x in r if (1.0/G(x, kap)).real < 0]


def load(fn, keycol=0):
    rows = list(csv.DictReader(open(os.path.join(DATA, fn))))
    return rows, list(csv.reader(open(os.path.join(DATA, fn))))[0]


def series(rows, key, keyfield, om="Omega", w="tongue_width"):
    o = [float(r[om]) for r in rows if r[keyfield] == key]
    t = [float(r[w]) for r in rows if r[keyfield] == key]
    return np.array(o), np.array(t)


fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
(ax, bx), (cx, dx) = axes

# ---- (a) the reconnection sequence -----------------------------------------
rows, _ = load("typeboundary_hb_verify.csv")
seq = [("0.036001", "beaks", r"$\kappa=0.0360$: connected", st.C_BEFORE),
       ("0.046001", "beaks", r"$\kappa=0.0460$: gap", st.C_AT),
       ("0.054222", "lips",  r"$\kappa=0.0542$: lens", st.C_AFTER),
       ("0.064222", "lips",  r"$\kappa=0.0642$: none", st.GREY)]
srows, _ = load("typeboundary_lips_scaling.csv")
for i, (kap, case, lab, col) in enumerate(seq):
    sub = [r for r in rows if r["case"] == case and r["kappa"] == kap]
    o = np.array([float(r["Omega"]) for r in sub])
    t = np.array([float(r["tongue_width"]) for r in sub])
    live = np.isfinite(t) & (t > 0)
    ax.plot(o[live], np.full(live.sum(), 3-i), "s", ms=2.6, color=col,
            mec="white", mew=0.4, label=lab)
    for c in cusps(float(kap)):
        if o.min()-2e-3 <= c <= o.max()+2e-3:
            ax.plot([c], [3-i], "|", ms=9, color="k", mew=1.0, zorder=5)
ax.annotate("no bistability in window", xy=(1.072, 0.0), fontsize=6.5,
            color=st.GREY, ha="center", va="center")
ax.set_yticks([3, 2, 1, 0])
ax.set_yticklabels([s[2].split(":")[0] for s in seq], fontsize=6.5)
ax.set_xlabel(r"$\Omega$"); ax.set_ylim(-0.6, 3.6)
ax.set_title("bistable frequencies (full HB)", fontsize=7.5, pad=3)
st.panel(ax, "a")

# ---- (b) where the lens closes ----------------------------------------------
trow, _ = load("typeboundary_lips_threshold.csv")
ks = sorted({r["kappa"] for r in trow}, key=float)
kk, ww = [], []
for k in ks:
    o, t = series(trow, k, "kappa")
    live = np.isfinite(t) & (t > 0)
    kk.append(float(k))
    ww.append(o[live].max() - o[live].min() if live.any() else 0.0)
kk, ww = np.array(kk), np.array(ww)
bx.plot(kk, ww, "o", ms=4, color=st.C_HB, mec="white", mew=0.6, label="full HB")
mu = np.linspace(0, KPRED + DELTA - kk.min(), 200)
bx.plot(KPRED + DELTA - mu, C_FIT*np.sqrt(mu), "-", lw=1.2, color=st.C_ANALYTIC,
        label=r"$c\sqrt{\mu+\delta}$")
bx.axvline(KPRED, ls="--", lw=1.0, color=st.GREY)
bx.annotate(r"receptance $\kappa^\ast$", xy=(KPRED + 3e-5, 0.0006),
            fontsize=6.5, color=st.GREY, ha="left", rotation=90, va="bottom")
bx.axvspan(0.0595, 0.0597, color=st.C_HB, alpha=0.15, lw=0)
bx.annotate("HB closes here", xy=(0.0596, 0.0016), fontsize=6.5,
            color=st.C_HB, ha="center")
bx.set_xlabel(r"$\kappa$"); bx.set_ylabel(r"lens width $\Delta\Omega$")
st.legend(bx, loc="upper right"); st.panel(bx, "b")

# ---- (c) isolation on a wide window -----------------------------------------
o, t = series(srows, "isolation", "case")
live = np.isfinite(t) & (t > 0)
cx.semilogy(o[live], t[live], "s", ms=3.0, color=st.C_HB, mec="white", mew=0.4)
for c in cusps(0.054222):
    cx.axvline(c, ls=":", lw=0.9, color=st.C_ANALYTIC)
cx.annotate("lens", xy=(1.0607, 4e-4), fontsize=6.5, color=st.C_HB, ha="center")
cx.annotate("main band", xy=(1.118, 4e-4), fontsize=6.5, color=st.C_HB,
            ha="center")
cx.set_xlabel(r"$\Omega$"); cx.set_ylabel(r"tongue width $\Delta F$")
cx.set_title(r"$\kappa=0.0542$, window to $1.130$", fontsize=7.5, pad=3)
st.panel(cx, "c")

# ---- (d) the sqrt(mu) law ----------------------------------------------------
mus, wid = [], []
for key in ("scaling_5e-03", "scaling_2e-03", "scaling_1e-03", "scaling_5e-04"):
    o, t = series(srows, key, "case")
    live = np.isfinite(t) & (t > 0)
    mus.append(float(key.split("_")[1]))
    wid.append(o[live].max() - o[live].min())
mus, wid = np.array(mus), np.array(wid)
mfine = np.linspace(3e-4, 6e-3, 200)
dx.plot(np.sqrt(mfine), C_FIT*np.sqrt(mfine + DELTA), "-", lw=1.2,
        color=st.C_ANALYTIC, label=r"$c\sqrt{\mu+\delta}$")
dx.plot(np.sqrt(mfine), 0.1408*np.sqrt(mfine), "--", lw=1.0, color=st.GREY,
        label=r"$c_\ast\sqrt{\mu}$, receptance")
dx.plot(np.sqrt(mus), wid, "o", ms=4.5, color=st.C_HB, mec="white", mew=0.6,
        label="full HB")
dx.set_xlabel(r"$\sqrt{\mu}$"); dx.set_ylabel(r"$\Delta\Omega$")
st.legend(dx, loc="upper left"); st.panel(dx, "d")

st.tight(fig)
st.save(fig, os.path.join(ROOT, "figures", "fig_typeboundary.pdf"))
print("wrote figures/fig_typeboundary.pdf")
