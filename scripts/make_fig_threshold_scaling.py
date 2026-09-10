"""Fig (threshold scaling): the isolated-pole law against Newton solutions.

Both panels use the modal two-pole receptance of Eq. (5) of the long paper,

    G(Om) = phi_a1^2 / D_a + phi_b1^2 / D_b,
    D_m   = om_m^2 - Om^2 + 2i gam_m Om,

obtained by diagonalizing K = [[w1^2+kap, -kap], [-kap, w2^2+kap]] with
proportional damping, gam_m = g1 phi_m1^2 + g2 phi_m2^2.  The birth is the
projection fold f = 0, d_Om f = 0 of f = rho^2 - 3 sigma^2, solved by Newton
(fsolve) and continued in the swept parameter.  At the baseline this returns
(Om*, kap*) = (1.3009974, 0.1183654), the value quoted in Sec. IV.

(a) kap*/sqrt(g2) over the damping sweep, against the weak-mixing prefactor
    sqrt((8-4 sqrt3) om_b (w2^2-w1^2)) = 0.8681 at om_b = w2.  Flat to about
    5% over g2 in [0.005, 0.080].
(b) kap* against sqrt(w2^2-w1^2) at g2 = 0.02, against Eq. (23),
    kap* ~ sqrt((8-4 sqrt3) om_b gam_b (w2^2-w1^2)).  The two separate as
    2 kap* << w2^2 - w1^2 is lost at small mode separation.

This script replaces an earlier ad-hoc generator that was never archived; it
reproduces the same two curves in the deposit's figure style.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import _prx_style as st

ROOT = Path(__file__).parent.parent
W1, G1 = 1.0, 0.015                     # driven mode, fixed throughout
W2_DEF, G2_DEF = 1.25, 0.02             # baseline auxiliary mode
CTHR = 8.0 - 4.0 * np.sqrt(3.0)         # (8 - 4 sqrt 3)
BASE = [1.3009974, 0.1183654]           # baseline birth, used to start each march


def modal(kappa, w2, g2):
    """(om_m, phi_m1^2, gam_m) for the two proportionally damped modes."""
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    ev, V = np.linalg.eigh(K)
    out = []
    for m in range(2):
        phi = V[:, m]
        out.append((np.sqrt(ev[m]), phi[0] ** 2, G1 * phi[0] ** 2 + g2 * phi[1] ** 2))
    return out


def f(Om, kappa, w2, g2):
    """rho^2 - 3 sigma^2 with rho + i sigma = 1/G."""
    G = sum(p / (wm**2 - Om**2 + 2j * gm * Om) for wm, p, gm in modal(kappa, w2, g2))
    inv = 1.0 / G
    return inv.real**2 - 3.0 * inv.imag**2


def birth(w2, g2, guess, h=1e-6):
    """Newton solve of f = 0, d_Om f = 0.  Returns (Om*, kappa*) or None."""
    def sys(x):
        Om, k = x
        return [f(Om, k, w2, g2),
                (f(Om + h, k, w2, g2) - f(Om - h, k, w2, g2)) / (2 * h)]
    x, _, ier, _ = fsolve(sys, guess, full_output=True)
    return (float(x[0]), float(x[1])) if ier == 1 else None


def march(values, solve):
    """Continue the birth along `values`, carrying the previous root forward."""
    got = {}
    guess = list(BASE)
    for v in values:
        r = solve(v, guess)
        if r is None:
            break
        guess = list(r)
        got[v] = r[1]
    return got


def sweep(values, solve, pivot):
    """March outward from the baseline in both directions and merge."""
    lo = [v for v in values if v <= pivot][::-1]
    hi = [v for v in values if v >= pivot]
    got = march(lo, solve)
    got.update(march(hi, solve))
    return np.array(sorted(got)), np.array([got[v] for v in sorted(got)])


# ---------------------------------------------------------------- (a) damping
g2s = np.round(np.arange(0.005, 0.08001, 0.00125), 6)
g2s, k_g = sweep(list(g2s), lambda g2, gu: birth(W2_DEF, g2, gu), G2_DEF)
pref = np.sqrt(CTHR * W2_DEF * (W2_DEF**2 - W1**2))

# --------------------------------------------------------- (b) mode separation
w2s = np.round(np.arange(1.055, 1.74501, 0.005), 6)
w2s, k_w = sweep(list(w2s), lambda w2, gu: birth(w2, G2_DEF, gu), W2_DEF)
sep = np.sqrt(w2s**2 - W1**2)
k_pred = np.sqrt(CTHR * w2s * G2_DEF * (w2s**2 - W1**2))

# ---------------------------------------------------------------------- plot
fig, (ax, bx) = plt.subplots(1, 2, figsize=st.COL_DOUBLE)

ax.plot(g2s, k_g / np.sqrt(g2s), "-", color=st.C_HB,
        label=r"Newton solution of $f=\partial_\Omega f=0$")
ax.axhline(pref, ls="--", color=st.C_ANALYTIC, label="weak-mixing prefactor")
ax.set_xlabel(r"auxiliary damping rate $\gamma_2$")
ax.set_ylabel(r"$\kappa^{*}/\sqrt{\gamma_2}$")
ax.set_xlim(g2s.min(), g2s.max())
st.legend(ax, loc="lower left")
st.panel(ax, "a")

bx.plot(sep, k_w, "-", color=st.C_HB, label="Newton solution")
bx.plot(sep, k_pred, "--", color=st.C_ANALYTIC, label="weak-mixing prediction")
bx.set_xlabel(r"$\sqrt{\omega_2^{2}-\omega_1^{2}}$")
bx.set_ylabel(r"birth coupling $\kappa^{*}$")
bx.set_xlim(sep.min(), sep.max())
st.legend(bx, loc="upper left")
st.panel(bx, "b")

st.tight(fig)
out = ROOT / "figures" / "fig_threshold_scaling.pdf"
st.save(fig, out)

csv = ROOT / "data" / "threshold_scaling.csv"
with open(csv, "w") as fh:
    fh.write("panel,x,kstar,reference\n")
    for x, k in zip(g2s, k_g):
        fh.write(f"a,{x:.6f},{k:.8f},{pref * np.sqrt(x):.8f}\n")
    for x, k, p in zip(sep, k_w, k_pred):
        fh.write(f"b,{x:.6f},{k:.8f},{p:.8f}\n")
print("wrote", csv)

r = k_g / np.sqrt(g2s) / pref
print(f"(a) kappa*/sqrt(g2) = {(k_g / np.sqrt(g2s)).min():.4f}"
      f"-{(k_g / np.sqrt(g2s)).max():.4f} vs prefactor {pref:.4f}"
      f"  -> within {100 * (1 - r.min()):.1f}%")
print(f"(b) Newton/prediction = {(k_w / k_pred).min():.3f} at "
      f"sqrt(dw2)={sep[int(np.argmin(k_w / k_pred))]:.3f} "
      f"-> {(k_w / k_pred).max():.3f} at {sep[int(np.argmax(k_w / k_pred))]:.3f}")
