"""PRL Fig. 3 -- zero-parameter validation of the comb closure.

(a) The slaved third harmonic A3 = -(1/4) beta1 G(3Om) A1^3 against full
    multi-harmonic balance. Nothing is fitted: A1 is taken from the HB
    solution and A3 is predicted from the LINEAR receptance at 3 Omega.

(b) Relative residual of the reduced equation against epsilon, for the
    cubic (first-harmonic) and quintic (comb-corrected) forms, with
    slope-2 and slope-3 guides.

SCOPE NOTE. This figure deliberately does NOT show an F_c error budget.
The comb bounds the harmonic-truncation contribution to F_c at 0.14% and
the modal-vs-exact receptance choice at 0.29%, against a reported 3-8%
discrepancy; the remainder is not yet understood, so putting it in a figure
would assert more than is known. The Omega argument, by contrast, closes:
the physical shift is 2.4e-5, far below the resolution of fold-tracking
cusp estimators, and that is stated in the caption as a number rather than
drawn as a panel.

Single column (3.4 in) on purpose: at 3750 APS words the binding constraint
for this Letter is caption plus figure area, not prose.

Environment: needs jax + matplotlib (default interpreter). Do NOT run under
PYTHONNOUSERSITE=1 -- that removes jax. No scipy is used here.

Output: figures/fig_prl_validation.pdf
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prx_style as st

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = [1.0, 1.25]
ZET = [0.015, 0.02]
BETA = 0.147
KAP = 0.10
NH = 9
NT = 1024
N = 2
C3 = 0.75 * BETA


def Kmat(kap):
    return np.array([[W[0] ** 2 + kap, -kap], [-kap, W[1] ** 2 + kap]])


def G(Om, kap):
    Z = Kmat(kap) - Om ** 2 * np.eye(2) + 1j * Om * np.diag(
        [2 * ZET[0], 2 * ZET[1]])
    return np.linalg.inv(Z)[0, 0]


def _residual(coeffs, K, force, Om):
    coeffs = coeffs.reshape(N, NH, 2)
    c, s = coeffs[:, :, 0], coeffs[:, :, 1]
    period = 2.0 * jnp.pi / Om
    t = jnp.arange(NT) * (period / NT)
    ms = jnp.arange(1, NH + 1)
    w = ms * Om
    wt = jnp.outer(ms, Om * t)
    cos, sin = jnp.cos(wt), jnp.sin(wt)
    x = c @ cos + s @ sin
    v = (-(c * w)) @ sin + (s * w) @ cos
    a = (-(c * w ** 2)) @ cos + (-(s * w ** 2)) @ sin
    zc = jnp.asarray(ZET)[:, None]
    res = (a + 2.0 * zc * v + K @ x
           + jnp.zeros_like(x).at[0].set(BETA * x[0] ** 3)
           - jnp.zeros_like(x).at[0].set(force * jnp.cos(Om * t)))
    dt = period / NT
    return jnp.stack([res @ cos.T * dt, res @ sin.T * dt], axis=-1).reshape(-1)


_res = jax.jit(_residual)
_jac = jax.jit(jax.jacfwd(_residual))


def solve_hb(K, force, Om, guess=None):
    z = np.zeros(N * NH * 2) if guess is None else np.asarray(guess).copy()
    if guess is None:
        z[0] = force / max(1e-6, abs(W[0] ** 2 - Om ** 2))
    Kj = jnp.asarray(K)
    for _ in range(300):
        r = np.asarray(_res(jnp.asarray(z), Kj, force, Om))
        nr = np.linalg.norm(r)
        if nr < 1e-14:
            break
        J = np.asarray(_jac(jnp.asarray(z), Kj, force, Om))
        try:
            dz = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam, ok = 1.0, False
        for _ in range(60):
            if np.linalg.norm(np.asarray(
                    _res(jnp.asarray(z + lam * dz), Kj, force, Om))) < nr:
                ok = True
                break
            lam *= 0.5
        if not ok:
            break
        z = z + lam * dz
    return z, float(np.linalg.norm(np.asarray(_res(jnp.asarray(z), Kj,
                                                   force, Om))))


def A_of(coeffs):
    a = coeffs.reshape(N, NH, 2)[0]
    return a[:, 0] - 1j * a[:, 1]


# --------------------------------------------------------------- gather data
K = Kmat(KAP)
OMS = [1.05, 1.0673142, 1.10, 1.20, 1.30]

# ADAPTIVE DRIVE RANGE. A fixed list of F samples wildly different eps at
# different Omega: |A1| ~ |G(Om)| F, and |G| varies by a factor ~10 across
# this set, so eps varies by ~100. With a fixed list the far-off-resonance
# points (Om = 1.30) sat at eps ~ 1e-5 where the quintic residual is at the
# double-precision floor, and the fitted slope came out 1.25 instead of 3.
# Targeting a common eps window instead, via F ~ sqrt(eps/beta1)/|G(Om)|,
# puts every Omega on the same footing and keeps all of them off the floor.
EPS_TARGETS = np.geomspace(4e-3, 6e-2, 8)
rec = []
dropped = 0
for Om in OMS:
    g1, g3 = G(Om, KAP), G(3 * Om, KAP)
    FS = [float(np.sqrt(t / BETA) / abs(g1)) for t in EPS_TARGETS]
    for F in FS:
        z, rn = solve_hb(K, F, Om, guess=None)
        if rn > 1e-11:                     # convergence guard: folds are stiff
            dropped += 1
            continue
        A = A_of(z)
        A1, A3 = A[0], A[2]
        eps = BETA * abs(A1) ** 2 / W[0] ** 2
        if eps > 0.25:                     # beyond the leading-order window
            dropped += 1
            continue
        A3p = -0.25 * BETA * g3 * A1 ** 3
        r_cub = abs(A1 * (1 / g1 + C3 * abs(A1) ** 2) - F) / F
        r_qui = abs(A1 * (1 / g1 + C3 * abs(A1) ** 2
                          - (3 / 16) * BETA ** 2 * g3 * abs(A1) ** 4) - F) / F
        rec.append((Om, F, eps, abs(A3), abs(A3p), r_cub, r_qui))

rec_all = np.array(rec)

# TWO FILTERS, both necessary, both found the hard way.
#
# (1) DOUBLE-PRECISION FLOOR. At Om = 1.30 the quintic residual reaches
#     2.5e-16 -- machine epsilon on a relative residual. Those points are
#     roundoff, not signal, and fitting through them returned a quintic
#     slope of 1.999 instead of 3. Keeping only residuals > 1e-13 leaves
#     three decades of headroom above the floor and recovers slope 3.
#
# (2) ASYMPTOTIC WINDOW. At Om = 1.05 the sweep reaches eps = 0.14, where
#     the cubic slope softens to 1.78: the eps^2 / eps^3 laws are asymptotic
#     as eps -> 0, and higher-order terms partially cancel at large drive.
#     This is a real limitation, not an artifact, so it is reported rather
#     than hidden -- but the scaling claim is made only inside the window.
FLOOR, EPS_MAX = 1e-12, 0.10
keep = (rec_all[:, 6] > FLOOR) & (rec_all[:, 2] < EPS_MAX)
rec = rec_all[keep]
print(f"  {len(rec_all)} converged points, {dropped} dropped at solve time")
print(f"  {keep.sum()} retained for the scaling fit "
      f"({(~keep).sum()} at the numerical floor or beyond eps={EPS_MAX})")

# ------------------------------------------------------------------- figure
fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 1.85))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(min(OMS), max(OMS))

# (a) A3 predicted vs measured
lim = [0.4 * rec_all[:, 4].min(), 2.5 * rec_all[:, 3].max()]
ax.plot(lim, lim, "-", color="0.6", lw=0.7, zorder=1)
for Om in OMS:
    m = rec_all[:, 0] == Om
    ax.loglog(rec_all[m, 4], rec_all[m, 3], "o", ms=2.8, mfc=cmap(norm(Om)),
              mec="white", mew=0.35, ls="", zorder=3)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel(r"$|A_3|$ from $G(3\Omega)$")
ax.set_ylabel(r"$|A_3|$ from full HB")
ax.set_aspect("equal", adjustable="box")
st.panel(ax, "a")

# (b) residual scaling
e = rec[:, 2]
bx.loglog(e, rec[:, 5], "o", ms=2.6, mfc=st.C_HB, mec="white", mew=0.35,
          ls="", label="cubic", zorder=3)
bx.loglog(e, rec[:, 6], "s", ms=2.6, mfc=st.C_ANALYTIC, mec="white", mew=0.35,
          ls="", label="quintic", zorder=3)
eg = np.array([e.min() * 0.8, e.max() * 1.25])
# Anchor each guide through the MEDIAN of its band, not the maximum: the
# vertical spread within a band is the Omega-dependent prefactor, and a
# guide pinned to the top of the scatter reads as a line that misses.
for p, col, col_i in [(2, st.C_HB, 5), (3, st.C_ANALYTIC, 6)]:
    off = np.median(rec[:, col_i] / rec[:, 2] ** p)
    bx.loglog(eg, off * eg ** p, "--", color=col, lw=0.7, zorder=2)
bx.text(0.06, 0.90, r"$\varepsilon^{2}$", transform=bx.transAxes,
        fontsize=7, color=st.C_HB)
bx.text(0.40, 0.09, r"$\varepsilon^{3}$", transform=bx.transAxes,
        fontsize=7, color=st.C_ANALYTIC)
bx.set_xlabel(r"$\varepsilon=|\beta_1||A_1|^2/\omega_1^2$")
bx.set_ylabel("relative residual")
st.legend(bx, loc="lower right", fontsize=6, handlelength=0.9,
          borderpad=0.2, labelspacing=0.15)
st.panel(bx, "b")

st.tight(fig)
out = os.path.join(ROOT, "figures", "fig_prl_validation.pdf")
plt.savefig(out)
print("wrote", out)
print("  per-Omega log-log slopes inside the window "
      "(pooling across Omega mixes the Omega-dependent prefactor "
      "and is NOT a valid fit):")
scs, sqs = [], []
for Om in OMS:
    k = rec[:, 0] == Om
    if k.sum() < 4:
        continue
    a = np.polyfit(np.log(rec[k, 2]), np.log(rec[k, 5]), 1)[0]
    b = np.polyfit(np.log(rec[k, 2]), np.log(rec[k, 6]), 1)[0]
    scs.append(a); sqs.append(b)
    print(f"    Om={Om:.4f}  n={k.sum():d}  cubic {a:.3f}  quintic {b:.3f}")
print(f"  cubic   {min(scs):.2f}-{max(scs):.2f} (expect 2)")
print(f"  quintic {min(sqs):.2f}-{max(sqs):.2f} (expect 3)")
print(f"  median |A3| relative error (all points): "
      f"{np.median(abs(rec_all[:,3]-rec_all[:,4])/rec_all[:,3]):.3%}")
g=rec[:,5]/rec[:,6]
print(f"  residual gain in window: median {np.median(g):.0f}x, "
      f"range {g.min():.0f}-{g.max():.0f}x")
