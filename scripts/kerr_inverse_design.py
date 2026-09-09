"""Inverse design of a cusp-pair birth in a Josephson/Kerr device.

Everything is in ordinary frequencies (MHz); the 2 pi is already divided out.

Model (coupled-mode, one drive tone):

    [[D1 + i k1/2, -J], [-J, D2 + i k2/2]] a + (K |a1|^2 a1, 0)^T
        = (sqrt(kext) s_in, 0)^T,     Dj = fj - fd,

with K < 0 the Josephson self-Kerr.

TWO POINTS OF PHYSICS DRIVE THE DESIGN.

1. K < 0 is a SOFTENING nonlinearity.  The physical branch requires
   c*rho < 0 with c = K, hence rho > 0, and the cusp condition selects
   arg G = -/+ 30 degrees, not the -/+ 150 degrees of a hardening cubic.
   Since rho = Re(1/G) ~ (f_mode - fd), rho > 0 means the organizers live
   BELOW the modes.

2. Consequently the auxiliary mode must be placed BELOW the nonlinear mode
   for its pole to be carried into the bistable window.  An auxiliary mode
   ABOVE it sits on the branch where the fold roots have negative sum,
   y_c = -2 rho/(3K) < 0, so no physical cusp exists there at all.

Within the coupled-mode description the scalar reduction is EXACT rather than
leading order, because a single drive tone generates no other harmonics: the
second equation gives a2 = J a1/(D2 + i k2/2) with no approximation.  So there
is no multi-harmonic validation to perform here, and none is claimed.  What can
be tested is the physics beyond the Kerr truncation, and the robustness checks
below do that: the next term of the Josephson cosine, and a cross-Kerr term
representing coupler participation, which is the principal threat to the
rank-one hypothesis.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve

F1 = 0.0
K1 = 5.0            # nonlinear-mode linewidth  (MHz)
K2 = 0.50           # auxiliary-mode linewidth  (MHz)
K_KERR = -0.080     # Josephson self-Kerr       (MHz), negative


def Zmat(fd, J, df, k1=K1, k2=K2):
    """df = f2 - f1 (negative places the auxiliary mode below)."""
    return np.array([[F1 - fd + 1j * k1 / 2, -J],
                     [-J, F1 + df - fd + 1j * k2 / 2]])


def G_of(fd, J, df, **kw):
    return np.linalg.inv(Zmat(fd, J, df, **kw))[0, 0]


def rho_sigma(fd, J, df, **kw):
    inv = 1.0 / G_of(fd, J, df, **kw)
    return inv.real, inv.imag


def f_disc(fd, J, df, **kw):
    r, s = rho_sigma(fd, J, df, **kw)
    return r * r - 3.0 * s * s


def physical(fd, J, df, K=K_KERR, **kw):
    r, _ = rho_sigma(fd, J, df, **kw)
    return K * r < 0.0


def cusps(J, df, lo=None, hi=5.0, n=80001, **kw):
    lo = lo if lo is not None else (df - 60.0)
    x = np.linspace(lo, hi, n)
    v = np.array([f_disc(t, J, df, **kw) for t in x])
    out = []
    for i in np.where(np.diff(np.sign(v)))[0]:
        t = brentq(lambda u: f_disc(u, J, df, **kw), x[i], x[i + 1], xtol=1e-13)
        if physical(t, J, df, **kw):
            out.append(float(t))
    return sorted(out)


def pair_near(J, df, window=6.0, **kw):
    """The two cusps flanking the auxiliary pole, if born."""
    cs = [c for c in cusps(J, df, **kw) if abs(c - df) < window]
    return (cs[0], cs[-1]) if len(cs) >= 2 else None


def birth_J(df, J_lo=0.2, J_hi=8.0, tol=1e-7, **kw):
    """Smallest coupling at which the pair near the auxiliary pole exists."""
    if pair_near(J_hi, df, **kw) is None:
        return None
    while J_hi - J_lo > tol:
        J_mid = 0.5 * (J_lo + J_hi)
        if pair_near(J_mid, df, **kw) is None:
            J_lo = J_mid
        else:
            J_hi = J_mid
    return 0.5 * (J_lo + J_hi)


def a_omega(fd, J, df, h=1e-4, **kw):
    """Curvature of the cusp condition: sign fixes beaks vs lips."""
    return 0.5 * (f_disc(fd + h, J, df, **kw) - 2 * f_disc(fd, J, df, **kw)
                  + f_disc(fd - h, J, df, **kw)) / h**2


def occupation(fd, J, df, K=K_KERR, **kw):
    r, _ = rho_sigma(fd, J, df, **kw)
    return -2.0 * r / (3.0 * K)


def separation(J, df, **kw):
    p = pair_near(J, df, **kw)
    return float("nan") if p is None else p[1] - p[0]


def design_for_separation(target, J_op, df0=-50.0, **kw):
    """INVERSE step: choose the mode separation df giving `target` MHz of
    cusp-pair splitting at the chosen operating coupling J_op."""
    def resid(df):
        s = separation(J_op, float(df[0]), **kw)
        return [1e3 if not np.isfinite(s) else s - target]
    sol = fsolve(resid, [df0], full_output=False, xtol=1e-10)
    return float(sol[0])


if __name__ == "__main__":
    DF = -50.0
    print("=== forward: corrected device (auxiliary mode BELOW) ===")
    Js = birth_J(DF)
    p = pair_near(Js + 1e-4, DF)
    fd_s = 0.5 * (p[0] + p[1]) if p else float("nan")
    aO = a_omega(fd_s, Js + 1e-4, DF)
    print(f"  f2 - f1                  = {DF:+.1f} MHz")
    print(f"  birth coupling  J*       = {Js:.4f} MHz")
    print(f"  birth detuning  fd*-f1   = {fd_s:.4f} MHz")
    print(f"  arg G at the birth       = "
          f"{np.degrees(np.angle(G_of(fd_s, Js, DF))):+.2f} deg")
    print(f"  curvature a_Omega        = {aO:+.4g}  ->  "
          f"{'beaks' if aO > 0 else 'lips'}")
    print(f"  cusp occupation n_c      = {occupation(fd_s, Js, DF):.0f} photons")

    print("\n  pair separation above the birth:")
    for J in (2.5, 3.0, 4.0, 5.0, 6.0):
        print(f"    J = {J:.1f} MHz -> {separation(J, DF):.4f} MHz")

    print("\n=== inverse: choose f2-f1 for a target splitting ===")
    for target, J_op in ((1.0, 5.0), (1.5, 5.0), (2.0, 6.0)):
        df = design_for_separation(target, J_op)
        got = separation(J_op, df)
        print(f"  target {target:.1f} MHz at J = {J_op:.1f}: "
              f"f2 - f1 = {df:+.3f} MHz  ->  achieved {got:.4f} MHz")
