"""Departures from rank-one geometry in the Josephson device.

CORRECTION.  An earlier version of this analysis kept the cross-Kerr term
chi|a2|^2 a1 in the nonlinear-mode equation while treating a2 = J a1/D2 as
exact.  That is inconsistent: a conservative cross-Kerr is reciprocal, so it
also appears in the auxiliary equation as chi|a1|^2 a2, and the resulting
back-action on a2 contributes at the same order.  The full quartic system is

    D1 a1 - J a2 + K1|a1|^2 a1 + chi|a2|^2 a1 = F,
    D2 a2 - J a1 + K2|a2|^2 a2 + chi|a1|^2 a2 = 0.

Eliminating a2 to first order in (chi, K2) gives

    K_eff = K1 + chi J^2 (1/|D2|^2 + 1/D2^2) + K2 J^4/(|D2|^2 D2^2),

where the second term of the chi bracket is what the earlier treatment
omitted.  Two consequences follow, and both matter:

  * 1/D2^2 largely cancels 1/|D2|^2 in the real part.  The real
    renormalisation is enhanced by about 12, not by the 92 previously
    claimed, so the earlier statement that a 5% cross-Kerr reverses the sign
    of the effective Kerr is wrong.  The real part turns over near
    chi/|K| = 0.083.

  * 1/D2^2 is complex, so the cross-Kerr is predominantly DISSIPATIVE, not
    reactive: |Im/Re| of the correction is about 3.8 at the operating point.
    An amplitude-dependent loss is not described by the real-coefficient fold
    polynomial at all, and it is this, rather than any sign reversal, that
    sets the tolerance.

The earlier analysis therefore arrived at roughly the right tolerance by the
wrong mechanism.  The corrected tolerance is quoted below.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import kerr_inverse_design as kid

DF = -50.0
K1 = kid.K_KERR


def D2_of(fd, df=DF, k2=kid.K2):
    return kid.F1 + df - fd + 1j * k2 / 2


def K_eff(fd, J, chi=0.0, K2=0.0, df=DF, k2=kid.K2):
    """Effective (generally complex) quartic coefficient after eliminating a2."""
    d = D2_of(fd, df, k2)
    m = abs(d) ** 2
    return (K1
            + chi * J**2 * (1.0 / m + 1.0 / d**2)
            + K2 * J**4 / (m * d**2))


def cusp_residual(fd, J, **kw):
    """Zero where the two folds merge, for complex quartic coefficient."""
    rho, sig = kid.rho_sigma(fd, J, kw.get("df", DF))
    c = K_eff(fd, J, **kw)
    a = 3.0 * (c.real**2 + c.imag**2)
    b = 4.0 * (c.real * rho + c.imag * sig)
    return b * b - 4.0 * a * (rho * rho + sig * sig)


def cusps(J, lo=None, hi=5.0, n=40001, **kw):
    df = kw.get("df", DF)
    lo = lo if lo is not None else df - 20.0
    x = np.linspace(lo, hi, n)
    v = np.array([cusp_residual(t, J, **kw) for t in x])
    out = []
    for i in np.where(np.diff(np.sign(v)))[0]:
        t = brentq(lambda u: cusp_residual(u, J, **kw), x[i], x[i + 1],
                   xtol=1e-13)
        rho, sig = kid.rho_sigma(t, J, df)
        c = K_eff(t, J, **kw)
        a = 3.0 * (c.real**2 + c.imag**2)
        b = 4.0 * (c.real * rho + c.imag * sig)
        if -b / (2 * a) > 0:                     # merged root must be positive
            out.append(float(t))
    return sorted(out)


def birth(J_lo=0.2, J_hi=9.0, tol=1e-6, window=8.0, **kw):
    df = kw.get("df", DF)
    def has_pair(J):
        return len([c for c in cusps(J, **kw) if abs(c - df) < window]) >= 2
    if not has_pair(J_hi):
        return None
    while J_hi - J_lo > tol:
        m = 0.5 * (J_lo + J_hi)
        if has_pair(m):
            J_hi = m
        else:
            J_lo = m
    return 0.5 * (J_lo + J_hi)


if __name__ == "__main__":
    J0 = kid.birth_J(DF)
    p0 = kid.pair_near(J0 + 1e-4, DF)
    fd0 = 0.5 * (p0[0] + p0[1])
    d0 = D2_of(fd0)
    coef = J0**2 * (1 / abs(d0)**2 + 1 / d0**2)
    print(f"unperturbed birth: J* = {J0:.4f} MHz, fd*-f1 = {fd0:.4f} MHz")
    print(f"cross-Kerr coefficient J^2(|D2|^-2 + D2^-2) = "
          f"{coef.real:+.2f}{coef.imag:+.2f}i")
    print(f"  (one-sided form used earlier: {J0**2/abs(d0)**2:+.2f}, real)\n")

    print("effect of a reciprocal cross-Kerr on the birth:")
    for r in (1e-3, 5e-3, 1e-2, 5e-2):
        chi = r * abs(K1)
        ke = K_eff(fd0, J0, chi=chi)
        Jb = birth(chi=chi)
        shift = "n/a" if Jb is None else f"{1e3*(Jb-J0):+8.2f} kHz"
        print(f"  chi/|K| = {r:6.3f}:  K_eff = {ke.real:+.5f}{ke.imag:+.5f}i, "
              f"|Im/Re| = {abs(ke.imag/ke.real):6.2f},  J* shift {shift}")

    print("\ntolerance from the dissipative part, |Im K_eff| < 0.05 |K|:")
    tol_chi = 0.05 * abs(K1) / abs(coef.imag)
    print(f"  chi/|K| < {tol_chi/abs(K1):.2e}")
    print("  (the earlier 5e-4 was numerically close but followed from a "
          "sign-reversal\n   argument that does not survive the reciprocal "
          "treatment)")

    print("\nauxiliary-mode population at the birth:")
    n1 = kid.occupation(fd0, J0, DF)
    ratio = J0**2 / abs(d0)**2
    print(f"  n1 = {n1:.0f},  n2/n1 = J^2/|D2|^2 = {ratio:.0f},  "
          f"n2 = {ratio*n1:.3g} photons")
