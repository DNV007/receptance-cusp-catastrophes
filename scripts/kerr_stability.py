"""Dynamical stability of the branches in the Josephson device.

The mechanical examples carry Floquet checks; the device example so far shows
only that solution branches exist.  Existence is not accessibility: a driven
Kerr dimer can lose its steady states to a Hopf bifurcation and settle into a
limit cycle while the static branch structure is untouched.  If the born
branches are Hopf-unstable the cusp topology is not experimentally usable,
whatever the receptance predicts.

In the rotating frame the coupled-mode equations are autonomous, so this is a
4x4 eigenvalue problem rather than a monodromy integration:

    da/dt = i[ Z a + K|a1|^2 a1 e1 - F e1 ],

whose steady states are exactly the solutions of the reduction used
throughout, and whose linearisation about them gives the stability directly.
The Kerr term is not holomorphic, so the Jacobian is taken on the real
4-vector (Re a1, Im a1, Re a2, Im a2).

A real eigenvalue crossing zero is a fold, expected at the wedge boundaries.
A complex pair crossing is a Hopf bifurcation, which is the outcome that
would matter.
"""
from __future__ import annotations

import numpy as np

import kerr_inverse_design as kid

DF = -50.0
K = kid.K_KERR


def Zmat(fd, J, df=DF):
    return kid.Zmat(fd, J, df) if hasattr(kid, "Zmat") else np.array(
        [[kid.F1 - fd + 1j * kid.K1 / 2, -J],
         [-J, kid.F1 + df - fd + 1j * kid.K2 / 2]])


def rhs(v, fd, J, F, df=DF):
    """Real 4-vector field."""
    a = np.array([v[0] + 1j * v[1], v[2] + 1j * v[3]])
    Z = Zmat(fd, J, df)
    d = 1j * (Z @ a + np.array([K * abs(a[0])**2 * a[0], 0.0])
              - np.array([F, 0.0]))
    return np.array([d[0].real, d[0].imag, d[1].real, d[1].imag])


def jacobian(v, fd, J, F, df=DF, h=1e-7):
    n = len(v)
    Jm = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = h
        Jm[:, k] = (rhs(v + e, fd, J, F, df) - rhs(v - e, fd, J, F, df)) / (2 * h)
    return Jm


def branches(fd, J, F, df=DF):
    """All steady states, from the scalar reduction."""
    rho, sig = kid.rho_sigma(fd, J, df)
    inv = rho + 1j * sig
    # |F|^2 = y[(rho + K y)^2 + sigma^2]
    co = [K * K, 2 * K * rho, rho * rho + sig * sig, -F * F]
    out = []
    for y in np.roots(co):
        if abs(y.imag) > 1e-9 or y.real <= 0:
            continue
        yr = y.real
        a1 = F / (inv + K * yr)
        Z = Zmat(fd, J, df)
        a2 = -Z[1, 0] * a1 / Z[1, 1]
        out.append((yr, np.array([a1.real, a1.imag, a2.real, a2.imag])))
    return sorted(out, key=lambda t: t[0])


def classify(fd, J, F, df=DF):
    res = []
    for y, v in branches(fd, J, F, df):
        r = np.linalg.norm(rhs(v, fd, J, F, df))
        ev = np.linalg.eigvals(jacobian(v, fd, J, F, df))
        mx = ev.real.max()
        # is the least-stable mode oscillatory?
        k = int(np.argmax(ev.real))
        osc = abs(ev[k].imag) > 1e-6
        res.append(dict(y=y, n1=y, resid=r, max_re=mx, osc=osc, ev=ev))
    return res


if __name__ == "__main__":
    from cusp_tip_intercept import F_folds  # noqa: F401  (not used; parity)

    J = 5.0
    pair = kid.pair_near(J, DF)
    print(f"J = {J} MHz, born cusp pair at "
          f"{pair[0]:.4f} / {pair[1]:.4f} MHz\n")

    # sample inside each sub-wedge, at a drive between the two folds
    for tag, fd in (("below the pair", pair[0] - 0.30),
                    ("inside the gap", 0.5 * (pair[0] + pair[1])),
                    ("above the pair", pair[1] + 0.30)):
        rho, sig = kid.rho_sigma(fd, J, DF)
        disc = rho * rho - 3 * sig * sig
        if disc <= 0 or K * rho >= 0:
            print(f"{tag:16s} fd={fd:+.4f}: not bistable "
                  f"(rho^2-3sig^2={disc:+.3f})")
            continue
        sd = np.sqrt(disc)
        ys = sorted([(-2 * rho + sd) / (3 * K), (-2 * rho - sd) / (3 * K)])
        Fs = [np.sqrt(y * ((rho + K * y)**2 + sig**2)) for y in ys]
        Fmid = 0.5 * (Fs[0] + Fs[1])
        print(f"{tag:16s} fd={fd:+.4f}: folds at F = "
              f"{min(Fs):.4f}, {max(Fs):.4f}; probing F = {Fmid:.4f}")
        for b in classify(fd, J, Fmid):
            verdict = "STABLE" if b["max_re"] < 0 else (
                "UNSTABLE (Hopf)" if b["osc"] else "UNSTABLE (real)")
            print(f"    n1 = {b['n1']:9.2f}  max Re lambda = "
                  f"{b['max_re']:+.5f}  {verdict}")
        print()
