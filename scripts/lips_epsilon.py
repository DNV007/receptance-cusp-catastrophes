"""Anharmonicity at the cusps of the lips lenses (omega_2 = 0.95 and 0.60).

Both manuscripts quote epsilon for these families, at different couplings, and
the statements have to be shown to agree rather than merely asserted to. At a
cusp the fundamental fold condition pins the nonlinear shift,

    (3/4) beta1 |A1|^2 = -(2/3) rho,        so   epsilon = (8/9)|rho|/omega_1^2,

which is a function of the LINEAR receptance alone -- no beta1, no drive. So
epsilon at a cusp is computable exactly, and the comparison is unambiguous.

Result: the two cusps of a lens separate strongly in epsilon as the region
grows.  At omega_2 = 0.95 the pair is near-symmetric at the birth, epsilon =
0.35 at both cusps, and has spread to 0.29 / 0.42 by kappa = 0.303 and to
0.22 / 0.65 by kappa = 0.333.  At omega_2 = 0.60 the same construction starts
at 0.69.  That asymmetry is why the (dF)^{2/3} tip estimator fails at the UPPER
tip of the 0.95 lens while the lower one stays much nearer the leading-order
window.

CAUTION.  Quote epsilon "at the born cusps" only with the coupling attached.
The pair is symmetric at kappa* and separates immediately above it, so a single
range is meaningless without saying where it was evaluated.  Values obtained at
a coarse-grid birth locator (find_birth in lips_moderate_eps.py returns
kappa = 0.30095 against the exact 0.29976) sit neither at the birth nor at a
figure panel and should not be quoted.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

W1, Z1, Z2 = 1.0, 0.015, 0.02

# (omega_2, kappa) pairs: the birth, then the couplings the manuscripts quote.
CASES = (
    (0.95, (0.29976, 0.303, 0.333)),
    (0.60, (0.44580, 0.450, 0.495)),
)


def G(Om, kap, w2):
    K = np.array([[W1**2 + kap, -kap], [-kap, w2**2 + kap]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def f(Om, kap, w2):
    inv = 1 / G(Om, kap, w2)
    return inv.real**2 - 3 * inv.imag**2


def eps(Om, kap, w2):
    return (8 / 9) * abs((1 / G(Om, kap, w2)).real) / W1**2


def cusps(kap, w2, lo=0.60, hi=1.20, n=40000):
    o = np.linspace(lo, hi, n)
    fv = np.array([f(x, kap, w2) for x in o])
    out = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            r = brentq(f, o[i], o[i + 1], args=(kap, w2), xtol=1e-12)
            if (1 / G(r, kap, w2)).real < 0:
                out.append(r)
    return out


def main():
    print("lips lenses: epsilon = (8/9)|rho|/omega_1^2 at each cusp")
    print(f"{'omega_2':>8} {'kappa':>9} {'cusp Omega':>12} {'epsilon':>9}")
    for w2, kappas in CASES:
        for kap in kappas:
            cs = cusps(kap, w2)
            if not cs:
                print(f"{w2:8.2f} {kap:9.5f}   no physical cusp in window")
                continue
            for c in cs:
                print(f"{w2:8.2f} {kap:9.5f} {c:12.5f} {eps(c, kap, w2):9.3f}")
        print()
    print("The first kappa of each block is the birth (pair merging), where the\n"
          "two cusps carry the SAME epsilon; the others are the figure panels and\n"
          "the resolved-band comparison, where they have separated.")


if __name__ == "__main__":
    main()
