"""Find a lips birth at moderate anharmonicity (epsilon <~ 0.4).

The lips case currently shown in Sec. V.C sits at omega_2 = 0.60, where the
active pole is far below the drive; the off-resonant receptance |rho| at the
born cusps is large and epsilon = (8/9)|rho|/omega_1^2 ~ 0.7.  There the
leading-order projection is out of its stated validity window, and full HB
does not reproduce the full closed-form lens.

epsilon grows with the separation of the active pole from the drive, so
moving omega_2 up towards omega_1 should bring it down while keeping
Re G_a > 0 (pole below the drive => lips).  This script scans omega_2, finds
the cusp-pair birth coupling kappa* for each, and reports epsilon at the born
cusps so a moderate-epsilon lips case can be picked for HB validation.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import brentq

W1 = 1.0
Z1, Z2 = 0.015, 0.02
BETA1 = 0.147
CVAL = 0.75 * BETA1


def modal(kappa, w2):
    """Modal (proportional-damping) data.  Retained for reference only.

    C = diag(2*Z1, 2*Z2) with Z1 != Z2 is NOT proportional damping, so the
    modal sum built from these diagonal ratios is an approximation to the
    driving-point receptance, not the receptance itself.  It shifts kappa* by
    ~0.5% at w2 = 1.25 and ~6.7% at w2 = 0.95.  inv_G below therefore uses the
    exact e_1^T Z^-1 e_1, matching make_fig_prl_design.py and the manuscript.
    """
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    ev, V = np.linalg.eigh(K)
    Cm = np.diag([2 * Z1, 2 * Z2])
    zm = [0.5 * (V[:, m] @ Cm @ V[:, m]) for m in range(2)]
    return ev, V[0, :] ** 2, zm


def inv_G(Om, kappa, w2):
    """1/G with G the exact driving-point receptance e_1^T Z^-1 e_1."""
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return 1.0 / np.linalg.inv(Z)[0, 0]


def f_of(Om, kappa, w2):
    inv = inv_G(Om, kappa, w2)
    return inv.real**2 - 3 * inv.imag**2


def cusps(kappa, w2, lo=0.3, hi=1.8, n=4000):
    oms = np.linspace(lo, hi, n)
    fv = np.array([f_of(o, kappa, w2) for o in oms])
    out = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            o = brentq(lambda x: f_of(x, kappa, w2), oms[i], oms[i + 1],
                       xtol=1e-10)
            rho = inv_G(o, kappa, w2).real
            if rho < 0:
                out.append((float(o), float(rho)))
    return sorted(out)


def eps_of(rho):
    """Anharmonicity at a cusp: eps = beta1 y_c / w1^2 = (8/9)|rho|/w1^2."""
    return (8.0 / 9.0) * abs(rho) / W1**2


def Fc_of(rho):
    return float(np.sqrt(-8.0 * rho**3 / (27.0 * CVAL)))


def find_birth(w2, kmin=0.02, kmax=1.2, nk=400):
    """Locate the coupling at which a new cusp pair appears below the drive."""
    ks = np.linspace(kmin, kmax, nk)
    prev = cusps(ks[0], w2)
    for k in ks[1:]:
        cur = cusps(k, w2)
        if len(cur) > len(prev):
            new = [c for c in cur
                   if min([abs(c[0] - q[0]) for q in prev], default=9) > 5e-3]
            if len(new) >= 2:
                return k, new, cur
        prev = cur
    return None, None, None


if __name__ == "__main__":
    w2s = [float(x) for x in sys.argv[1:]] or [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    print(f"{'w2':>6s} {'kappa*':>8s} {'Om_-':>9s} {'Om_+':>9s} "
          f"{'eps_-':>7s} {'eps_+':>7s} {'Fc_-':>8s} {'Fc_+':>8s}")
    for w2 in w2s:
        k, new, cur = find_birth(w2)
        if k is None:
            print(f"{w2:6.2f}   no clean pair birth found")
            continue
        new = sorted(new)[:2]
        (o1, r1), (o2, r2) = new[0], new[1]
        print(f"{w2:6.2f} {k:8.4f} {o1:9.5f} {o2:9.5f} "
              f"{eps_of(r1):7.3f} {eps_of(r2):7.3f} "
              f"{Fc_of(r1):8.4f} {Fc_of(r2):8.4f}")
