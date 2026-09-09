"""
Comb-corrected cusp condition.

Exact closure:   X1^(n) = G(n Om) [ (F/2)(d_{n,1}+d_{n,-1}) - beta1 (x1^3)^(n) ]

Truncating at n=3 (complete to O(beta1^2)):

    F = A [ 1/G1 + (3/4) b |A|^2 - (3/16) b^2 G3 |A|^4 ]

With u = c y, c = (3/4) b, y = |A|^2, and (3/16)b^2 / c^2 = 1/3 exactly:

    F^2 = (u/c) * Phi,   Phi = (rho + u - g_r u^2/3)^2 + (sigma - g_i u^2/3)^2

Cusp:  W'(u) = W''(u) = 0  with W = u * Phi   (c drops out of the Om-condition).
"""
import numpy as np
from scipy.optimize import brentq, fsolve

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
BETA = 0.147


def Zmat(Om, kap):
    K = np.array([[W1**2 + kap, -kap], [-kap, W2**2 + kap]])
    return K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])


def G(Om, kap):
    return np.linalg.inv(Zmat(Om, kap))[0, 0]


def Wfun(u, Om, kap, comb=True):
    g1 = G(Om, kap)
    rho, sig = (1 / g1).real, (1 / g1).imag
    if comb:
        g3 = G(3 * Om, kap)
        gr, gi = g3.real, g3.imag
    else:
        gr = gi = 0.0
    P = rho + u - gr * u**2 / 3.0
    Q = sig - gi * u**2 / 3.0
    return u * (P**2 + Q**2)


def dW(u, Om, kap, comb=True, h=1e-6):
    return (Wfun(u + h, Om, kap, comb) - Wfun(u - h, Om, kap, comb)) / (2 * h)


def d2W(u, Om, kap, comb=True, h=1e-5):
    return (Wfun(u + h, Om, kap, comb) - 2 * Wfun(u, Om, kap, comb)
            + Wfun(u - h, Om, kap, comb)) / h**2


def cusp(kap, Om0, comb=True):
    """Solve W'=W''=0 for (u_c, Om_c)."""
    g1 = G(Om0, kap)
    u0 = -2 * (1 / g1).real / 3.0

    def sys(v):
        u, Om = v
        return [dW(u, Om, kap, comb), d2W(u, Om, kap, comb)]

    sol, info, ier, msg = fsolve(sys, [u0, Om0], full_output=True, xtol=1e-13)
    u_c, Om_c = sol
    c = 0.75 * BETA
    Fc = np.sqrt(Wfun(u_c, Om_c, kap, comb) / c)
    return Om_c, Fc, u_c, ier


def cusp_hb1_phase(kap, br):
    """HB1 cusp: root of rho^2 - 3 sigma^2 with c*rho<0."""
    def f(Om):
        g = 1 / G(Om, kap)
        return g.real**2 - 3 * g.imag**2
    return brentq(f, *br, xtol=1e-14)


print("=" * 74)
print("BASELINE  kappa=0.10  (lower cusp; paper: HB1 Om=1.06724 F=0.02292,")
print("                       full HB Om~1.066, F~0.022)")
print("=" * 74)

kap = 0.10
Om_hb1 = cusp_hb1_phase(kap, (1.04, 1.10))
Om0, F0, u0, _ = cusp(kap, Om_hb1, comb=False)
Om1, F1, u1, ier = cusp(kap, Om_hb1, comb=True)

print(f"  phase-contour root (rho^2=3sig^2) : Om = {Om_hb1:.9f}")
print(f"  W'=W''=0, no comb                 : Om = {Om0:.9f}  F = {F0:.7f}")
print(f"  W'=W''=0, with G(3Om)             : Om = {Om1:.9f}  F = {F1:.7f}  (ier={ier})")
print(f"  comb shift in Om                  : {Om1 - Om0:+.3e}  ({(Om1-Om0)/Om0*100:+.4f} %)")
print(f"  comb shift in F                   : {F1 - F0:+.3e}  ({(F1-F0)/F0*100:+.4f} %)")

g1 = G(Om1, kap)
g3 = G(3 * Om1, kap)
print(f"\n  |G(Om)|  = {abs(g1):.5f}   arg = {np.degrees(np.angle(g1)):+.4f} deg")
print(f"  |G(3Om)| = {abs(g3):.5f}   arg = {np.degrees(np.angle(g3)):+.4f} deg")
print(f"  comb ratio |G3/G1|          = {abs(g3/g1):.3e}")
print(f"  predicted small param 0.128*|G3/G1| = {0.128*abs(g3/g1):.3e}")

print("\n" + "=" * 74)
print("beta1-INDEPENDENCE OF THE CUSP FREQUENCY (with comb correction on)")
print("=" * 74)
print(f"{'beta1':>8} {'Om_c (comb)':>16} {'F_c':>12} {'F_c*sqrt(b)':>14}")
for b in [0.05, 0.1, 0.147, 0.3, 0.6, 1.2]:
    BETA = b
    Om_b, F_b, _, _ = cusp(kap, Om_hb1, comb=True)
    print(f"{b:8.3f} {Om_b:16.12f} {F_b:12.6f} {F_b*np.sqrt(b):14.6f}")
BETA = 0.147

print("\n  -> Om_c identical across beta1 => cusp frequency is a property of")
print("     the LINEAR network alone, exact through O(beta1^2).")
print("     F_c * sqrt(beta1) constant => F_c ~ beta1^{-1/2} preserved.")

print("\n" + "=" * 74)
print("COUPLING SWEEP: comb shift vs HB1, both cusps")
print("=" * 74)
print(f"{'kappa':>7} {'Om_HB1':>14} {'Om_comb':>14} {'dOm':>12} {'dF/F %':>9}")
for kk in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    try:
        Ohb1 = cusp_hb1_phase(kk, (1.02, 1.12))
    except ValueError:
        continue
    Oa, Fa, _, _ = cusp(kk, Ohb1, comb=False)
    Ob, Fb, _, _ = cusp(kk, Ohb1, comb=True)
    print(f"{kk:7.3f} {Oa:14.9f} {Ob:14.9f} {Ob-Oa:+12.3e} {(Fb-Fa)/Fa*100:+9.4f}")
