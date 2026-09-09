"""Scan kappa* dependence on (omega2, zeta1, zeta2) to extract a scaling law.

Cusp-of-cusps existence depends on the linear modal admittance only;
beta_1 enters only through F_c at the tip. We scan the four linear
parameters and report kappa*(.) and Omega*(.) along each axis.
"""
import numpy as np
from scipy.optimize import brentq

W1_DEF, W2_DEF = 1.0, 1.25
Z1_DEF, Z2_DEF = 0.015, 0.02


def transfer(Om, kappa, w1, w2, z1, z2):
    K = np.array([[w1**2 + kappa, -kappa], [-kappa, w2**2 + kappa]])
    evals, evecs = np.linalg.eigh(K)
    wa2, wb2 = evals
    pa1, pb1 = evecs[0, 0], evecs[0, 1]
    pa, pb = evecs[:, 0], evecs[:, 1]
    Cmat = np.diag([2 * z1, 2 * z2])
    za = 0.5 * (pa @ Cmat @ pa)
    zb = 0.5 * (pb @ Cmat @ pb)
    Da = wa2 - Om**2 + 2j * za * Om
    Db = wb2 - Om**2 + 2j * zb * Om
    G = pa1**2 / Da + pb1**2 / Db
    inv = 1.0 / G
    return inv.real, inv.imag


def n_upper_cusps(kappa, params):
    w1, w2, z1, z2 = params
    om_lo, om_hi = 0.5 * (w1 + w2), 1.5 * w2
    oms = np.linspace(om_lo, om_hi, 6001)
    a = np.empty(len(oms)); b = np.empty(len(oms))
    for i, o in enumerate(oms):
        a[i], b[i] = transfer(o, kappa, w1, w2, z1, z2)
    f = a * a - 3 * b * b
    sc = np.where(np.diff(np.sign(f)) != 0)[0]
    n = 0
    for i in sc:
        try:
            Om = brentq(lambda o: transfer(o, kappa, w1, w2, z1, z2)[0]**2
                                  - 3 * transfer(o, kappa, w1, w2, z1, z2)[1]**2,
                        oms[i], oms[i + 1])
            a_c, _ = transfer(Om, kappa, w1, w2, z1, z2)
            if a_c < 0:
                n += 1
        except ValueError:
            continue
    return n


def find_kstar(params, klo=0.001, khi=0.5):
    for _ in range(50):
        mid = 0.5 * (klo + khi)
        n = n_upper_cusps(mid, params)
        if n >= 2:
            khi = mid
        else:
            klo = mid
        if khi - klo < 5e-7:
            break
    return 0.5 * (klo + khi)


def find_Omstar(kstar, params):
    w1, w2, z1, z2 = params
    om_lo, om_hi = 0.5 * (w1 + w2), 1.5 * w2
    oms = np.linspace(om_lo, om_hi, 80001)
    f = np.empty(len(oms))
    for i, o in enumerate(oms):
        a, b = transfer(o, kstar, w1, w2, z1, z2)
        f[i] = a*a - 3*b*b
    # Touch-zero from below (or local min of |f| where a<0)
    mask = np.array([transfer(o, kstar, params[0], params[1], params[2], params[3])[0] < 0
                     for o in oms])
    if not mask.any():
        return np.nan
    i = np.argmin(np.abs(f[mask]))
    return oms[mask][i]


print("Baseline: w1=1.0, w2=1.25, z1=0.015, z2=0.02 -> expect k* ~ 0.1184")
base = (W1_DEF, W2_DEF, Z1_DEF, Z2_DEF)
k0 = find_kstar(base)
Om0 = find_Omstar(k0, base)
print(f"  k* = {k0:.6f},  Om* = {Om0:.5f}")

print("\n--- vary w2 (modal-frequency separation), others fixed ---")
print(f"  {'w2':>6} {'(w2^2-w1^2)':>12} {'k*':>10} {'k*/dw2^2':>12} {'Om*':>10}")
for w2 in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00]:
    p = (W1_DEF, w2, Z1_DEF, Z2_DEF)
    try:
        k = find_kstar(p, klo=1e-4, khi=2.0)
        dw2 = w2**2 - W1_DEF**2
        Om = find_Omstar(k, p)
        print(f"  {w2:6.3f} {dw2:12.4f} {k:10.5f} {k/dw2:12.5f} {Om:10.4f}")
    except Exception as e:
        print(f"  {w2:6.3f}  ERROR: {e}")

print("\n--- vary z2 (driven-mode damping affects upper-cusp visibility) ---")
print(f"  {'z2':>8} {'k*':>10} {'k*/z2':>10} {'Om*':>10}")
for z2 in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080]:
    p = (W1_DEF, W2_DEF, Z1_DEF, z2)
    try:
        k = find_kstar(p)
        Om = find_Omstar(k, p)
        print(f"  {z2:8.4f} {k:10.5f} {k/z2:10.4f} {Om:10.4f}")
    except Exception as e:
        print(f"  {z2:8.4f}  ERROR")

print("\n--- vary z1 (other-mode damping) ---")
print(f"  {'z1':>8} {'k*':>10} {'k*/z1':>10} {'Om*':>10}")
for z1 in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040]:
    p = (W1_DEF, W2_DEF, z1, Z2_DEF)
    try:
        k = find_kstar(p)
        Om = find_Omstar(k, p)
        print(f"  {z1:8.4f} {k:10.5f} {k/z1:10.4f} {Om:10.4f}")
    except Exception as e:
        print(f"  {z1:8.4f}  ERROR")

print("\n--- joint zeta scaling at z1=z2=z (proportional damping) ---")
print(f"  {'z':>8} {'k*':>10} {'k*/z':>10} {'k*/z^2':>10} {'Om*':>10}")
for z in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080]:
    p = (W1_DEF, W2_DEF, z, z)
    try:
        k = find_kstar(p)
        Om = find_Omstar(k, p)
        print(f"  {z:8.4f} {k:10.5f} {k/z:10.4f} {k/(z*z):10.2f} {Om:10.4f}")
    except Exception as e:
        print(f"  {z:8.4f}  ERROR")
