"""Hub topology: one driven+cubic nonlinear resonator (mass 1) coupled directly
to (N-1) linear satellite resonators, no satellite-satellite coupling. Mass 1
then has direct receptance to every mode, so each satellite pole should spawn a
cusp-pair birth at a COMPARABLE, ACCESSIBLE drive (unlike the chain, whose high
rungs sit at F_c~2-6). Closed-form check: births + their cusp drive F_c."""
import numpy as np

BETA1 = 0.147
C3 = 0.75 * BETA1


def K_hub(omegas, kappa):
    N = len(omegas)
    K = np.diag(np.asarray(omegas, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += kappa
        K[j, j] += kappa
        K[0, j] -= kappa
        K[j, 0] -= kappa
    return K


def rho_sigma(Om, omegas, zetas, kappa):
    N = len(omegas)
    Zm = K_hub(omegas, kappa) - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in zetas])
    G = np.linalg.solve(Zm, np.eye(N)[:, 0])[0]
    inv = 1.0 / G
    return inv.real, inv.imag


def f(Om, omegas, zetas, kappa):
    r, s = rho_sigma(Om, omegas, zetas, kappa)
    return r ** 2 - 3 * s ** 2


def Fc(Om, omegas, zetas, kappa):
    r, s = rho_sigma(Om, omegas, zetas, kappa)
    if r >= 0:
        return np.nan
    yc = -2 * r / (3 * C3)
    F2 = yc * ((r + C3 * yc) ** 2 + s ** 2)
    return np.sqrt(F2) if F2 > 0 else np.nan


def cusp_count(omegas, zetas, kappa, om_lo, om_hi, n=6000):
    oms = np.linspace(om_lo, om_hi, n)
    fv = np.array([f(o, omegas, zetas, kappa) for o in oms])
    pts = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            o = oms[i] - fv[i] * (oms[i + 1] - oms[i]) / (fv[i + 1] - fv[i])
            r, _ = rho_sigma(o, omegas, zetas, kappa)
            if r < 0:
                pts.append((o, Fc(o, omegas, zetas, kappa)))
    return pts


def scan(name, omegas, zetas, k_lo, k_hi, nk, om_lo, om_hi):
    print("=" * 70)
    print(f"{name}: omega={omegas}")
    ks = np.linspace(k_lo, k_hi, nk)
    counts = [len(cusp_count(omegas, zetas, k, om_lo, om_hi)) for k in ks]
    print(f"  cusp count over kappa[{k_lo},{k_hi}]: {min(counts)}..{max(counts)}")
    for j in range(len(ks) - 1):
        if counts[j + 1] > counts[j]:
            lo, hi, c0 = ks[j], ks[j + 1], counts[j]
            for _ in range(40):
                m = 0.5 * (lo + hi)
                if len(cusp_count(omegas, zetas, m, om_lo, om_hi)) > c0:
                    hi = m
                else:
                    lo = m
            ks_ = 0.5 * (lo + hi)
            pts = cusp_count(omegas, zetas, ks_ + 5e-4, om_lo, om_hi)
            a = 0.5 * (f(np.mean([p[0] for p in pts[-2:]]) + 1e-4, omegas, zetas, ks_)
                       - 2 * f(np.mean([p[0] for p in pts[-2:]]), omegas, zetas, ks_)
                       + f(np.mean([p[0] for p in pts[-2:]]) - 1e-4, omegas, zetas, ks_))
            newF = [p[1] for p in pts[-2:]]
            print(f"  birth kappa*={ks_:.4f}  count {c0}->{counts[j+1]}  "
                  f"new cusp F_c ~ {np.round(newF,3)}")


scan("HUB N=3 (satellites 1.25,1.5)", [1.0, 1.25, 1.5], [0.015, 0.02, 0.02],
     0.02, 0.6, 150, 0.9, 2.2)
scan("HUB N=4 (satellites 1.25,1.5,1.75)", [1.0, 1.25, 1.5, 1.75],
     [0.015, 0.02, 0.02, 0.02], 0.02, 0.7, 150, 0.9, 2.6)
