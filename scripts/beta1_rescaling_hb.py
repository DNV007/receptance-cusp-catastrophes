"""Full harmonic-balance test of the homogeneous-cubic rescaling.

The Letter claims that the independence of the cusp frequency from |beta_1| is
exact for a pure homogeneous cubic, not an artifact of retaining one harmonic:
under

    x~ = sqrt(|b|) x,      F~ = sqrt(|b|) F,

the FULL equations of motion depend on beta_1 only through sgn(beta_1).  Nothing
in that argument is perturbative, so it must hold in the full harmonic-balance
solution with every harmonic retained -- not merely in the fundamental
reduction, and not merely in the comb-corrected reduction of
`comb_cusp_shift.py`, which is what the beta1-independence table there tests.

This script closes that gap.  It solves the full NH-harmonic balance system for
both coordinates at a fixed drive frequency, for a baseline (b0, F0) and then
for a range of beta_1 with the drive rescaled as F(b) = F0 sqrt(b0/b).  The
prediction is A_n(b) = A_n(b0) sqrt(b0/b) for EVERY harmonic n, exactly.

Result (Om = 1.0673142, kappa = 0.10, NH = 9): the rescaled |A_1| agrees with
the baseline to all 12 printed decimals, and the largest relative deviation
over all nine harmonics is <= 1.4e-16 -- machine precision, across a factor of
24 in beta_1.  No fitting, no tolerance tuning.
"""
from __future__ import annotations

import numpy as np

W = np.array([1.0, 1.25])
ZET = np.array([0.015, 0.02])
KAP = 0.10
N, NH, NT = 2, 9, 1024

# Baseline: the comb-corrected cusp of the kappa = 0.10 benchmark, so the test
# runs at the operating point the Letter actually quotes.
OM = 1.0673142
B0, F0 = 0.147, 0.0229505
BETAS = (0.05, 0.10, 0.30, 0.60, 1.20)


def build_K(kap):
    return np.array([[W[0] ** 2 + kap, -kap], [-kap, W[1] ** 2 + kap]])


def residual(z, K, force, Om, beta):
    c = z.reshape(N, NH, 2)[:, :, 0]
    s = z.reshape(N, NH, 2)[:, :, 1]
    period = 2 * np.pi / Om
    t = np.arange(NT) * (period / NT)
    ms = np.arange(1, NH + 1)
    w = ms * Om
    wt = np.outer(ms, Om * t)
    cos, sin = np.cos(wt), np.sin(wt)
    x = c @ cos + s @ sin
    v = (-(c * w)) @ sin + (s * w) @ cos
    a = (-(c * w ** 2)) @ cos + (-(s * w ** 2)) @ sin
    cub = np.zeros_like(x)
    cub[0] = beta * x[0] ** 3
    drv = np.zeros_like(x)
    drv[0] = force * np.cos(Om * t)
    res = a + 2 * ZET[:, None] * v + K @ x + cub - drv
    dt = period / NT
    return np.stack([res @ cos.T * dt, res @ sin.T * dt], axis=-1).reshape(-1)


def solve(K, force, Om, beta, guess=None):
    """Damped Newton on the square HB system (finite-difference Jacobian)."""
    n = N * NH * 2
    z = np.zeros(n) if guess is None else guess.copy()
    if guess is None:
        z[0] = force / max(1e-6, abs(W[0] ** 2 - Om ** 2))
    for _ in range(300):
        r = residual(z, K, force, Om, beta)
        nr = np.linalg.norm(r)
        if nr < 1e-15:
            break
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(z[j]))
            zp = z.copy()
            zp[j] += h
            J[:, j] = (residual(zp, K, force, Om, beta) - r) / h
        dz = np.linalg.solve(J, -r)
        lam = 1.0
        for _ in range(50):
            if np.linalg.norm(residual(z + lam * dz, K, force, Om, beta)) < nr:
                break
            lam *= 0.5
        z = z + lam * dz
    return z, float(np.linalg.norm(residual(z, K, force, Om, beta)))


def harmonics(z):
    """Complex A_n on the driven coordinate, x1 = sum_n Re(A_n e^{i n Om t})."""
    c = z.reshape(N, NH, 2)[0, :, 0]
    s = z.reshape(N, NH, 2)[0, :, 1]
    return c - 1j * s


def main():
    K = build_K(KAP)
    z0, r0 = solve(K, F0, OM, B0)
    A0 = harmonics(z0)
    print(f"FULL HB rescaling test: NH={NH}, Om={OM}, kappa={KAP}")
    print(f"baseline beta1={B0}, F={F0}, HB residual={r0:.2e}\n")
    print(f"{'beta1':>8} {'F(beta1)':>12} {'|A1| raw':>18} "
          f"{'|A1|*sqrt(b/b0)':>20} {'max rel dev (all n)':>22} {'HB res':>10}")
    print(f"{B0:8.3f} {F0:12.6f} {abs(A0[0]):18.12f} {abs(A0[0]):20.12f} "
          f"{0.0:22.3e} {r0:10.1e}")
    worst = 0.0
    for b in BETAS:
        F = F0 * np.sqrt(B0 / b)
        z, r = solve(K, F, OM, b, guess=z0 * np.sqrt(B0 / b))
        Ab = harmonics(z)
        scaled = Ab * np.sqrt(b / B0)
        dev = np.max(np.abs(scaled - A0)) / np.max(np.abs(A0))
        worst = max(worst, dev)
        print(f"{b:8.3f} {F:12.6f} {abs(Ab[0]):18.12f} {abs(scaled[0]):20.12f} "
              f"{dev:22.3e} {r:10.1e}")
    print(f"\n-> worst relative deviation over all {NH} harmonics and a factor "
          f"{max(BETAS)/min(BETAS):.0f} in beta1: {worst:.1e}")
    print("   The rescaling is exact in the FULL harmonic-balance solution, "
          "not just in the reduction.")


if __name__ == "__main__":
    main()
