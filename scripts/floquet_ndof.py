"""Floquet stability of the born bistable orbits (the referee's first question).
Reconstruct a periodic orbit from its harmonic-balance coefficients, integrate the
variational equation over one drive period to get the monodromy, and report its
spectral radius. Outer branches of a genuine bistable window must have spectral
radius < 1 (and no secondary Neimark-Sacker / period-doubling escape).

System: M xddot + C xdot + K x + beta1 x1^3 e1 = F cos(Om t) e1, M=I, C=diag(2 zeta).
Jacobian of the vector field u=(x,v): Df = [[0, I], [-(K + 3 beta1 x1(t)^2 P), -C]],
P = e1 e1^T.
"""
import numpy as np
from ndof_hb import build_K, make_funcs


def K_hub(om, k):
    N = len(om); K = np.diag(np.asarray(om, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += k; K[j, j] += k; K[0, j] -= k; K[j, 0] -= k
    return K


def closed_form_y(omegas, zetas, K, Om, F, beta1):
    """Real positive |X1|^2 roots at (Om,F): low, (mid), high branches."""
    N = len(omegas)
    Z = K - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in zetas])
    G = np.linalg.solve(Z, np.eye(N)[:, 0])[0]
    inv = 1.0 / G
    rho, sig = inv.real, inv.imag
    c = 0.75 * beta1
    roots = np.roots([c ** 2, 2 * c * rho, rho ** 2 + sig ** 2, -F ** 2])
    return sorted(r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0)


def orbit_x1_of_t(coeffs, N, nh, Om, ts):
    c = coeffs.reshape(N, nh, 2)
    x1 = np.zeros_like(ts)
    for m in range(1, nh + 1):
        x1 += c[0, m - 1, 0] * np.cos(m * Om * ts) + c[0, m - 1, 1] * np.sin(m * Om * ts)
    return x1


def spectral_radius(coeffs, omegas, zetas, K, beta1, Om, nh, nsteps=4000):
    N = len(omegas)
    C = np.diag([2 * z for z in zetas])
    P = np.zeros((N, N)); P[0, 0] = 1.0
    T = 2 * np.pi / Om
    dt = T / nsteps
    I = np.eye(N)

    def Df(t):
        x1 = orbit_x1_of_t(coeffs, N, nh, Om, np.array([t]))[0]
        Kd = K + 3 * beta1 * x1 ** 2 * P
        top = np.hstack([np.zeros((N, N)), I])
        bot = np.hstack([-Kd, -C])
        return np.vstack([top, bot])

    Phi = np.eye(2 * N)
    t = 0.0
    for _ in range(nsteps):
        k1 = Df(t) @ Phi
        k2 = Df(t + dt / 2) @ (Phi + dt / 2 * k1)
        k3 = Df(t + dt / 2) @ (Phi + dt / 2 * k2)
        k4 = Df(t + dt) @ (Phi + dt * k3)
        Phi = Phi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    mult = np.linalg.eigvals(Phi)
    return np.max(np.abs(mult)), mult


def check(label, omegas, zetas, Kf, kappa, Om, F, beta1=0.147, nh=5):
    K = Kf(omegas, kappa)
    N = len(omegas)
    fns = make_funcs(omegas, zetas, beta1, nh=nh, nt=512)
    ys = closed_form_y(omegas, zetas, K, Om, F, beta1)
    print(f"\n{label}: kappa={kappa} Om={Om} F={F}")
    print(f"  closed-form |X1| branches: {[round(np.sqrt(y),4) for y in ys]}")
    if len(ys) < 3:
        print("  (not in a 3-solution bistable window at this (Om,F))")
    tags = ["low", "mid", "high"] if len(ys) == 3 else [f"b{i}" for i in range(len(ys))]
    for y, tag in zip(ys, tags):
        seed = np.zeros(N * nh * 2); seed[0] = np.sqrt(y)
        coeffs, rn = fns["solve"](K, F, Om, guess=seed)
        A1 = float(np.hypot(coeffs.reshape(N, nh, 2)[0, 0, 0],
                            coeffs.reshape(N, nh, 2)[0, 0, 1]))
        sr, _ = spectral_radius(coeffs, omegas, zetas, K, beta1, Om, nh)
        verdict = "STABLE" if sr < 1.0 + 1e-3 else "UNSTABLE"
        print(f"  {tag:4s}: HB A1={A1:.4f} (res {rn:.1e})  spectral radius={sr:.4f}  {verdict}")


if __name__ == "__main__":
    # 2-DOF baseline window (paper reports Floquet ~0.916 here) -- sanity
    check("2-DOF baseline window (sanity)", [1.0, 1.25], [0.015, 0.02],
          build_K, 0.10, 1.10, 0.050)
    # hub beaks born orbit: upper sub-wedge above kappa*
    check("hub beaks, upper sub-wedge", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02],
          K_hub, 0.14, 1.27, 0.19)
    check("hub beaks, lower sub-wedge", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02],
          K_hub, 0.14, 1.15, 0.12)
    # lips born orbit
    check("lips born orbit", [1.0, 0.60], [0.015, 0.02],
          build_K, 0.65, 0.85, 0.7, nh=7)
