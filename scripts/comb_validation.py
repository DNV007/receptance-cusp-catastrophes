"""
DIRECT test of the comb closure against full multi-harmonic balance.

Self-contained (jax + numpy only; no scipy) to avoid the broken user-site
scipy in this environment. HB residual matches scripts/ndof_hb.py exactly:

    xddot + 2 zeta xdot + K x + beta1 x1^3 e1 = F cos(Om t) e1

Two ZERO-PARAMETER predictions of the comb closure
    X1^(n) = G(n Om) [ (F/2)(d_{n,1}+d_{n,-1}) - beta1 (x1^3)^(n) ]
tested on converged HB solutions:

  (A)  A3  = -(1/4) beta1 G(3Om) A1^3
  (B)  F   = A1 [ 1/G1 + (3/4)b|A1|^2 - (3/16) b^2 G3 |A1|^4 ]     (quintic)
       vs   A1 [ 1/G1 + (3/4)b|A1|^2 ]                             (HB1 cubic)

Both use the HB-measured A1. Nothing is fitted.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

W = [1.0, 1.25]
ZET = [0.015, 0.02]
BETA = 0.147
KAP = 0.10
NH = 9
NT = 1024
N = 2


def build_K(kap):
    return np.array([[W[0] ** 2 + kap, -kap], [-kap, W[1] ** 2 + kap]])


def Zmat(Om, kap):
    return build_K(kap) - Om**2 * np.eye(2) + 1j * Om * np.diag(
        [2 * ZET[0], 2 * ZET[1]])


def G(Om, kap):
    return np.linalg.inv(Zmat(Om, kap))[0, 0]


def _residual(coeffs, K, force, Om):
    coeffs = coeffs.reshape(N, NH, 2)
    c, s = coeffs[:, :, 0], coeffs[:, :, 1]
    period = 2.0 * jnp.pi / Om
    t = jnp.arange(NT) * (period / NT)
    ms = jnp.arange(1, NH + 1)
    w = ms * Om
    wt = jnp.outer(ms, Om * t)
    cos, sin = jnp.cos(wt), jnp.sin(wt)
    x = c @ cos + s @ sin
    v = (-(c * w)) @ sin + (s * w) @ cos
    a = (-(c * w**2)) @ cos + (-(s * w**2)) @ sin
    zc = jnp.asarray(ZET)[:, None]
    lin = K @ x
    cubic = jnp.zeros_like(x).at[0].set(BETA * x[0] ** 3)
    drive = jnp.zeros_like(x).at[0].set(force * jnp.cos(Om * t))
    res = a + 2.0 * zc * v + lin + cubic - drive
    dt = period / NT
    return jnp.stack([res @ cos.T * dt, res @ sin.T * dt], axis=-1).reshape(-1)


_res = jax.jit(_residual)
_jac = jax.jit(jax.jacfwd(_residual))


def solve_hb(K, force, Om, guess=None):
    """Plain Newton on the square HB system."""
    n = N * NH * 2
    z = np.zeros(n) if guess is None else guess.copy()
    if guess is None:
        z[0] = force / max(1e-6, abs(W[0] ** 2 - Om**2))
    Kj = jnp.asarray(K)
    for _ in range(200):
        r = np.asarray(_res(jnp.asarray(z), Kj, force, Om))
        nr = np.linalg.norm(r)
        if nr < 1e-14:
            break
        J = np.asarray(_jac(jnp.asarray(z), Kj, force, Om))
        try:
            dz = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam = 1.0
        for _ in range(40):  # damped step
            zt = z + lam * dz
            if np.linalg.norm(np.asarray(_res(jnp.asarray(zt), Kj, force, Om))) < nr:
                break
            lam *= 0.5
        z = z + lam * dz
    return z, float(np.linalg.norm(np.asarray(_res(jnp.asarray(z), Kj, force, Om))))


def harmonics(coeffs):
    """Complex A_m on coordinate 1, x1 = sum_m Re(A_m e^{i m Om t})."""
    a = coeffs.reshape(N, NH, 2)[0]
    return a[:, 0] - 1j * a[:, 1]


K = build_K(KAP)
c = 0.75 * BETA

print("=" * 84)
print(f"COMB CLOSURE vs FULL HB   (kappa={KAP}, beta1={BETA}, N_HARM={NH})")
print("=" * 84)

for Om in [1.0673, 1.05, 1.10, 1.20]:
    g1, g3 = G(Om, KAP), G(3 * Om, KAP)
    print(f"\n--- Om={Om}  |G1|={abs(g1):.4f}  |G3|={abs(g3):.5f}  "
          f"|G3/G1|={abs(g3/g1):.3e} ---")
    print(f"{'F':>8} {'|A1|':>9} {'|A3| HB':>11} {'|A3| pred':>11} {'A3 err':>8}"
          f" {'cubic res':>11} {'quintic res':>12} {'gain':>8}")
    guess = None
    for F in [0.008, 0.015, 0.022, 0.035, 0.05, 0.08]:
        x, rn = solve_hb(K, F, Om, guess=guess)
        if rn > 1e-10:
            x, rn = solve_hb(K, F, Om, guess=None)
        guess = x
        A = harmonics(x)
        A1, A3 = A[0], A[2]
        A3p = -0.25 * BETA * g3 * A1**3
        e3 = abs(A3 - A3p) / max(abs(A3), 1e-30)
        r_cub = abs(A1 * (1 / g1 + c * abs(A1) ** 2) - F) / F
        r_qui = abs(A1 * (1 / g1 + c * abs(A1) ** 2
                          - (3 / 16) * BETA**2 * g3 * abs(A1) ** 4) - F) / F
        print(f"{F:8.4f} {abs(A1):9.5f} {abs(A3):11.3e} {abs(A3p):11.3e}"
              f" {e3:8.2%} {r_cub:11.3e} {r_qui:12.3e} {r_cub/r_qui:7.1f}x")

print("\n" + "=" * 84)
print("RESIDUAL SCALING WITH THE COMB RATIO |G3/G1|   (F = 0.03 fixed)")
print("=" * 84)
print(f"{'Om':>7} {'|G3/G1|':>11} {'|A1|':>9} {'cubic res':>11}"
      f" {'quintic res':>12} {'gain':>8}")
for Om in [0.95, 1.00, 1.0673, 1.15, 1.30, 1.45]:
    g1, g3 = G(Om, KAP), G(3 * Om, KAP)
    F = 0.03
    x, rn = solve_hb(K, F, Om, guess=None)
    A1 = harmonics(x)[0]
    r_cub = abs(A1 * (1 / g1 + c * abs(A1) ** 2) - F) / F
    r_qui = abs(A1 * (1 / g1 + c * abs(A1) ** 2
                      - (3 / 16) * BETA**2 * g3 * abs(A1) ** 4) - F) / F
    print(f"{Om:7.4f} {abs(g3/g1):11.3e} {abs(A1):9.5f} {r_cub:11.3e}"
          f" {r_qui:12.3e} {r_cub/r_qui:7.1f}x")
