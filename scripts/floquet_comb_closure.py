"""
Does the harmonic-comb closure extend to the FLOQUET problem?

CLAIM.  Linearizing about a periodic orbit x*(t),

    M d2(dx) + C d(dx) + K dx + p(t) dx_1 e_1 = 0,   p = 3 beta1 x1*^2

is rank one along e_1, so with dx = e^{lam t} sum_n u^(n) e^{i n Om t},

    u_1^(n) + Ghat(lam + i n Om) * sum_k p^(k) u_1^(n-k) = 0,
    Ghat(s) = e_1^T (M s^2 + C s + K)^{-1} e_1.

=> every true Floquet exponent must make  [1 + Gcal(lam) Pcal]  singular.

TEST.  Ground truth = monodromy by direct RK4 integration of the variational
equation (no Hill truncation anywhere).  Then evaluate sigma_min of the
scalar comb matrix at those exponents, against a control set of nearby
non-exponents.  A decisive pass needs sigma_min to collapse at the true
exponents and NOT at the controls.

Also checks the lam -> lam + i Om periodicity the closure implies.
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

Kmat = np.array([[W[0] ** 2 + KAP, -KAP], [-KAP, W[1] ** 2 + KAP]])
Cmat = np.diag([2 * ZET[0], 2 * ZET[1]])


def Ghat(s):
    """e_1^T (I s^2 + C s + K)^{-1} e_1, s complex."""
    A = np.eye(2) * s**2 + Cmat * s + Kmat
    return np.linalg.inv(A)[0, 0]


# ------------------------------------------------------------------ HB solve
def _residual(coeffs, force, Om):
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
    res = (a + 2.0 * zc * v + jnp.asarray(Kmat) @ x
           + jnp.zeros_like(x).at[0].set(BETA * x[0] ** 3)
           - jnp.zeros_like(x).at[0].set(force * jnp.cos(Om * t)))
    dt = period / NT
    return jnp.stack([res @ cos.T * dt, res @ sin.T * dt], axis=-1).reshape(-1)


_res = jax.jit(_residual)
_jac = jax.jit(jax.jacfwd(_residual))


def solve_hb(force, Om, guess):
    z = np.asarray(guess, float).copy()
    for _ in range(400):
        r = np.asarray(_res(jnp.asarray(z), force, Om))
        nr = np.linalg.norm(r)
        if nr < 1e-14:
            break
        J = np.asarray(_jac(jnp.asarray(z), force, Om))
        try:
            dz = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam, ok = 1.0, False
        for _ in range(60):
            if np.linalg.norm(np.asarray(
                    _res(jnp.asarray(z + lam * dz), force, Om))) < nr:
                ok = True
                break
            lam *= 0.5
        if not ok:
            break
        z = z + lam * dz
    return z, float(np.linalg.norm(np.asarray(_res(jnp.asarray(z), force, Om))))


def x1_series(coeffs):
    a = coeffs.reshape(N, NH, 2)[0]
    return a[:, 0].copy(), a[:, 1].copy()   # cos, sin coefficients


# ------------------------------------------------- ground truth: monodromy
def monodromy(cc, ss, Om, nstep=40000):
    """RK4 integrate dz/dt = A(t) z over one period, z(0)=I_4."""
    T = 2 * np.pi / Om
    h = T / nstep
    ms = np.arange(1, NH + 1)

    def x1(t):
        return float(cc @ np.cos(ms * Om * t) + ss @ np.sin(ms * Om * t))

    def A(t):
        p = 3.0 * BETA * x1(t) ** 2
        Keff = Kmat + p * np.array([[1.0, 0.0], [0.0, 0.0]])
        return np.block([[np.zeros((2, 2)), np.eye(2)],
                         [-Keff, -Cmat]])

    Z = np.eye(4)
    t = 0.0
    for _ in range(nstep):
        k1 = A(t) @ Z
        k2 = A(t + h / 2) @ (Z + h / 2 * k1)
        k3 = A(t + h / 2) @ (Z + h / 2 * k2)
        k4 = A(t + h) @ (Z + h * k3)
        Z = Z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return Z


# ------------------------------------------------- scalar comb Floquet test
def comb_sigma_min(lam, cc, ss, Om, M=24):
    """sigma_min of [1 + Gcal(lam) Pcal] on comb indices -M..M."""
    # p(t) = 3 beta1 x1*(t)^2 : trig polynomial of degree 2*NH -> exact FFT
    ntt = 8 * NH + 8
    tt = np.arange(ntt) * (2 * np.pi / Om) / ntt
    ms = np.arange(1, NH + 1)
    x1t = np.cos(np.outer(tt, ms * Om)) @ cc + np.sin(np.outer(tt, ms * Om)) @ ss
    pt = 3.0 * BETA * x1t**2
    pk_full = np.fft.fft(pt) / ntt          # index k -> coefficient of e^{i k Om t}

    def pk(k):
        return pk_full[k % ntt]

    idx = np.arange(-M, M + 1)
    P = np.array([[pk(n - m) for m in idx] for n in idx])
    G = np.array([Ghat(lam + 1j * n * Om) for n in idx])
    Mtx = np.eye(len(idx)) + G[:, None] * P
    return float(np.linalg.svd(Mtx, compute_uv=False)[-1])


# ============================================================= run the test
Om, F = 1.20, 0.30
print("=" * 78)
print(f"FLOQUET COMB CLOSURE TEST   Om={Om}  F={F}  beta1={BETA}  kappa={KAP}")
print("=" * 78)

# find coexisting solutions from spread initial guesses
sols, seen = [], []
rng = np.random.default_rng(0)
for trial in range(60):
    g = rng.normal(scale=0.5, size=N * NH * 2)
    z, rn = solve_hb(F, Om, g)
    if rn > 1e-12:
        continue
    a = abs(z.reshape(N, NH, 2)[0, 0, 0] - 1j * z.reshape(N, NH, 2)[0, 0, 1])
    if all(abs(a - b) > 1e-6 for b in seen):
        seen.append(a)
        sols.append(z)
order = np.argsort(seen)
sols = [sols[i] for i in order]
seen = [seen[i] for i in order]
print(f"\nfound {len(sols)} coexisting periodic solutions: "
      + ", ".join(f"|A1|={a:.5f}" for a in seen))

for si, z in enumerate(sols):
    cc, ss = x1_series(z)
    T = 2 * np.pi / Om
    Phi = monodromy(cc, ss, Om)
    Phi2 = monodromy(cc, ss, Om, nstep=80000)
    mu = np.linalg.eigvals(Phi)
    mu2 = np.linalg.eigvals(Phi2)
    conv = np.max(np.abs(np.sort_complex(mu) - np.sort_complex(mu2)))
    lam = np.log(mu) / T
    print(f"\n--- branch {si+1}:  |A1| = {seen[si]:.5f} "
          f"(monodromy converged to {conv:.1e}) ---")
    print(f"  multipliers |mu| = {', '.join(f'{abs(m):.6f}' for m in mu)}"
          f"   -> {'STABLE' if max(abs(mu)) < 1 else 'UNSTABLE'}")
    print(f"  {'Floquet exponent lam':>34}   {'sigma_min(comb)':>16}"
          f"   {'control (lam+0.05)':>19}")
    for L in lam:
        s_true = comb_sigma_min(L, cc, ss, Om)
        s_ctrl = comb_sigma_min(L + 0.05, cc, ss, Om)
        print(f"  {L.real:+.9f} {L.imag:+.9f}i   {s_true:16.3e}"
              f"   {s_ctrl:19.3e}")

# ------------------------------------------------ periodicity lam -> lam+iOm
print("\n" + "=" * 78)
print("IMPLIED PERIODICITY  sigma_min(lam) = sigma_min(lam + i Om)")
print("=" * 78)
cc, ss = x1_series(sols[0])
L0 = np.log(np.linalg.eigvals(monodromy(cc, ss, Om)))[0] / (2 * np.pi / Om)
for shift in [0, 1, 2, -1]:
    v = comb_sigma_min(L0 + 1j * shift * Om, cc, ss, Om)
    print(f"  lam + {shift:+d} i Om :  sigma_min = {v:.3e}")

# ------------------------------------------------ comb truncation convergence
print("\n" + "=" * 78)
print("COMB TRUNCATION CONVERGENCE of sigma_min at a true exponent")
print("=" * 78)
for Mt in [6, 10, 16, 24, 32]:
    v = comb_sigma_min(L0, cc, ss, Om, M=Mt)
    print(f"  M = {Mt:3d} :  sigma_min = {v:.3e}")
