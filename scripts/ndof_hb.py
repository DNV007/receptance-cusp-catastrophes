"""
General N-DOF harmonic-balance (alternating frequency-time) for a linear chain
with drive + cubic on mass 1 only:

    M xddot + C xdot + K x + beta1 x1^3 e1 = F cos(Om t) e1,   M = I,
    C = diag(2 zeta_i),  K = chain stiffness (nearest-neighbour coupling kappa).

Independent of the Part-I 2-DOF solver (which is frozen in the submission
package). Used to VALIDATE, in the true ODEs, the receptance-predicted
higher-rung cusp-pair births found by the leading-order analysis.

Fold (saddle-node) detection: the square coefficient Jacobian dR/dcoeffs is
nonsingular at a regular periodic orbit and singular at a saddle-node, so its
determinant changes sign across a fold in F (at fixed Om) or Om (at fixed F).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)


def build_K(omegas, kappa):
    N = len(omegas)
    K = np.diag(np.asarray(omegas, float) ** 2).copy()
    for i in range(N - 1):
        K[i, i] += kappa
        K[i + 1, i + 1] += kappa
        K[i, i + 1] -= kappa
        K[i + 1, i] -= kappa
    return K


def _residual_jax(coeffs, K, zetas, beta1, force, Om, N, nh, nt):
    coeffs = coeffs.reshape(N, nh, 2)
    c = coeffs[:, :, 0]           # (N, nh) cosine
    s = coeffs[:, :, 1]           # (N, nh) sine
    period = 2.0 * jnp.pi / Om
    t = jnp.arange(nt) * (period / nt)
    ms = jnp.arange(1, nh + 1)
    w = ms * Om                   # (nh,)
    wt = jnp.outer(ms, Om * t)    # (nh, T)
    cos = jnp.cos(wt)             # (nh, T)
    sin = jnp.sin(wt)

    x = c @ cos + s @ sin                              # (N, T)
    v = (-(c * w)) @ sin + (s * w) @ cos               # (N, T)
    a = (-(c * w ** 2)) @ cos + (-(s * w ** 2)) @ sin  # (N, T)

    zc = jnp.asarray(zetas)[:, None]
    lin = K @ x                                        # (N, T)
    cubic = jnp.zeros_like(x).at[0].set(beta1 * x[0] ** 3)
    drive = jnp.zeros_like(x).at[0].set(force * jnp.cos(Om * t))
    res = a + 2.0 * zc * v + lin + cubic - drive       # (N, T)

    dt = period / nt
    proj_c = res @ cos.T * dt      # (N, nh)
    proj_s = res @ sin.T * dt      # (N, nh)
    return jnp.stack([proj_c, proj_s], axis=-1).reshape(-1)


def make_funcs(omegas, zetas, beta1, nh=3, nt=256):
    N = len(omegas)
    zc = jnp.asarray(zetas, float)

    def resid(coeffs, K, force, Om):
        return _residual_jax(jnp.asarray(coeffs), jnp.asarray(K), zc,
                             beta1, force, Om, N, nh, nt)

    resid_jit = jax.jit(lambda coeffs, K, force, Om:
                        _residual_jax(coeffs, K, zc, beta1, force, Om, N, nh, nt))
    jac_jit = jax.jit(jax.jacfwd(
        lambda coeffs, K, force, Om:
        _residual_jax(coeffs, K, zc, beta1, force, Om, N, nh, nt)))

    def solve(K, force, Om, guess=None):
        n = N * nh * 2
        if guess is None:
            guess = np.zeros(n)
            guess[0] = force / max(1e-6, abs(omegas[0] ** 2 - Om ** 2))
        Kj = jnp.asarray(K)
        r = lambda z: np.asarray(resid_jit(jnp.asarray(z), Kj, force, Om))
        j = lambda z: np.asarray(jac_jit(jnp.asarray(z), Kj, force, Om))
        sol = least_squares(r, guess, jac=j, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=400)
        return sol.x, float(np.linalg.norm(sol.fun))

    def jac_det_sign(coeffs, K, force, Om):
        J = np.asarray(jac_jit(jnp.asarray(coeffs), jnp.asarray(K), force, Om))
        sign, logdet = np.linalg.slogdet(J)
        return sign, logdet

    def amp1(coeffs):
        return float(np.hypot(coeffs.reshape(N, nh, 2)[0, 0, 0],
                              coeffs.reshape(N, nh, 2)[0, 0, 1]))

    return dict(N=N, nh=nh, solve=solve, jac_det_sign=jac_det_sign, amp1=amp1)


def sweep_amp(fns, K, force, Om_lo, Om_hi, n, reverse=False):
    """Warm-started Om sweep at fixed force; returns (Om, A1, detsign) arrays."""
    Oms = np.linspace(Om_lo, Om_hi, n)
    if reverse:
        Oms = Oms[::-1]
    out = []
    guess = None
    for Om in Oms:
        x, rn = fns["solve"](K, force, Om, guess=guess)
        if rn > 1e-6:
            guess = None
            x, rn = fns["solve"](K, force, Om, guess=None)
        guess = x
        sgn, _ = fns["jac_det_sign"](x, K, force, Om)
        out.append((Om, fns["amp1"](x), sgn, rn))
    return np.array(out)
