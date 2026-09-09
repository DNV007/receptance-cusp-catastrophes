"""Floquet stability of all three coexisting orbits, seeded from continuation.

floquet_ndof.check seeds each branch with a bare first-harmonic guess
(seed[0] = sqrt(y) from the closed form).  That is adequate for a wide
bistable tongue but collapses when the tongue is narrow: on the
omega_2 = 0.95 lips at kappa = 0.333 the tongue is only ~2% wide in F, and
the middle and upper seeds both converge to the same orbit.

Here the three branches are taken from the pseudo-arclength continuation
itself, which traverses all of them in one sweep, so no branch-specific
guess is needed.  The middle branch (between the two folds) must come out
unstable for the window to be a genuine saddle-node pair.

Validated against the 2-DOF baseline window, where the manuscript reports
spectral radii ~0.916.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, _continue_F, _solve_point  # noqa: E402
from floquet_ndof import spectral_radius  # noqa: E402


def _continue_states(S, K, Om, ds=0.004, n_steps=6000, F_max=None):
    """Arclength continuation retaining the full state at each step."""
    n = S["n"]
    u0 = np.zeros(n + 1)
    u0[n] = 0.005
    u, rn = _solve_point(S, u0, K, Om)
    if not np.isfinite(rn) or rn > 1e-8:
        return None
    from cusp_tip_intercept import _tangent
    tau = _tangent(S, u, K, Om)
    if tau[n] < 0:
        tau = -tau
    states, taus = [], []
    for _ in range(n_steps):
        states.append(u.copy())
        taus.append(tau[n])
        up = u + ds * tau
        v = up.copy()
        ok = False
        for _ in range(50):
            r = S["resid"](v, K, Om)
            g = np.dot(tau, v - up)
            Fv = np.concatenate([r, [g]])
            if np.linalg.norm(Fv) < 1e-10:
                ok = True
                break
            J = S["jac"](v, K, Om)
            Jaug = np.vstack([J, tau[None, :]])
            try:
                v = v - np.linalg.solve(Jaug, Fv)
            except np.linalg.LinAlgError:
                ok = False
                break
        if not ok:
            break
        tau = _tangent(S, v, K, Om, prev=tau)
        u = v
        if F_max is not None and u[n] > F_max:
            break
    return np.array(states), np.array(taus)


def branches_at_F(states, taus, F_target, n):
    """States on the trajectory whose forcing crosses F_target."""
    F = states[:, n]
    out = []
    for i in range(len(F) - 1):
        if (F[i] - F_target) * (F[i + 1] - F_target) <= 0 and F[i] != F[i + 1]:
            w = (F_target - F[i]) / (F[i + 1] - F[i])
            out.append(states[i] + w * (states[i + 1] - states[i]))
    return out


def check(label, omegas, zetas, K, Om, F_target, beta1=0.147, nh=7, nt=512,
          ds=0.004):
    N = len(omegas)
    S = make_solver(N, nh, nt, zetas, beta1)
    res = _continue_states(S, K, Om, ds=ds, F_max=3.0 * F_target)
    if res is None:
        print(f"{label}: continuation failed")
        return
    states, taus = res
    cand = branches_at_F(states, taus, F_target, S["n"])
    print(f"\n{label}: Om={Om}  F={F_target:.5f}  -> {len(cand)} coexisting "
          f"orbit(s)")
    seen = []
    for u in cand:
        u2 = u.copy()
        u2[S["n"]] = F_target
        u2, rn = _solve_point(S, u2, K, Om)
        coeffs = u2[:S["n"]]
        c = coeffs.reshape(N, nh, 2)
        A1 = float(np.hypot(c[0, 0, 0], c[0, 0, 1]))
        if any(abs(A1 - a) < 1e-6 for a in seen):
            continue
        seen.append(A1)
        sr, _ = spectral_radius(coeffs, omegas, zetas, K, beta1, Om, nh)
        verdict = "STABLE" if sr < 1.0 else "UNSTABLE"
        print(f"   A1={A1:.4f} (res {rn:.1e})  spectral radius={sr:.4f}  "
              f"{verdict}")


if __name__ == "__main__":
    # sanity: 2-DOF baseline window, manuscript reports ~0.916/0.917
    K0 = np.array([[1.0 + 0.10, -0.10], [-0.10, 1.25**2 + 0.10]])
    check("2-DOF baseline window (sanity)", [1.0, 1.25], [0.015, 0.02],
          K0, 1.10, 0.050, nh=5)

    # omega_2 = 0.95 lips at kappa = 0.333, inside the born lens
    kap = 0.333
    Kl = np.array([[1.0 + kap, -kap], [-kap, 0.95**2 + kap]])
    for Om, F in ((1.030, 0.28774), (1.045, 0.47959)):
        check(f"lips omega_2=0.95, kappa={kap}", [1.0, 0.95], [0.015, 0.02],
              Kl, Om, F, nh=7)
