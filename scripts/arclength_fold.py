"""Pseudo-arclength continuation of the general-N HB orbit in the forcing F at
fixed drive frequency Om. Robustly tracks through saddle-node folds (turning
points in F) that a naive F- or Om-sweep falls off of. A bistable window exists
where the continuation has >=2 folds (an S-curve); counting folds vs coupling
detects a receptance-predicted cusp-pair birth in the TRUE ODEs.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from ndof_hb import _residual_jax

jax.config.update("jax_enable_x64", True)


def make_continuation(omegas, zetas, beta1, Om, nh=6, nt=512):
    N = len(omegas)
    zc = jnp.asarray(zetas, float)
    n = N * nh * 2

    def R_of_u(u, K):
        z = u[:n]; F = u[n]
        return _residual_jax(z, K, zc, beta1, F, Om, N, nh, nt)

    R_jit = jax.jit(lambda u, K: R_of_u(u, K))
    dR_jit = jax.jit(jax.jacfwd(lambda u, K: R_of_u(u, K)))  # (n, n+1)

    def residual(u, K):
        return np.asarray(R_jit(jnp.asarray(u), jnp.asarray(K)))

    def jac(u, K):
        return np.asarray(dR_jit(jnp.asarray(u), jnp.asarray(K)))

    def solve_point(u0, K, tol=1e-11, it=60):
        """Newton to R=0 with F fixed (u0[n] held), refine z only."""
        u = u0.copy()
        Kj = jnp.asarray(K)
        for _ in range(it):
            r = np.asarray(R_jit(jnp.asarray(u), Kj))
            if np.linalg.norm(r) < tol:
                break
            J = np.asarray(dR_jit(jnp.asarray(u), Kj))[:, :n]
            u[:n] -= np.linalg.solve(J, r)
        return u, float(np.linalg.norm(np.asarray(R_jit(jnp.asarray(u), Kj))))

    def tangent(u, K, prev=None):
        J = jac(u, K)              # (n, n+1)
        # null vector of J
        _, _, Vt = np.linalg.svd(J)
        tau = Vt[-1]
        tau /= np.linalg.norm(tau)
        if prev is not None and np.dot(tau, prev) < 0:
            tau = -tau
        return tau

    def continue_F(K, u_start, ds=0.02, n_steps=400, F_max=None, F_min=0.0):
        """Arclength-continue from a converged start; return array of (F, A1, tauF)."""
        Kj = jnp.asarray(K)
        u, rn = solve_point(u_start, K)
        if rn > 1e-7:
            return None
        tau = tangent(u, K)
        if tau[n] < 0:          # start going in +F direction
            tau = -tau
        out = []
        for _ in range(n_steps):
            a1 = float(np.hypot(u[0], u[1]))
            out.append((u[n], a1, tau[n]))
            up = u + ds * tau
            # corrector: R(u)=0 and tau^T (u-up)=0
            v = up.copy()
            ok = False
            for _ in range(40):
                r = np.asarray(R_jit(jnp.asarray(v), Kj))
                g = np.dot(tau, v - up)
                F = np.concatenate([r, [g]])
                if np.linalg.norm(F) < 1e-10:
                    ok = True
                    break
                J = np.asarray(dR_jit(jnp.asarray(v), Kj))       # (n, n+1)
                Jaug = np.vstack([J, tau[None, :]])              # (n+1, n+1)
                v = v - np.linalg.solve(Jaug, F)
            if not ok:
                break
            newtau = tangent(v, K, prev=tau)
            u, tau = v, newtau
            if (F_max is not None and u[n] > F_max) or u[n] < F_min:
                break
        return np.array(out)

    return dict(N=N, n=n, nh=nh, solve_point=solve_point, continue_F=continue_F)


def count_folds(traj):
    """Number of turning points in F (sign changes of the tangent's F-component)."""
    if traj is None or len(traj) < 3:
        return 0, traj
    tauF = traj[:, 2]
    return int(np.sum(np.abs(np.diff(np.sign(tauF))) > 0)), traj
