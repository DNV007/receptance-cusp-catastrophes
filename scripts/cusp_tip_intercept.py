"""Locate cusp tips by the (Delta F)^{2/3} intercept method.

Motivation
----------
`hh_cusp.sn_lips_jax.find_cusp_tip_jax` locates a cusp tip as the Omega at
which Newton on the augmented saddle-node system stops converging, bisected
to 1e-7.  That is a convergence-failure boundary, not the cusp: its distance
from the true tip depends on conditioning, and therefore drifts with the
harmonic truncation.  Re-running kappa=0.13 at N_HARM = 2, 3, 4 moves the
tip by up to 5e-3, i.e. ~1e4 times the quoted bracket.

This module instead measures the tongue width Delta F(Omega) at points where
both saddle-nodes are well conditioned, and obtains the tip as the intercept
of the linear law (Delta F)^{2/3} ~ (Omega - Omega_c) implied by the cusp
normal form.  No solve is ever attempted at the tip.  This is the estimator
the manuscript already validates for the lower cusp (Sec. IV.B).

Both fold forcings at a given Omega come from one pseudo-arclength sweep in
F, so no branch-steering heuristics are needed.

Public API
----------
    solver = make_solver(N, nh, nt, zetas, beta1)
    F_folds(solver, K, Omega)              -> sorted list of fold forcings
    tongue_width(solver, K, Omega)         -> Delta F  (or nan)
    cusp_tip(solver, K, Omega_pred, side)  -> dict with Omega_tip, fit stats
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndof_hb import _residual_jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# solver construction:  Omega is a traced argument, so one compile serves all
# frequencies (make_continuation in arclength_fold.py closes over Om and so
# recompiles per frequency).
# --------------------------------------------------------------------------
def make_solver(N, nh, nt, zetas, beta1):
    zc = jnp.asarray(zetas, float)
    n = N * nh * 2

    def R(u, K, Om):
        return _residual_jax(u[:n], K, zc, beta1, u[n], Om, N, nh, nt)

    R_j = jax.jit(R)
    dR_j = jax.jit(jax.jacfwd(R))          # d/du -> (n, n+1)

    def resid(u, K, Om):
        return np.asarray(R_j(jnp.asarray(u), jnp.asarray(K), float(Om)))

    def jac(u, K, Om):
        return np.asarray(dR_j(jnp.asarray(u), jnp.asarray(K), float(Om)))

    return dict(N=N, nh=nh, nt=nt, n=n, resid=resid, jac=jac)


def _solve_point(S, u0, K, Om, tol=1e-11, maxit=80):
    """Newton on R(z)=0 at fixed F."""
    n = S["n"]
    u = np.asarray(u0, float).copy()
    for _ in range(maxit):
        r = S["resid"](u, K, Om)
        rn = np.linalg.norm(r)
        if rn < tol:
            return u, rn
        J = S["jac"](u, K, Om)[:, :n]
        try:
            u[:n] -= np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return u, np.inf
    return u, float(np.linalg.norm(S["resid"](u, K, Om)))


def _tangent(S, u, K, Om, prev=None):
    J = S["jac"](u, K, Om)
    _, _, Vt = np.linalg.svd(J)
    tau = Vt[-1]
    tau /= np.linalg.norm(tau)
    if prev is not None and np.dot(tau, prev) < 0:
        tau = -tau
    return tau


def _continue_F(S, K, Om, F_start=0.005, ds=0.01, n_steps=3000,
                F_max=None, F_min=0.0):
    """Pseudo-arclength continuation in F from a low-F seed.

    Returns array of rows (F, A1, tauF); folds are sign changes of tauF.
    """
    n = S["n"]
    u0 = np.zeros(n + 1)
    u0[n] = F_start
    u, rn = _solve_point(S, u0, K, Om)
    if not np.isfinite(rn) or rn > 1e-8:
        return None
    tau = _tangent(S, u, K, Om)
    if tau[n] < 0:
        tau = -tau                      # head towards increasing F
    out = []
    for _ in range(n_steps):
        out.append((u[n], float(np.hypot(u[0], u[1])), tau[n]))
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
        if (F_max is not None and u[n] > F_max) or u[n] < F_min:
            break
    return np.array(out) if len(out) > 2 else None


def F_folds(S, K, Omega, **kw):
    """Fold forcings at fixed Omega, from tangent-F sign changes (interpolated)."""
    traj = _continue_F(S, K, Omega, **kw)
    if traj is None:
        return []
    F, tauF = traj[:, 0], traj[:, 2]
    folds = []
    sgn = np.sign(tauF)
    idx = np.where(np.abs(np.diff(sgn)) > 0)[0]
    for i in idx:
        t0, t1 = tauF[i], tauF[i + 1]
        if t1 == t0:
            folds.append(float(F[i]))
        else:
            w = t0 / (t0 - t1)           # linear interp of tauF -> 0
            folds.append(float(F[i] + w * (F[i + 1] - F[i])))
    return sorted(folds)


def tongue_width(S, K, Omega, **kw):
    f = F_folds(S, K, Omega, **kw)
    if len(f) < 2:
        return float("nan")
    return float(f[-1] - f[0])


def cusp_tip(S, K, Omega_pred, side, span=0.020, inner=0.002, n_pts=12,
             min_keep=5, verbose=False, **kw):
    """Cusp tip by linear fit of (Delta F)^{2/3} against Omega.

    side = -1 : bistable region lies at Omega < Omega_tip (approach from below)
    side = +1 : bistable region lies at Omega > Omega_tip (approach from above)
    """
    if side < 0:
        oms = Omega_pred - np.linspace(inner, span, n_pts)
    else:
        oms = Omega_pred + np.linspace(inner, span, n_pts)

    rows = []
    for om in oms:
        dF = tongue_width(S, K, float(om), **kw)
        if verbose:
            print(f"      Om={om:.6f}  dF={dF:.6e}", flush=True)
        if np.isfinite(dF) and dF > 0:
            rows.append((float(om), dF))
    if len(rows) < min_keep:
        return None

    om = np.array([r[0] for r in rows])
    dF = np.array([r[1] for r in rows])
    y = dF ** (2.0 / 3.0)
    # (Delta F)^{2/3} is linear in Omega near the tip; intercept y=0 -> tip.
    A = np.vstack([om, np.ones_like(om)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    if slope == 0:
        return None
    Omega_tip = -intercept / slope
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(
        Omega_tip=float(Omega_tip),
        slope=float(slope),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_pts=len(rows),
        omegas=om.tolist(),
        dF=dF.tolist(),
        Omega_pred=float(Omega_pred),
        side=int(side),
    )


def K_2dof(kappa, w1=1.0, w2=1.25):
    return np.array([[w1 ** 2 + kappa, -kappa],
                     [-kappa, w2 ** 2 + kappa]])
