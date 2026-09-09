"""Stress test: what happens to the receptance cusp prediction when 3*Omega
sits on a resonance of the network.

The Letter states that the leading third-harmonic correction to the cusp
condition is controlled by the comb gain eta_3 = (sqrt3/9)|G(3Om)/G(Om)|, and
that the construction "can become large if 3*Omega approaches another
resonance".  That caveat was asserted, not measured.  This script measures it.

Construction
------------
We need a network in which (i) the cusp itself sits at large anharmonicity, so
|G(Om_c)| is small, and (ii) a resonance with real participation sits at
3*Om_c, so |G(3Om_c)| is large.  A grounded-spring chain cannot do this: a mode
three octaves up has negligible participation at the driven coordinate, and
coupling to it strongly pushes the driven resonance up with it.

We therefore specify the receptance in MODAL form,

    G(Om) = sum_m  phi_m^2 / (w_m^2 - Om^2 + 2 i zeta_m w_m Om),

which for phi_m^2 > 0, zeta_m > 0 is exactly a passive collocated
driving-point receptance.  The corresponding equations of motion are

    qddot_m + 2 zeta_m w_m qdot_m + w_m^2 q_m
        + phi_m beta1 x1^3 = phi_m F cos(Om t),      x1 = sum_m phi_m q_m,

i.e. M = I, C = diag(2 zeta_m w_m), K = diag(w_m^2) with the drive and the
cubic both along the single vector phi.  This is the Letter's rank-one
collocated setting, so every condition in the Letter applies unchanged.

Modes 1-2 place a cusp at eps ~ 1, matching the hardest member of the Letter's
hardening family.  Mode 3 is the probe: sweeping w3 through 3*Om_c sweeps the
comb gain over roughly two decades at fixed cusp geometry.

For each w3 we compare three locations of the same cusp:
  * fundamental        : arg G = -150 deg           (Eq. 3 of the Letter)
  * comb-corrected     : double root of the fold cubic (Eq. S8)
  * multi-harmonic     : (Delta F)^{2/3} tongue-width intercept, which never
                         solves at the tip and never evaluates a cusp condition

Output: data/stress_3omega_resonance.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, fsolve

jax.config.update("jax_enable_x64", True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BETA1 = 0.147
# modes 1-2: cusp near the antiresonance at eps ~ 1
BASE_MODES = [(1.00, 0.70, 0.015), (1.30, 0.30, 0.020)]
P3, Z3 = 0.05, 0.005          # probe participation and damping


# --------------------------------------------------------------------------
# receptance-level predictions
# --------------------------------------------------------------------------
def modes_of(w3):
    return BASE_MODES + ([] if w3 is None else [(w3, P3, Z3)])


def G(Om, modes):
    return sum(p / (w ** 2 - Om ** 2 + 2j * z * w * Om) for w, p, z in modes)


def f_fund(Om, modes):
    g = 1.0 / G(Om, modes)
    return g.real ** 2 - 3.0 * g.imag ** 2


def _fold_cubic_coeffs(Om, modes, beta1):
    """Coefficients (highest power first) of the O(G3) fold cubic, Eq. (S8)."""
    c, q = 0.75 * beta1, 3.0 / 16.0 * beta1 ** 2
    g1 = 1.0 / G(Om, modes)
    rho, sig = g1.real, g1.imag
    g3 = G(3.0 * Om, modes)
    gr, gi = g3.real, g3.imag
    return [-8 * q * c * gr,
            3 * c ** 2 - 6 * q * (gr * rho + gi * sig),
            4 * c * rho,
            rho ** 2 + sig ** 2]


def f_comb(Om, modes, beta1=BETA1):
    """SM normalisation (9/4)c^2 (y+ - y-)^2 over the two physical roots."""
    c, q = 0.75 * beta1, 3.0 / 16.0 * beta1 ** 2
    r = np.roots(_fold_cubic_coeffs(Om, modes, beta1))
    # the spurious root runs off as y ~ 3c/(8 q g_r); drop the root nearest it
    gr = G(3.0 * Om, modes).real
    runaway = 3 * c / (8 * q * gr) if gr != 0 else np.inf
    idx = np.argsort(np.abs(r - runaway))[1:]
    y1, y2 = r[idx]
    return float((2.25 * c ** 2 * (y1 - y2) ** 2).real)


def cusps(fn, modes, lo, hi, n=4001):
    xs = np.linspace(lo, hi, n)
    ys = [fn(x, modes) for x in xs]
    out = []
    for i in range(n - 1):
        if ys[i] * ys[i + 1] < 0:
            r = brentq(fn, xs[i], xs[i + 1], args=(modes,), xtol=1e-14)
            if (1.0 / G(r, modes)).real < 0:      # beta1 > 0 physical branch
                out.append(r)
    return out


# --------------------------------------------------------------------------
# multi-harmonic balance for the modal system
# --------------------------------------------------------------------------
def make_solver(modes, nh, nt, beta1):
    ws = jnp.array([m[0] for m in modes])
    ph = jnp.array([np.sqrt(m[1]) for m in modes])
    zs = jnp.array([m[2] for m in modes])
    N = len(modes)
    n = N * nh * 2

    def R(u, Om):
        coeffs = u[:n].reshape(N, nh, 2)
        F = u[n]
        c, s = coeffs[:, :, 0], coeffs[:, :, 1]
        period = 2.0 * jnp.pi / Om
        t = jnp.arange(nt) * (period / nt)
        ms = jnp.arange(1, nh + 1)
        w = ms * Om
        wt = jnp.outer(ms, Om * t)
        cos, sin = jnp.cos(wt), jnp.sin(wt)
        q = c @ cos + s @ sin
        v = (-(c * w)) @ sin + (s * w) @ cos
        a = (-(c * w ** 2)) @ cos + (-(s * w ** 2)) @ sin
        x1 = ph @ q
        cubic = beta1 * x1 ** 3
        res = (a + 2.0 * (zs * ws)[:, None] * v + (ws ** 2)[:, None] * q
               + ph[:, None] * cubic[None, :]
               - ph[:, None] * (F * jnp.cos(Om * t))[None, :])
        dt = period / nt
        return jnp.stack([res @ cos.T * dt, res @ sin.T * dt], axis=-1).reshape(-1)

    R_j = jax.jit(R)
    dR_j = jax.jit(jax.jacfwd(R))
    return dict(n=n, N=N, nh=nh, ph=np.asarray(ph),
                resid=lambda u, Om: np.asarray(R_j(jnp.asarray(u), float(Om))),
                jac=lambda u, Om: np.asarray(dR_j(jnp.asarray(u), float(Om))))


def _solve_point(S, u0, Om, tol=1e-11, maxit=80):
    n = S["n"]
    u = np.asarray(u0, float).copy()
    for _ in range(maxit):
        r = S["resid"](u, Om)
        rn = np.linalg.norm(r)
        if rn < tol:
            return u, rn
        J = S["jac"](u, Om)[:, :n]
        try:
            u[:n] -= np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return u, np.inf
    return u, np.linalg.norm(S["resid"](u, Om))


def _tangent(S, u, Om, prev=None):
    J = S["jac"](u, Om)
    try:
        tau = np.linalg.svd(J)[2][-1]
    except np.linalg.LinAlgError:
        # LAPACK gesdd occasionally fails to converge on these Jacobians at
        # large harmonic truncation; gesvd is slower but more robust, and the
        # nullspace it returns is the same when gesdd does converge.
        from scipy.linalg import svd as _svd
        tau = _svd(J, lapack_driver="gesvd")[2][-1]
    tau /= np.linalg.norm(tau)
    if prev is not None and np.dot(tau, prev) < 0:
        tau = -tau
    return tau


def F_folds(S, Om, F_start=0.002, ds=0.004, n_steps=20000, F_max=None):
    n = S["n"]
    u0 = np.zeros(n + 1)
    u0[n] = F_start
    u, rn = _solve_point(S, u0, Om)
    if not np.isfinite(rn) or rn > 1e-8:
        return []
    tau = _tangent(S, u, Om)
    if tau[n] < 0:
        tau = -tau
    Fs, taus = [], []
    for _ in range(n_steps):
        Fs.append(u[n]); taus.append(tau[n])
        up = u + ds * tau
        v = up.copy()
        ok = False
        for _ in range(50):
            r = S["resid"](v, Om)
            Fv = np.concatenate([r, [np.dot(tau, v - up)]])
            if np.linalg.norm(Fv) < 1e-10:
                ok = True
                break
            try:
                v = v - np.linalg.solve(np.vstack([S["jac"](v, Om), tau[None, :]]), Fv)
            except np.linalg.LinAlgError:
                break
        if not ok:
            break
        tau = _tangent(S, v, Om, prev=tau)
        u = v
        if (F_max is not None and u[n] > F_max) or u[n] < 0.0:
            break
    Fs, taus = np.array(Fs), np.array(taus)
    if len(Fs) < 3:
        return []
    out = []
    sgn = np.sign(taus)
    for i in np.where(np.abs(np.diff(sgn)) > 0)[0]:
        t0, t1 = taus[i], taus[i + 1]
        w = 0.5 if t1 == t0 else t0 / (t0 - t1)
        out.append(float(Fs[i] + w * (Fs[i + 1] - Fs[i])))
    return sorted(out)


def tongue_width(S, Om, **kw):
    fl = F_folds(S, Om, **kw)
    return float(fl[-1] - fl[0]) if len(fl) >= 2 else float("nan")


def bracket_edge(S, Om_in, Om_out, n_bisect=7, **kw):
    """Frequency at which the tongue closes, assuming no exponent.

    Anchoring the fit window on the fundamental prediction fails exactly where
    that prediction is worst, which is the regime this script exists to probe,
    so the window is anchored on the measured bistable/non-bistable transition
    instead.  Om_in must lie inside the tongue and Om_out outside it; both are
    checked, and the transition between them is bisected.
    """
    bist = lambda om: len(F_folds(S, float(om), **kw)) >= 2
    if not bist(Om_in) or bist(Om_out):
        return None
    lo, hi = Om_in, Om_out
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if bist(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def cusp_tip(S, Om_pred, side, span=0.0035, inner=0.0002, n_pts=12,
             Om_alt=None, pad=0.008, **kw):
    """(Delta F)^{2/3} intercept, anchored on the measured tongue edge."""
    cands = [Om_pred] + ([Om_alt] if Om_alt is not None
                         and np.isfinite(Om_alt) else [])
    if side < 0:
        Om_in, Om_out = min(cands) - pad, max(cands) + 0.5 * pad
    else:
        Om_in, Om_out = max(cands) + pad, min(cands) - 0.5 * pad
    edge = bracket_edge(S, Om_in, Om_out, **kw)
    if edge is None:
        return None
    if side < 0:
        oms = edge - np.linspace(inner, span, n_pts)
    else:
        oms = edge + np.linspace(inner, span, n_pts)
    dat = []
    for om in oms:
        dF = tongue_width(S, float(om), **kw)
        if np.isfinite(dF) and dF > 0:
            dat.append((om, dF ** (2.0 / 3.0)))
    if len(dat) < 5:
        return None
    x = np.array([d[0] for d in dat]); y = np.array([d[1] for d in dat])
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - (res[0] / np.sum((y - y.mean()) ** 2) if len(res) else 0.0)
    # model-free bracket: the extreme frequency at which a tongue is still
    # resolved, which assumes no exponent at all
    return dict(Om_tip=float(-b / m), n=len(dat), r2=float(r2),
                edge=float(edge))


# --------------------------------------------------------------------------
def main():
    nh = int(os.environ.get("NH", 5))
    nt = int(os.environ.get("NT", 256))
    base = modes_of(None)
    Om0 = [c for c in cusps(f_fund, base, 1.05, 1.28) if True]
    Om0 = Om0[0]
    print(f"baseline (no probe mode): Om_c = {Om0:.6f}, 3*Om_c = {3*Om0:.4f}, "
          f"eps = {8/9*abs((1/G(Om0, base)).real):.3f}", flush=True)

    w3_list = [float(x) for x in np.round(
        np.array([2.90, 3.20, 3.40, 3.50, 3.545, 3.566, 3.59, 3.65, 3.90, 4.40]), 4)]

    rows = []
    for w3 in w3_list:
        md = modes_of(w3)
        cf = cusps(f_fund, md, 1.05, 1.28)
        cc = cusps(f_comb, md, 1.05, 1.28)
        if not cf:
            print(f"w3={w3}: no fundamental cusp"); continue
        Omf = min(cf, key=lambda o: abs(o - Om0))
        Omc = min(cc, key=lambda o: abs(o - Omf)) if cc else float("nan")
        cg = abs(G(3 * Omf, md) / G(Omf, md))
        eta3 = np.sqrt(3) / 9 * cg
        eps = 8 / 9 * abs((1.0 / G(Omf, md)).real)
        rows.append(dict(w3=w3, Om_fund=float(Omf), Om_comb=float(Omc),
                         comb_gain=float(cg), eta3=float(eta3), eps=float(eps)))
        print(f"  w3={w3:6.3f}  Om_fund={Omf:.6f}  Om_comb={Omc:.6f}  "
              f"eta3={eta3:.4f}  eps={eps:.3f}", flush=True)

    # which side does the tongue lie on?  probe the baseline
    S0 = make_solver(base, nh, nt, BETA1)
    side = None
    for s in (-1, +1):
        dF = tongue_width(S0, Om0 + s * 0.004, F_max=5.0)
        print(f"  probe side {s:+d}: dF = {dF}", flush=True)
        if np.isfinite(dF) and dF > 0:
            side = s
            break
    if side is None:
        print("could not find the bistable side; aborting HB stage")
        json.dump(dict(rows=rows, nh=nh), open(
            os.path.join(DATA, "stress_3omega_resonance.json"), "w"), indent=2)
        return
    print(f"bistable tongue on side {side:+d}; running HB at nh={nh}", flush=True)

    for r in rows:
        md = modes_of(r["w3"])
        S = make_solver(md, nh, nt, BETA1)
        t0 = time.time()
        tip = cusp_tip(S, r["Om_fund"], side, Om_alt=r["Om_comb"], F_max=5.0)
        r["hb"] = tip
        if tip:
            r["err_fund"] = tip["Om_tip"] - r["Om_fund"]
            r["err_comb"] = tip["Om_tip"] - r["Om_comb"]
            print(f"  w3={r['w3']:6.3f} eta3={r['eta3']:.4f}  "
                  f"Om_HB={tip['Om_tip']:.6f} (r2={tip['r2']:.5f}, n={tip['n']})  "
                  f"err_fund={r['err_fund']:+.2e}  err_comb={r['err_comb']:+.2e}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"  w3={r['w3']:6.3f}: tip fit failed [{time.time()-t0:.0f}s]",
                  flush=True)

    os.makedirs(DATA, exist_ok=True)
    json.dump(dict(nh=nh, nt=nt, beta1=BETA1, base_modes=BASE_MODES,
                   p3=P3, z3=Z3, side=side, rows=rows),
              open(os.path.join(DATA, "stress_3omega_resonance.json"), "w"),
              indent=2)
    print("\nwrote data/stress_3omega_resonance.json")


if __name__ == "__main__":
    main()
