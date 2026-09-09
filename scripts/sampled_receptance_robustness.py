"""How well does the birth survive being read off a *sampled, noisy* receptance?

The Letter motivates the construction by noting that G is cheap to obtain from
a small-signal sweep.  A sweep does not return G(Omega) in closed form: it
returns samples on a finite frequency grid, each carrying measurement noise.
The birth conditions f = d_Om f = 0 involve first and second derivatives of
f = rho^2 - 3 sigma^2, so it is fair to ask whether they survive that.

Two reconstructions are compared, both fed the same noisy samples:

  raw       : cubic-spline interpolation of Re G and Im G, derivatives taken
              from the spline.  The naive route.
  rational  : least-squares fit of a rational FRF model N(Om)/D(Om) with
              complex coefficients, of exactly the degree a two-degree-of-
              freedom driving-point receptance has, derivatives taken
              analytically from the fit.  This is the standard rational fit of
              experimental modal analysis and the same "fitted causal model"
              the End Matter already invokes for the Floquet continuation.
              A real-residue modal sum is NOT used: the absorber is
              non-proportionally damped, so its residues are complex and a
              positive-real-residue fit carries a systematic 3e-4 bias in
              kappa* that swamps the noise being measured.

Ground truth is the exact two-mode absorber, whose birth sits at
(Omega*, kappa*) = (1.3008787, 0.11895895).

Output: data/sampled_receptance_robustness.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.02
OM_TRUE, KAP_TRUE = 1.3008786700785653, 0.11895895342472275
BAND = (1.24, 1.38)            # sweep band around the birth
KAP_HALF = 0.010               # coupling half-range probed around kappa*
N_REAL = 200                   # noise realizations per configuration


def G_exact(Om, kap):
    K = np.array([[W1 ** 2 + kap, -kap], [-kap, W2 ** 2 + kap]])
    Z = K - Om ** 2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def f_of(g):
    inv = 1.0 / g
    return inv.real ** 2 - 3.0 * inv.imag ** 2


# --------------------------------------------------------------------------
def rational_fit(oms, gs, num_deg=2, den_deg=4, n_iter=6):
    """Fit G ~ N(Om)/D(Om) with complex coefficients (Sanathanan-Koerner).

    A two-degree-of-freedom driving-point receptance is exactly rational of
    this degree in Omega, so with no noise the fit is exact.  Unlike a
    real-residue modal sum it does not assume proportional damping, and it is
    the standard rational FRF fit of experimental modal analysis.
    """
    A_num = np.vander(oms, num_deg + 1, increasing=True)
    A_den = np.vander(oms, den_deg + 1, increasing=True)
    w = np.ones_like(oms, dtype=float)
    coef = None
    for _ in range(n_iter):
        # minimise |N - G D|^2 / |D_prev|^2 with D normalised by d0 = 1
        M = np.hstack([A_num / w[:, None],
                       -(gs[:, None] * A_den[:, 1:]) / w[:, None]])
        rhs = gs * A_den[:, 0] / w
        coef, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        den = np.concatenate([[1.0], coef[num_deg + 1:]])
        w = np.abs(A_den @ den)
        w = np.maximum(w, 1e-12 * np.max(w))
    num = coef[:num_deg + 1]
    den = np.concatenate([[1.0], coef[num_deg + 1:]])
    def Gfit(x):
        x = np.asarray(x, dtype=float)
        Vn = np.vander(x, num_deg + 1, increasing=True)
        Vd = np.vander(x, den_deg + 1, increasing=True)
        return (Vn @ num) / (Vd @ den)
    return Gfit


def modal_fit(oms, gs, n_modes=2, seed_w=(1.03, 1.30)):
    """Fit sum_m phi_m^2/(w_m^2 - Om^2 + 2 i z_m w_m Om) to sampled G."""
    def unpack(p):
        w = np.abs(p[0:n_modes])
        a = p[n_modes:2 * n_modes] ** 2          # phi^2 >= 0 by construction
        z = np.abs(p[2 * n_modes:3 * n_modes])
        return w, a, z

    def model(p, x):
        w, a, z = unpack(p)
        return sum(a[m] / (w[m] ** 2 - x ** 2 + 2j * z[m] * w[m] * x)
                   for m in range(n_modes))

    def resid(p):
        d = model(p, oms) - gs
        return np.concatenate([d.real, d.imag])

    p0 = np.concatenate([np.array(seed_w[:n_modes]),
                         np.full(n_modes, 0.8),
                         np.full(n_modes, 0.02)])
    sol = least_squares(resid, p0, xtol=1e-14, ftol=1e-14, max_nfev=20000)
    w, a, z = unpack(sol.x)
    return lambda x: sum(a[m] / (w[m] ** 2 - np.asarray(x) ** 2
                                 + 2j * z[m] * w[m] * np.asarray(x))
                         for m in range(n_modes))


def birth_from_curves(f_of_om, kaps, oms):
    """Locate f = d_Om f = 0 on a tabulated f(Om; kappa).

    For each kappa, the tangency of the level set is the minimum over Omega of
    f (the birth is where that minimum touches zero).  Interpolating
    min_Om f(Om) across kappa and finding its zero gives kappa*, and the
    argmin there gives Omega*.
    """
    mins, argmins = [], []
    for k in kaps:
        fv = f_of_om(oms, k)
        j = int(np.argmin(fv))
        if 0 < j < len(oms) - 1:                 # parabolic refinement
            y0, y1, y2 = fv[j - 1], fv[j], fv[j + 1]
            d = y0 - 2 * y1 + y2
            sh = 0.5 * (y0 - y2) / d if d != 0 else 0.0
            h = oms[1] - oms[0]
            argmins.append(oms[j] + sh * h)
            mins.append(y1 - 0.25 * (y0 - y2) * sh)
        else:
            argmins.append(oms[j]); mins.append(fv[j])
    mins = np.array(mins); argmins = np.array(argmins)
    s = np.where(np.diff(np.sign(mins)) != 0)[0]
    if len(s) == 0:
        return None
    i = s[0]
    t = mins[i] / (mins[i] - mins[i + 1])
    return float(kaps[i] + t * (kaps[i + 1] - kaps[i])), \
           float(argmins[i] + t * (argmins[i + 1] - argmins[i]))


def run(dOm, noise_db, n_real, rng, kap_step=5e-4):
    """Return arrays of (dkappa, dOmega) errors for raw and modal routes."""
    oms = np.arange(BAND[0], BAND[1] + 0.5 * dOm, dOm)
    kaps = np.arange(KAP_TRUE - KAP_HALF, KAP_TRUE + KAP_HALF, kap_step)
    delta = 10 ** (noise_db / 20.0) if noise_db is not None else 0.0
    fine = np.linspace(BAND[0] + 2 * dOm, BAND[1] - 2 * dOm, 1400)
    out = {"raw": [], "modal": [], "rational": []}
    for _ in range(n_real):
        splines, fits = {}, {}
        for k in kaps:
            g = np.array([G_exact(o, k) for o in oms])
            if delta > 0:
                # multiplicative complex perturbation whose relative RMS
                # magnitude is exactly delta: each quadrature gets delta/sqrt2
                g = g * (1 + (delta / np.sqrt(2))
                         * (rng.standard_normal(len(g))
                            + 1j * rng.standard_normal(len(g))))
            splines[k] = (CubicSpline(oms, g.real), CubicSpline(oms, g.imag))
            fits[k] = rational_fit(oms, g)
        def f_raw(x, k):
            cr, ci = splines[k]
            return f_of(cr(x) + 1j * ci(x))
        def f_rat(x, k):
            return f_of(fits[k](x))
        for tag, fn in (("raw", f_raw), ("rational", f_rat)):
            r = birth_from_curves(fn, kaps, fine)
            if r is None:
                continue
            out[tag].append((r[0] - KAP_TRUE, r[1] - OM_TRUE))
    return out


def main():
    rng = np.random.default_rng(20260828)
    # sanity: noiseless, fine grid must recover the exact birth
    base = run(2e-4, None, 1, rng)
    for tag in ("raw", "rational"):
        if base[tag]:
            dk, do = base[tag][0]
            print(f"noiseless dOm=2e-4  {tag:5s}: dkappa={dk:+.2e}  dOmega={do:+.2e}")

    rows = []
    for dOm in (1e-3, 5e-4, 2e-4):
        for ndb in (-80, -60, -40):   # delta = 1e-4, 1e-3, 1e-2 rms
            res = run(dOm, ndb, N_REAL, rng)
            row = dict(dOm=dOm, noise_db=ndb)
            for tag in ("raw", "rational"):
                a = np.array(res[tag])
                if len(a) == 0:
                    row[tag] = None
                    continue
                row[tag] = dict(
                    n=len(a),
                    med_dkappa=float(np.median(np.abs(a[:, 0]))),
                    p05_dkappa=float(np.percentile(np.abs(a[:, 0]), 5)),
                    p95_dkappa=float(np.percentile(np.abs(a[:, 0]), 95)),
                    med_dOmega=float(np.median(np.abs(a[:, 1]))))
            rows.append(row)
            r_, m_ = row["raw"], row["rational"]
            if r_ and m_:
                msg = (f"dOm={dOm:.0e} noise={ndb:+d}dB  "
                       f"raw |dk|={r_['med_dkappa']:.2e} ({r_['n']}/{N_REAL})  "
                       f"rational |dk|={m_['med_dkappa']:.2e} ({m_['n']}/{N_REAL})")
            else:
                msg = f"dOm={dOm:.0e} noise={ndb:+d}dB  incomplete"
            print(msg, flush=True)

    os.makedirs(DATA, exist_ok=True)
    json.dump(dict(om_true=OM_TRUE, kap_true=KAP_TRUE, band=BAND,
                   kap_half=KAP_HALF, rows=rows),
              open(os.path.join(DATA, "sampled_receptance_robustness.json"), "w"),
              indent=2)
    print("\nwrote data/sampled_receptance_robustness.json")


if __name__ == "__main__":
    main()
