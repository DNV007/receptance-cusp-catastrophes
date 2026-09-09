"""Redo the 100-network ensemble with the EXACT damped receptance.

`ensemble_unified.py` (and therefore `ensemble_validity_boundary.py`) builds the
driving-point receptance as a modal sum,

    G(Om) = sum_m  phi_m1^2 / (om_m^2 - Om^2 + 2i z_m Om),
    z_m   = V[:,m] . diag(zeta) . V[:,m],

keeping only the DIAGONAL of the modal damping matrix V^T diag(zeta) V.  The
per-coordinate damping rates are drawn independently on [0.01, 0.04], so the
damping is NOT proportional and that sum is not the exact receptance of the
physical network: V^T diag(zeta) V has off-diagonal entries, and

    G_exact(Om) = e_1^T (K - Om^2 I + 2i Om diag(zeta))^{-1} e_1

differs from it.  A referee is entitled to ask whether the ensemble's two
headline statements survive that, so this script recomputes everything from
G_exact:

  * the cusp locus, birth events, and the P-1 coexisting-pair bound;
  * the background-sign type rule, with an EXACT pole/background split.

The exact split comes from the quadratic eigenvalue problem.  Linearizing
(s^2 I + s C + K) phi = 0 with C = 2 diag(zeta) gives 2N poles s_k in conjugate
pairs and, since M, C, K are symmetric,

    G(s) = sum_k  phi_k1^2 / (a_k (s - s_k)),
    a_k  = 2 s_k phi_k^T phi_k + phi_k^T C phi_k,

which reproduces G_exact to machine precision.  The active pole at a born cusp
of frequency Om is the conjugate PAIR whose |Im s_k| is closest to Om, and the
background G_a is G_exact(i Om) with that pair's two terms removed.  That is
the auditor's "best fix": an exactly pole-removed background, no proportional-
damping assumption anywhere.

Draws, coupling grid, association tolerance and visibility rule are identical
to `ensemble_validity_boundary.py`, so the two runs are comparable line by line.

Result (2026-09-08), quoted in the long paper's ensemble appendix:

                              modal (published)   exact damping
    birth events / cusps            236 / 474        233 / 470
    networks above P-1 rungs            0/100            0/100
    full (P-1) ladders                 23/100           23/100
    type rule                     462/474 97.5%   457/470 97.2%
    all cusps with psi < 0.2          417 ok           404 ok

    |G_modal - G_exact|/|G_exact| at the born cusps:
        median 2.5%,  90th pct 7.2%,  max 19%

So the modal sum is NOT a numerically negligible stand-in for the exact
receptance -- but every conclusion the ensemble is used for survives it.

Run:  .venv/bin/python scripts/ensemble_exact_damping.py
Cost: ~12 min on one core.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ensemble_unified import DEDUP, NK, random_network, visible

SEED = 20260811
N_PER_N = 25
NS = (3, 4, 5, 6)
NSCAN = 1500
PSI_EDGES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, np.inf)


def G_exact(Om, K, ze):
    """Driving-point receptance at coordinate 0; Om may be an array."""
    Om = np.atleast_1d(np.asarray(Om, dtype=float))
    N = K.shape[0]
    Z = (K[None, :, :].astype(complex)
         - Om[:, None, None] ** 2 * np.eye(N)
         + 2j * Om[:, None, None] * np.diag(ze))
    return np.linalg.inv(Z)[:, 0, 0]


def f_exact(Om, K, ze):
    inv = 1.0 / G_exact(Om, K, ze)
    return inv.real ** 2 - 3 * inv.imag ** 2


def qep(K, ze):
    """Poles and residues of the exact receptance, from the linearized QEP."""
    N = K.shape[0]
    C = 2 * np.diag(ze)
    A = np.block([[np.zeros((N, N)), np.eye(N)], [-K, -C]])
    lam, W = np.linalg.eig(A)
    phi = W[:N, :]
    a = np.array([2 * lam[k] * (phi[:, k] @ phi[:, k]) + phi[:, k] @ C @ phi[:, k]
                  for k in range(2 * N)])
    r = np.array([phi[0, k] ** 2 / a[k] for k in range(2 * N)])
    return lam, r


def cusp_freqs(K, ze, lo, hi):
    oms = np.linspace(lo, hi, NSCAN)
    fv = f_exact(oms, K, ze)
    out = []
    for i in range(NSCAN - 1):
        if fv[i] * fv[i + 1] < 0:
            o = brentq(lambda x: float(f_exact(x, K, ze)[0]),
                       oms[i], oms[i + 1], xtol=1e-9)
            if (1.0 / G_exact(o, K, ze)[0]).real < 0:
                out.append(o)
    return out


def background(Om, K, ze, lam, r):
    """G_a at Om: exact receptance minus the active conjugate pole pair."""
    s = 1j * Om
    # pair the 2N poles by |Im s|; the active pair is the one nearest Om
    im = np.abs(lam.imag)
    order = np.argsort(im)
    pairs = [order[i:i + 2] for i in range(0, len(order), 2)]
    freqs = [im[p].mean() for p in pairs]
    act = pairs[int(np.argmin(np.abs(np.array(freqs) - Om)))]
    keep = np.ones(len(lam), bool)
    keep[act] = False
    return np.sum(r[keep] / (s - lam[keep]))


def born(N, w, ze, K0, L):
    lo, hi = 0.5, w.max() + 0.6
    ks = np.linspace(0.0, 1.5, NK)
    recs, events = [], []
    prev = cusp_freqs(K0, ze, lo, hi)
    max_count, nonmono = len(prev), False
    for k in ks[1:]:
        K = K0 + k * L
        cur = cusp_freqs(K, ze, lo, hi)
        max_count = max(max_count, len(cur))
        if len(cur) < len(prev):
            nonmono = True
        if len(cur) > len(prev):
            size = len(cur) - len(prev)
            dist = sorted(((min([abs(o - q) for q in prev], default=9.0), o)
                           for o in cur), reverse=True)
            new = [o for d, o in dist[:size] if d > DEDUP]
            events.append(size)
            lam, r = qep(K, ze)
            wm = np.sqrt(np.linalg.eigvalsh(K))
            V = np.linalg.eigh(K)[1]
            zm = np.array([V[:, m] @ np.diag(ze) @ V[:, m] for m in range(N)])
            res = V[0, :] ** 2
            for Om in new:
                h = 1e-4
                aO = 0.5 * float(f_exact(Om + h, K, ze)[0]
                                 - 2 * f_exact(Om, K, ze)[0]
                                 + f_exact(Om - h, K, ze)[0]) / h ** 2
                Ga = background(Om, K, ze, lam, r)
                Gm = np.sum(res / (wm ** 2 - Om ** 2 + 2j * zm * Om))
                Ge = G_exact(Om, K, ze)[0]
                recs.append(dict(N=N, kappa=float(k), Om=float(Om), aO=aO,
                                 reGa=float(Ga.real), imGa=float(Ga.imag),
                                 psi=abs(Ga.imag) / abs(Ga.real),
                                 dG=abs(Gm - Ge) / abs(Ge),
                                 ok=bool(np.sign(aO) == -np.sign(Ga.real))))
        prev = cur
    return recs, events, (max_count - 1) // 2, nonmono


def main():
    rng = np.random.default_rng(SEED)
    recs, events = [], []
    nets = over = full = dark = nonmono = 0
    for N in NS:
        for _ in range(N_PER_N):
            w, ze, K0, L = random_network(rng, N)
            P = visible(K0, L, np.linspace(0.0, 1.5, NK)[1:])
            r, ev, rungs, nm = born(N, w, ze, K0, L)
            recs += r
            events += ev
            nets += 1
            dark += P < N
            over += rungs > P - 1
            full += rungs == P - 1
            nonmono += nm
        print(f"  N={N} done", flush=True)

    psi = np.array([x["psi"] for x in recs])
    ok = np.array([x["ok"] for x in recs])
    dG = np.array([x["dG"] for x in recs])
    print("\n" + "=" * 66)
    print(f"EXACT-DAMPING ENSEMBLE: {nets} networks, N = 3..6, seed {SEED}")
    print("=" * 66)
    print(f"  birth events: {len(events)}   born cusps: {len(recs)}")
    print(f"  networks with a dark mode:         {dark}/{nets}")
    print(f"  networks exceeding P-1 rungs:      {over}/{nets}")
    print(f"  full (P-1)-rung ladder realized:   {full}/{nets}")
    print(f"  cusp count non-monotonic in kappa: {nonmono}/{nets}")
    print(f"  type rule holds:                   {ok.sum()}/{len(ok)} "
          f"({100 * ok.mean():.1f}%)")
    print(f"\n  modal-vs-exact receptance error at the born cusps:")
    print(f"    median {np.median(dG):.2e}   90th pct {np.quantile(dG, .9):.2e}"
          f"   max {dG.max():.2e}")
    print("\n  ACCURACY vs BACKGROUND PHASE (exact pole-removed G_a)")
    for a, b in zip(PSI_EDGES[:-1], PSI_EDGES[1:]):
        m = (psi >= a) & (psi < b)
        if m.any():
            lab = f"[{a:.2f}, {b:.2f})" if np.isfinite(b) else f"[{a:.2f}, inf)"
            print(f"  {lab:>16} {m.sum():5d} {ok[m].sum():5d} "
                  f"{100 * ok[m].mean():8.1f}%")


if __name__ == "__main__":
    main()
