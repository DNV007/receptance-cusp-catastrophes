"""Unified random-network ensemble: birth statistics AND type criterion, one pass.

Supersedes ensemble_stress_test2.py + ensemble_criterion_check.py, which used
the same 24-network ensemble but reported counts that could not be reconciled:
stress_test2 counted birth EVENTS (43, all of size +2) while criterion_check
counted individual born CUSPS (91) on a coarser coupling grid, and the
manuscript described both as "births".

Everything here is measured on one coupling grid so the counts are consistent
by construction:
  - n_events        cusp-pair birth events, with the size distribution
  - n_cusps         individual cusps born (= 2 * n_events for clean pairs)
  - criterion       sign(a_Omega) = -sign(Re G_a), evaluated at each born cusp
  - ladder          networks realizing the full (P-1)-rung ladder

A RUNG IS A COEXISTING CUSP PAIR, so the ladder height of a network is
(max_kappa cusp_count - 1) / 2, NOT the number of times the count stepped up.
The two differ whenever the cusp count is non-monotonic in kappa -- pairs
annihilate and re-form as the coupling sweeps -- which happens in 13 of these
24 networks. Counting increase events overstates the ladder height (13/24
against the correct 9/24) and can even manufacture apparent violations of the
P-1 bound in larger networks. Event counts remain the right instrument for the
birth-event statistics (how many births, and whether each was a clean pair).

P is the number of modes VISIBLE at the nonlinear coordinate, i.e. with nonzero
participation res[m] = V[0, m]**2 there; a dark mode cancels from G and cannot
organize a birth, so the candidate-rung count is P-1 <= N-1.  With pairwise
coupling probability 0.7 the ensemble really does contain dark modes, so the
two bounds differ here and the (P-1) one is the statement to quote.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

SEED = 20260718
BETA1 = 0.147
N_PER_N = 12
NS = (3, 4)
NK = 300                      # coupling grid, [0, 1.5]
DEDUP = 5e-3                  # frequency tolerance for "same cusp"


def random_network(rng, N):
    w = np.sort(rng.uniform(0.6, 2.4, N)); w[0] = 1.0
    ze = rng.uniform(0.01, 0.04, N)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.7:
                A[i, j] = A[j, i] = rng.uniform(0.3, 1.0)
    return w, ze, np.diag(w**2), np.diag(A.sum(1)) - A


def modal(K, ze, N):
    e, V = np.linalg.eigh(K)
    zm = np.array([V[:, m] @ np.diag(ze) @ V[:, m] for m in range(N)])
    return np.sqrt(e), zm, V[0, :] ** 2


def visible(K0, L, ks, tol=1e-10):
    """Modes with nonzero participation at coordinate 0, maximized over the
    coupling grid: a mode dark at kappa = 0 may light up as coupling is
    turned, so P is taken as the largest count seen on the grid."""
    return max(int(np.sum(np.linalg.eigh(K0 + k * L)[1][0, :] ** 2 > tol))
               for k in ks)


def G_of(Om, wm, zm, res):
    return np.sum(res / (wm**2 - Om**2 + 2j * zm * Om))


def f_of(Om, wm, zm, res):
    inv = 1.0 / G_of(Om, wm, zm, res)
    return inv.real**2 - 3 * inv.imag**2


def cusp_freqs(K, ze, N, lo, hi, n=1500):
    wm, zm, res = modal(K, ze, N)
    oms = np.linspace(lo, hi, n)
    fv = np.array([f_of(o, wm, zm, res) for o in oms])
    out = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            o = brentq(lambda x: f_of(x, wm, zm, res), oms[i], oms[i + 1],
                       xtol=1e-9)
            if (1.0 / G_of(o, wm, zm, res)).real < 0:
                out.append(o)
    return out, wm, zm, res


def main():
    rng = np.random.default_rng(SEED)
    n_nets = 0
    sizes: dict[int, int] = {}
    n_cusps = 0
    n_agree = 0
    n_complex = 0
    n_complex_agree = 0
    n_full_ladder = 0
    n_full_ladder_N = 0
    n_dark = 0
    n_over = 0
    n_nontrivial = 0
    n_full_ladder_nt = 0
    n_full_ladder_ev = 0
    failures: list[tuple[float, float, float]] = []
    all_reGa: list[float] = []

    for N in NS:
        for _ in range(N_PER_N):
            w, ze, K0, L = random_network(rng, N)
            n_nets += 1
            lo, hi = 0.5, w.max() + 0.6
            ks = np.linspace(0.0, 1.5, NK)
            P = visible(K0, L, ks[1:])
            n_dark += P < N
            prev, _, _, _ = cusp_freqs(K0, ze, N, lo, hi)
            max_count = len(prev)
            net_pairs = 0
            for k in ks[1:]:
                K = K0 + k * L
                cur, wm, zm, res = cusp_freqs(K, ze, N, lo, hi)
                max_count = max(max_count, len(cur))
                if len(cur) > len(prev):
                    size = len(cur) - len(prev)
                    # A fixed dedup tolerance mis-labels a *moved* cusp as new
                    # whenever it drifts more than DEDUP in one coupling step.
                    # That happened once in this ensemble (N=4, kappa~0.351),
                    # inflating the cusp count to 87 against 43 x 2 = 86.
                    # The count jump is the reliable quantity, so keep exactly
                    # `size` candidates: the ones furthest from any predecessor.
                    dist = sorted(
                        ((min([abs(o - q) for q in prev], default=9.0), o)
                         for o in cur), reverse=True)
                    new = [o for d, o in dist[:size] if d > DEDUP]
                    sizes[size] = sizes.get(size, 0) + 1
                    if size == 2:
                        net_pairs += 1
                    for Om in new:
                        h = 1e-4
                        aO = 0.5 * (f_of(Om + h, wm, zm, res)
                                    - 2 * f_of(Om, wm, zm, res)
                                    + f_of(Om - h, wm, zm, res)) / h**2
                        act = int(np.argmin(np.abs(Om - wm)))
                        Ga = np.sum([res[m] / (wm[m]**2 - Om**2
                                               + 2j * zm[m] * Om)
                                     for m in range(N) if m != act])
                        n_cusps += 1
                        ok = np.sign(aO) == -np.sign(Ga.real)
                        n_agree += bool(ok)
                        all_reGa.append(abs(Ga.real))
                        if not ok:
                            failures.append((abs(Ga.real), abs(Ga.imag), aO))
                        if abs(Ga.imag) > 0.1 * abs(Ga.real):
                            n_complex += 1
                            n_complex_agree += bool(ok)
                prev = cur
            rungs = (max_count - 1) // 2      # coexisting pairs, see header
            n_full_ladder += rungs == P - 1
            n_full_ladder_N += rungs == N - 1
            n_over += rungs > P - 1
            n_full_ladder_ev += net_pairs == P - 1
            # A P = 1 realization (driven coordinate uncoupled) satisfies the
            # (P-1) ladder trivially with zero rungs, so report it separately.
            n_nontrivial += P > 1
            n_full_ladder_nt += (P > 1) and (rungs == P - 1)

    n_events = sum(sizes.values())
    print(f"networks:            {n_nets}  (N = {', '.join(map(str, NS))})")
    print(f"birth events:        {n_events}")
    for s in sorted(sizes):
        print(f"   size +{s}:          {sizes[s]}")
    print(f"individual cusps born: {n_cusps}")
    print(f"networks with a dark mode (P < N): {n_dark}/{n_nets}")
    print(f"networks exceeding P-1 rungs: {n_over}/{n_nets}")
    print(f"full (P-1)-rung ladder: {n_full_ladder}/{n_nets}"
          f"   ({n_full_ladder_nt}/{n_nontrivial} among P > 1)"
          f"   [(N-1)-rung: {n_full_ladder_N}/{n_nets}]")
    print(f"   (counting increase events instead would say "
          f"{n_full_ladder_ev}/{n_nets} -- wrong, see header)")
    print(f"\ntype criterion sign(a_Om) = -sign(Re G_a):")
    print(f"   holds at            {n_agree}/{n_cusps} born cusps")
    print(f"   appreciably complex background: {n_complex}"
          f"  (holds at {n_complex_agree}/{n_complex})")
    if all_reGa:
        med = float(np.median(all_reGa))
        print(f"\n   median |Re G_a| over all born cusps: {med:.4f}")
        for r, i, a in failures:
            print(f"   FAILURE: |Re G_a| = {r:.5f} ({r/med:.3f} x median), "
                  f"|Im G_a| = {i:.5f}, a_Omega = {a:+.1f}")


if __name__ == "__main__":
    main()
