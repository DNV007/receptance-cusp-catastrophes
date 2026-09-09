"""Reachable set of cusp-pair births for the 2-DOF network: the design chart.

`inverse_design_mech.py` prescribes a target birth frequency and Newton-solves
for (w2, kappa).  It succeeded on 3 of 11 targets, which leaves the question
its output cannot answer: are the other 8 unreachable, or did the solver just
start from bad guesses?

This settles it by mapping the birth locus instead of sampling it.  A birth is

    f(Om, w2, kap) = 0,   d_Om f(Om, w2, kap) = 0,      f = rho^2 - 3 sigma^2

which for fixed w2 is two equations in (Om, kap).  Sweeping w2 by continuation
-- each solve seeded from the previous one -- traces the whole locus, so the
reachable birth frequencies are read off directly and any gap is a property of
the network family rather than of the solver.

Output: data/inverse_design_map.csv, one row per w2 with the birth (Om*,
kappa*) and the classifier a_Omega.

Run with the project venv:  ../.venv/bin/python inverse_design_map.py
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

W1, Z1, Z2 = 1.0, 0.015, 0.02
BETA_SIGN = +1.0                  # hardening: physical branch has rho < 0


def G(Om: float, w2: float, kap: float) -> complex:
    K = np.array([[W1**2 + kap, -kap], [-kap, w2**2 + kap]])
    Z = K - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def f(Om: float, w2: float, kap: float) -> float:
    inv = 1.0 / G(Om, w2, kap)
    return inv.real**2 - 3.0 * inv.imag**2


def df_dOm(Om, w2, kap, h=1e-6):
    return (f(Om + h, w2, kap) - f(Om - h, w2, kap)) / (2 * h)


def a_Omega(Om, w2, kap, h=1e-4):
    """(1/2) d2f/dOm2 -- the beaks/lips classifier."""
    return 0.5 * (f(Om + h, w2, kap) - 2 * f(Om, w2, kap)
                  + f(Om - h, w2, kap)) / h**2


def solve_birth(w2, seed, iters=300):
    """Newton on [f, d_Om f] = 0 for (Om, kappa) at fixed w2.

    Undamped, matching make_fig_prl_design.py: a line search on this system
    stalls and bails out on perfectly good starting points, which is what made
    an earlier version of this sweep report spurious gaps in the reachable set.
    """
    v = np.array(seed, float)

    def res(v):
        Om, kap = v
        return np.array([f(Om, w2, kap), df_dOm(Om, w2, kap)])

    for _ in range(iters):
        r = res(v)
        if np.linalg.norm(r) < 1e-14:
            break
        J = np.zeros((2, 2))
        for j, hh in enumerate((1e-7, 1e-7)):
            vp = v.copy(); vp[j] += hh
            vm = v.copy(); vm[j] -= hh
            J[:, j] = (res(vp) - res(vm)) / (2 * hh)
        try:
            v = v + np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None

    Om, kap = v
    if not (np.isfinite(Om) and np.isfinite(kap)) or kap <= 0 or kap > 1.5:
        return None
    # Om > 0 is NOT optional: f has roots at negative drive frequency, and
    # without this guard a seed grid wide enough to find every branch also
    # collects them. An earlier branch map reported Om* down to -1.53.
    if Om < 0.2:
        return None
    if abs(f(Om, w2, kap)) + abs(df_dOm(Om, w2, kap)) > 1e-8:
        return None
    # physical branch: beta1 * rho < 0
    rho = (1.0 / G(Om, w2, kap)).real
    if BETA_SIGN * rho >= 0:
        return None
    return float(Om), float(kap), float(a_Omega(Om, w2, kap))


def sweep(w2_grid, seed):
    """Continuation in w2, each solve seeded from the previous solution."""
    out, cur = [], seed
    for w2 in w2_grid:
        got = solve_birth(w2, cur)
        if got is None:                      # retry from a few fixed seeds
            for s in ((1.30, 0.12), (1.05, 0.30), (0.95, 0.40), (1.5, 0.2)):
                got = solve_birth(w2, s)
                if got is not None:
                    break
        if got is None:
            continue
        Om, kap, aO = got
        out.append(dict(w2=float(w2), Om=Om, kappa=kap, a_Om=aO,
                        typ="beaks" if aO > 0 else "lips"))
        cur = (Om, kap)
    return out


def main() -> None:
    # two passes, outward from the known baseline in each direction, so the
    # continuation never has to jump the degenerate region near w2 ~ w1
    up = sweep(np.arange(1.26, 2.401, 0.01), (1.30, 0.12))
    down = sweep(np.arange(1.25, 0.599, -0.01), (1.30, 0.12))
    rows = sorted(down + up, key=lambda d: d["w2"])

    if not rows:
        print("no births found")
        return

    print(f"{'w2':>7} {'Om*':>10} {'kappa*':>10} {'a_Om':>12} {'type':>7}")
    for d in rows[::10]:
        print(f"{d['w2']:7.3f} {d['Om']:10.5f} {d['kappa']:10.5f} "
              f"{d['a_Om']:+12.4g} {d['typ']:>7}")

    oms = np.array([d["Om"] for d in rows])
    beaks = [d for d in rows if d["typ"] == "beaks"]
    lips = [d for d in rows if d["typ"] == "lips"]
    print("\n" + "-" * 62)
    print(f"solved {len(rows)} of {len(np.arange(0.60, 2.401, 0.01))} w2 values")
    print(f"  reachable Om*: [{oms.min():.4f}, {oms.max():.4f}]")
    if beaks:
        ob = np.array([d["Om"] for d in beaks])
        wb = np.array([d["w2"] for d in beaks])
        print(f"  beaks: n={len(beaks)}  w2 in [{wb.min():.3f}, {wb.max():.3f}]"
              f"  Om* in [{ob.min():.4f}, {ob.max():.4f}]")
    if lips:
        ol = np.array([d["Om"] for d in lips])
        wl = np.array([d["w2"] for d in lips])
        print(f"  lips:  n={len(lips)}  w2 in [{wl.min():.3f}, {wl.max():.3f}]"
              f"  Om* in [{ol.min():.4f}, {ol.max():.4f}]")

    # gaps in the reachable Om* set
    so = np.sort(oms)
    gaps = [(so[i], so[i + 1]) for i in range(len(so) - 1)
            if so[i + 1] - so[i] > 0.02]
    if gaps:
        print("  gaps in reachable Om*:")
        for a, b in gaps:
            print(f"    ({a:.4f}, {b:.4f})   width {b - a:.4f}")
    else:
        print("  reachable Om* is a single interval, no gaps > 0.02")

    out = os.path.join(DATA, "inverse_design_map.csv")
    with open(out, "w") as fh:
        fh.write("w2,Om_star,kappa_star,a_Omega,type\n")
        for d in rows:
            fh.write(f"{d['w2']:.4f},{d['Om']:.9f},{d['kappa']:.9f},"
                     f"{d['a_Om']:.6e},{d['typ']}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
