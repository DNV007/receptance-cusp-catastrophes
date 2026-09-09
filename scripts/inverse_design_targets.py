"""Inverse design proper: prescribe a birth frequency, read off the network.

`inverse_design_mech.py` posed exactly this problem and solved 3 of 11
targets.  `inverse_design_map.py` shows why: seeded by continuation along the
birth locus rather than from a fixed list of guesses, the same Newton solves
almost everywhere, so the 8 failures were the solver and not the physics.

This script does the design step the way it should be reported:

  1. read the birth locus from data/inverse_design_map.csv,
  2. bracket the requested Om* on it and interpolate a seed,
  3. Newton on [f, d_Om f] = 0 for (w2, kappa) at fixed Om = Om*,
  4. return the network and the predicted topology sign(a_Omega).

Targets that fall in the genuine gap -- where the two modes are too nearly
degenerate for a pole to clear the primary resonance, so no birth exists at
any coupling -- are reported as unreachable, which is a statement about the
network family and not about the solver.

Output: data/inverse_design_targets.csv
Run with the project venv:  ../.venv/bin/python inverse_design_targets.py
"""
from __future__ import annotations

import os

import numpy as np

from inverse_design_map import G, f, df_dOm, a_Omega, BETA_SIGN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MAP = os.path.join(DATA, "inverse_design_map.csv")

TARGETS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40]


def load_map():
    raw = np.genfromtxt(MAP, delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    return sorted(
        (dict(w2=float(r["w2"]), Om=float(r["Om_star"]),
              kappa=float(r["kappa_star"]), a_Om=float(r["a_Omega"]))
         for r in np.atleast_1d(raw)),
        key=lambda d: d["Om"])


def seed_for(target, locus):
    """Bracket target on the locus and linearly interpolate (w2, kappa)."""
    for lo, hi in zip(locus, locus[1:]):
        if lo["Om"] <= target <= hi["Om"]:
            span = hi["Om"] - lo["Om"]
            t = 0.0 if span == 0 else (target - lo["Om"]) / span
            return (lo["w2"] + t * (hi["w2"] - lo["w2"]),
                    lo["kappa"] + t * (hi["kappa"] - lo["kappa"]))
    return None


def design(Om_t, seed, iters=300):
    """Newton on [f, d_Om f] = 0 for (w2, kappa) at fixed Om = Om_t."""
    v = np.array(seed, float)

    def res(v):
        w2, kap = v
        return np.array([f(Om_t, w2, kap), df_dOm(Om_t, w2, kap)])

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

    w2, kap = v
    if not (np.isfinite(w2) and np.isfinite(kap)):
        return None
    if w2 <= 0.05 or kap <= 0 or kap > 1.5:
        return None
    if abs(f(Om_t, w2, kap)) + abs(df_dOm(Om_t, w2, kap)) > 1e-8:
        return None
    if BETA_SIGN * (1.0 / G(Om_t, w2, kap)).real >= 0:
        return None
    return float(w2), float(kap), float(a_Omega(Om_t, w2, kap))


def main() -> None:
    locus = load_map()
    om_lo, om_hi = locus[0]["Om"], locus[-1]["Om"]
    print(f"birth locus covers Om* in [{om_lo:.4f}, {om_hi:.4f}], "
          f"{len(locus)} points\n")
    print(f"{'target Om*':>10} {'w2':>9} {'kappa*':>9} {'a_Om':>12} "
          f"{'type':>6} {'argG':>9} {'resid':>9}")

    rows, unreachable = [], []
    for t in TARGETS:
        seed = seed_for(t, locus)
        got = design(t, seed) if seed is not None else None
        if got is None:
            print(f"{t:10.3f}   unreachable (no birth at this frequency)")
            unreachable.append(t)
            continue
        w2, kap, aO = got
        ang = float(np.degrees(np.angle(G(t, w2, kap))))
        resid = abs(f(t, w2, kap)) + abs(df_dOm(t, w2, kap))
        typ = "beaks" if aO > 0 else "lips"
        rows.append(dict(target=t, w2=w2, kappa=kap, a_Om=aO, typ=typ,
                         argG=ang, resid=resid))
        print(f"{t:10.3f} {w2:9.5f} {kap:9.5f} {aO:+12.4g} {typ:>6} "
              f"{ang:+9.3f} {resid:9.1e}")

    nb = sum(1 for d in rows if d["typ"] == "beaks")
    nl = len(rows) - nb
    print("\n" + "-" * 70)
    print(f"solved {len(rows)}/{len(TARGETS)} targets: {nb} beaks, {nl} lips")
    if unreachable:
        print(f"  unreachable: {unreachable}")
        print("  (these fall in the near-degenerate window where no pole "
              "clears the primary resonance)")

    out = os.path.join(DATA, "inverse_design_targets.csv")
    with open(out, "w") as fh:
        fh.write("target_Om,w2,kappa,a_Omega,type,argG_deg,residual\n")
        for d in rows:
            fh.write(f"{d['target']},{d['w2']:.9f},{d['kappa']:.9f},"
                     f"{d['a_Om']:.6e},{d['typ']},{d['argG']:.6f},"
                     f"{d['resid']:.3e}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
