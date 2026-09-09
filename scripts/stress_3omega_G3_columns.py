"""Complex G(3*Omega_c) for every member of the 3-Omega stress-test family.

SM Table I reports eta3 = (sqrt3/9)|G(3Om)/G(Om)|, a MODULUS.  The fold cubic
Eq. (S8) does not use the modulus: it uses g_r = Re G3 and g_i = Im G3
separately, through the term -6q(g_r rho + g_i sigma).  So eta3 fixes the scale
of the correction while the complex G3 fixes its sign and the ordering of the
residuals across the family.  Two members can share eta3 and land on opposite
sides; two can share sign(Re G3) and still differ, once |Im G3| grows.

This script emits those two columns so the table can show them rather than
assert a rule.  G3 is evaluated at the FUNDAMENTAL prediction Omega_1 of each
row (the reference the residuals are measured against); using Omega_HB instead
moves 3*Omega by ~3e-3, a fifth of the probe half-width, and changes no sign.

Reads   data/stress_3omega_resonance.json   (produced by stress_3omega_resonance.py)
Writes  data/stress_3omega_G3_columns.csv
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# identical to stress_3omega_resonance.py
BASE_MODES = [(1.00, 0.70, 0.015), (1.30, 0.30, 0.020)]
P3, Z3 = 0.05, 0.005


def G(Om, modes):
    return sum(p / (w ** 2 - Om ** 2 + 2j * z * w * Om) for w, p, z in modes)


def main():
    with open(os.path.join(DATA, "stress_3omega_resonance.json")) as fh:
        d = json.load(fh)
    rows = sorted(d["rows"], key=lambda r: r["eta3"])
    out = []
    print(f"{'eta3':>7} {'w3':>7} {'Re G3':>9} {'Im G3':>9} {'|G3|':>8} "
          f"{'err_fund':>10}")
    for r in rows:
        g3 = G(3.0 * r["Om_fund"], BASE_MODES + [(r["w3"], P3, Z3)])
        out.append(dict(eta3=r["eta3"], w3=r["w3"], ReG3=g3.real, ImG3=g3.imag,
                        absG3=abs(g3), err_fund=r["err_fund"]))
        print(f"{r['eta3']:7.4f} {r['w3']:7.3f} {g3.real:+9.4f} {g3.imag:+9.4f} "
              f"{abs(g3):8.4f} {r['err_fund']:+10.2e}")

    path = os.path.join(DATA, "stress_3omega_G3_columns.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {path}")

    pos = [r for r in out if r["ReG3"] > 0]
    print(f"rows with Re G3 > 0: {len(pos)} "
          f"({', '.join('w3=%.3f' % r['w3'] for r in pos)})")
    print("NOTE: sign(Re G3) alone does not order the residuals -- "
          "w3=3.500 and 3.545 share it and differ in residual sign, "
          "because |Im G3| grows from 0.03 to 0.17 between them.")


if __name__ == "__main__":
    main()
