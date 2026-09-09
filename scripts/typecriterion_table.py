"""Regenerate Table `tab:typecriterion` of the long paper from the exact receptance.

The table sweeps the absorber pole omega_2 through the driven resonance and
reports, at the codimension-two birth, the coupling kappa* and the curvature

    a_Omega = (1/2) d^2 f / dOmega^2,      f = rho^2 - 3 sigma^2,

whose sign is the beaks/lips classifier. The caption claims these are computed
from the exact driving-point receptance e_1^T Z^-1 e_1, so they must not be
modal (proportional-damping) values: C = diag(2 z1, 2 z2) with z1 != z2 is NOT
proportional damping, and the modal sum is only an approximation to G. The
error is ~0.5% in kappa* at omega_2 = 1.25 but ~6% at omega_2 = 0.95, which is
how a stale modal entry (kappa* = 0.283 against the exact 0.2998) survived in
that row.

a_Omega is a second difference, so its step dependence is checked here rather
than assumed: it is stable over four decades in h.
"""
from __future__ import annotations

from collar_width import _d, birth_robust, f_lead

W2S = (0.60, 0.80, 0.90, 0.95, 1.10, 1.25, 1.40)
HS = (1e-4, 1e-5, 1e-6, 1e-7)


def main():
    print("tab:typecriterion, exact receptance")
    print(f"{'omega_2':>8} {'kappa*':>9} {'Omega*':>9} {'a_Omega':>10} "
          f"{'type':>6}   a_Omega over h = " + ", ".join(str(h) for h in HS))
    for w2 in W2S:
        got = birth_robust(f_lead, w2)
        if got is None:
            print(f"{w2:8.2f}   birth_robust found no birth")
            continue
        Om, kap = got[0], got[1]
        vals = [0.5 * _d(f_lead, Om, kap, w2, h, 2) for h in HS]
        a = vals[1]
        print(f"{w2:8.2f} {kap:9.4f} {Om:9.4f} {a:10.4g} "
              f"{'beaks' if a > 0 else 'lips':>6}   "
              + ", ".join(f"{v:.5g}" for v in vals))
    print("\nRow as typeset: omega_2, kappa* (4 dp), a_Omega (3 s.f.), type.")


if __name__ == "__main__":
    main()
