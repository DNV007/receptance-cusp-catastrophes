"""How closely the closed-form cusp locus tracks full harmonic balance.

The Discussion quotes a per-cusp agreement in the cusp Omega-coordinate across
the sixteen couplings of tab:beaks_hb. That comparison is only meaningful once
it is said WHICH receptance the closed form uses, because the modal
(proportional-damping) form and the exact e_1^T Z^-1 e_1 differ by more than
the residual being reported.

Both are computed here. The modal form reproduces the quoted figures --
max 1.29e-3, median 7.81e-4 -- and the exact receptance does substantially
better, max 9.46e-4 and median 2.24e-4. So a third of the "leading-order
projection error" attributed to the truncation is in fact the modal
approximation to G.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.02

# Full-HB upper-pair tips, tab:beaks_hb (N_HARM = 3, beta1 = 0.147)
HB = [(0.1200, 1.29902, 1.30391), (0.1210, 1.29838, 1.30540),
      (0.1225, 1.29785, 1.30727), (0.1250, 1.29743, 1.30993),
      (0.1275, 1.29732, 1.31231), (0.1300, 1.29737, 1.31454),
      (0.1400, 1.29842, 1.32265), (0.1500, 1.30016, 1.33014),
      (0.1600, 1.30227, 1.33734), (0.1700, 1.30461, 1.34437),
      (0.1800, 1.30712, 1.35128), (0.2000, 1.31248, 1.36488),
      (0.2200, 1.31826, 1.37831), (0.2500, 1.32728, 1.39825),
      (0.2800, 1.33666, 1.41802), (0.3000, 1.34307, 1.43113)]


def _K(kap):
    return np.array([[W1**2 + kap, -kap], [-kap, W2**2 + kap]])


def G_exact(Om, kap):
    Z = _K(kap) - Om**2 * np.eye(2) + 1j * Om * np.diag([2 * Z1, 2 * Z2])
    return np.linalg.inv(Z)[0, 0]


def G_modal(Om, kap):
    e, V = np.linalg.eigh(_K(kap))
    wm = np.sqrt(e)
    zm = [V[:, m] @ np.diag([Z1, Z2]) @ V[:, m] for m in range(2)]
    return sum(V[0, m]**2 / (wm[m]**2 - Om**2 + 2j * zm[m] * Om)
               for m in range(2))


def pair(Gf, kap, lo=1.24, hi=1.52, n=60000):
    """The upper cusp pair: roots of rho^2 = 3 sigma^2 on the branch rho < 0."""
    def f(o):
        inv = 1 / Gf(o, kap)
        return inv.real**2 - 3 * inv.imag**2
    o = np.linspace(lo, hi, n)
    fv = np.array([f(x) for x in o])
    out = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0:
            r = brentq(f, o[i], o[i + 1], xtol=1e-13)
            if (1 / Gf(r, kap)).real < 0:
                out.append(r)
    return sorted(out)


def main():
    for name, Gf in (("modal (eq. closed_form)", G_modal),
                     ("exact  e_1^T Z^-1 e_1", G_exact)):
        d = []
        for k, om, op in HB:
            cs = pair(Gf, k)
            if len(cs) < 2:
                print(f"  kappa={k}: only {len(cs)} cusp(s) found")
                continue
            d += [abs(cs[0] - om), abs(cs[-1] - op)]
        d = np.array(d)
        print(f"{name}: n={len(d)}  max={d.max():.3e}  "
              f"median={np.median(d):.3e}  mean={d.mean():.3e}")
    print("\nmanuscript (Discussion): 'better than 1.3e-3 per cusp, "
          "median 7.8e-4' -- the modal row.")


if __name__ == "__main__":
    main()
