"""Full-HB validation of a LIPS-type cusp-pair birth (contrast to the beaks
validation in hub_clean_validate.py). Absorber BELOW the drive (om2<1) gives
a_Om<0 (lips). LIPS signature in full HB: below kappa* NO bistable band near the
birth frequency Om*; above kappa* an ISOLATED bistable lens appears from nothing
(single connected band). Contrast beaks, where a connected band is present below
and a GAP opens above.

Config (om2, kappa list, Om band) is set from type_boundary_map.py.
"""
import numpy as np
from ndof_hb import build_K
from arclength_fold import make_continuation, count_folds

# ---- config set from the type-boundary map (clean lips: a_Om<0, accessible eps)
W = [1.0, 0.60]            # absorber below drive
Z = [0.015, 0.02]
BETA1 = 0.147
C3 = 0.75 * BETA1
KAPPAS = [0.45, 0.55, 0.65]   # HB threshold is eps-shifted above closed-form 0.446
OM_BAND = (0.78, 0.90)        # tight around the lips tip ~0.815
NH, NT = 7, 512
# -------------------------------------------------------------------------------


def rs(Om, k):
    Zm = build_K(W, k) - Om ** 2 * np.eye(2) + 1j * Om * np.diag([2 * z for z in Z])
    inv = 1.0 / np.linalg.solve(Zm, np.eye(2)[:, 0])[0]
    return inv.real, inv.imag


def cf_Fc(Om, k):
    r, s = rs(Om, k)
    if r >= 0:
        return None
    yc = -2 * r / (3 * C3)
    F2 = yc * ((r + C3 * yc) ** 2 + s ** 2)
    return np.sqrt(F2) if F2 > 0 else None


def cf_tips(k, lo, hi, n=3000):
    oms = np.linspace(lo, hi, n)
    fv = [rs(o, k)[0] ** 2 - 3 * rs(o, k)[1] ** 2 for o in oms]
    out = []
    for i in range(n - 1):
        if fv[i] * fv[i + 1] < 0 and rs(0.5 * (oms[i] + oms[i + 1]), k)[0] < 0:
            out.append(round(oms[i] - fv[i] * (oms[i + 1] - oms[i]) / (fv[i + 1] - fv[i]), 4))
    return out


def start_u(nh, Om, F0):
    u = np.zeros(2 * nh * 2 + 1)
    u[0] = F0 / max(1e-6, abs(W[0] ** 2 - Om ** 2))
    u[-1] = F0
    return u


oms = np.linspace(OM_BAND[0], OM_BAND[1], 13)
for k in KAPPAS:
    print(f"\nkappa={k:.3f}  closed-form cusp tips in band: {cf_tips(k, *OM_BAND)}")
    pat = []
    for Om in oms:
        Fc = cf_Fc(Om, k)
        Fmax = (1.6 * Fc) if Fc else 1.2
        c = make_continuation(W, Z, BETA1, Om, nh=NH, nt=NT)
        traj = c["continue_F"](build_K(W, k), start_u(NH, Om, 0.008),
                               ds=0.006, n_steps=900, F_max=Fmax)
        nf, _ = count_folds(traj)
        m = "##" if nf >= 2 else ".."
        pat.append(m)
        print(f"  Om={Om:.4f}: folds={nf} {m}")
    print(f"  pattern {oms[0]:.2f}->{oms[-1]:.2f}: {''.join(pat)}")
