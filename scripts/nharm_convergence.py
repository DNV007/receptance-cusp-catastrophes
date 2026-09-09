"""N_HARM convergence spot-check for the multimode bistability maps.

The 2-DOF results use N_HARM=3 (validated in Appendix, tab:nharm). The multimode
figure fig:maps used N_HARM=5 (hub beaks) and N_HARM=7 (lips, eps~0.7). A referee
will ask whether 5/7 is converged at the larger anharmonicity. Here we recompute
the saddle-node fold forcings F_SN(Omega) inside each born window at the reported
N_HARM and at higher truncations, and report the shift.

Reuses the exact machinery of make_fig_maps_data.py (ndof_hb + arclength_fold).
"""
import numpy as np
from ndof_hb import build_K
from arclength_fold import make_continuation

BETA1 = 0.147
C3 = 0.75 * BETA1


def K_hub(om, k):
    N = len(om)
    K = np.diag(np.asarray(om, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += k; K[j, j] += k
        K[0, j] -= k; K[j, 0] -= k
    return K


def start_u(N, nh, om, Om, F0):
    u = np.zeros(N * nh * 2 + 1)
    u[0] = F0 / max(1e-6, abs(om[0] ** 2 - Om ** 2))
    u[-1] = F0
    return u


def fold_forcings(om, ze, Om, k, Kf, nh, Fmax, nt=384):
    cont = make_continuation(om, ze, BETA1, Om, nh=nh, nt=nt)
    N = len(om)
    traj = cont["continue_F"](Kf(om, k), start_u(N, nh, om, Om, 0.008),
                              ds=0.008, n_steps=700, F_max=Fmax)
    if traj is None or len(traj) < 3:
        return []
    F, tau = traj[:, 0], traj[:, 2]
    out = []
    for i in range(len(tau) - 1):
        if tau[i] * tau[i + 1] < 0:
            out.append(0.5 * (F[i] + F[i + 1]))
    return sorted(out)


CONFIGS = [
    # label, om, ze, Kbuilder, kappa, Omega samples, nh list, Fmax
    ("hub beaks (reported nh=5)", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02],
     K_hub, 0.14, [1.13, 1.15, 1.17], [5, 7, 9], 1.3),
    ("lips  (reported nh=7)", [1.0, 0.60], [0.015, 0.02],
     build_K, 0.65, [0.83, 0.85, 0.87], [7, 9, 11], 1.3),
]

for lab, om, ze, Kf, k, oms, nhs, Fmax in CONFIGS:
    print("=" * 70)
    print(f"{lab}   kappa={k}")
    print("=" * 70)
    for Om in oms:
        rows = {}
        for nh in nhs:
            rows[nh] = fold_forcings(om, ze, Om, k, Kf, nh, Fmax)
        base = rows[nhs[0]]
        top = rows[nhs[-1]]
        # match by count; only compare when both give the same number of folds
        msg = ""
        if base and top and len(base) == len(top):
            rel = max(abs(b - t) / max(1e-9, abs(t)) for b, t in zip(base, top))
            msg = f"  max |dF_SN|/F_SN ({nhs[0]}->{nhs[-1]}) = {100*rel:.3f}%"
        print(f"  Om={Om:.3f}:")
        for nh in nhs:
            fl = ", ".join(f"{x:.5f}" for x in rows[nh]) or "(no folds)"
            print(f"     nh={nh:2d}:  F_SN = [{fl}]")
        if msg:
            print(msg)
    print()
