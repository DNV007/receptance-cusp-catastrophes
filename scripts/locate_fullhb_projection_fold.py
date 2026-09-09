"""Locate the codimension-two point directly in the full harmonic-balance system.

Section IV.C obtains kappa*_HB by assuming the square-root form
Delta Omega = c0 sqrt(kappa - kappa*) and solving it from the near-threshold
tips.  That is an extrapolation under an assumed normal form, so it does not by
itself establish that the full multi-harmonic problem carries the same
codimension-two germ; it establishes agreement with a form taken from the
reduced model.

This script measures the same point without assuming any exponent.  A beaks
birth is defined by the opening of a GAP: below kappa* the bistable band near
the active pole is connected, above it an interval of Omega carries no
bistability at all, flanked on both sides by bistable frequencies.  The
projection fold is therefore the coupling at which that gap first exists, and
it can be bracketed by bisection on a purely topological test -- "is there an
interior run of frequencies with fewer than two folds?" -- evaluated on the
full harmonic-balance residual.

Agreement between this directly located fold and the extrapolated
kappa*_HB is evidence that the full system carries the codimension-two point
itself, not merely a look-alike unfolding.

Output: data/fullhb_projection_fold.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, F_folds, K_2dof  # noqa: E402
from recompute_beaks_table import closed_form_cusps, Fc_of, KAP_STAR  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
Z1, Z2, BETA1 = 0.015, 0.02, 0.147

# The published bracket was obtained at NH = 3.  The gap-opening test is a
# topological verdict, and check_bisection_nharm.py re-evaluates it at both
# bracket ends for NH = 3, 5, 7, 9 and gets the same answer every time, so the
# bracket is truncation independent over that range.  NH is overridable so the
# archived run can be repeated at the truncation the manuscript quotes:
#     NH=9 python locate_fullhb_projection_fold.py
NH = int(os.environ.get("NH", 3))
NT = int(os.environ.get("NT", 256))


def gap_width(S, kappa, n_om=90, pad=0.35, verbose=False):
    """Width of the non-bistable interval separating the two sub-wedges.

    Returns 0.0 when the band is connected (no interior gap).  The scan window
    is set from the closed-form pair and padded, so it always spans the region
    where a gap could appear.
    """
    cf = [c for c in closed_form_cusps(kappa) if c > 1.2]
    if len(cf) >= 2:
        lo_cf, hi_cf = cf[0], cf[1]
        w = max(hi_cf - lo_cf, 4e-3)
        lo, hi = lo_cf - pad * w, hi_cf + pad * w
        F_max = 3.0 * max(Fc_of(lo_cf, kappa), Fc_of(hi_cf, kappa))
    else:
        # Below the REDUCED-model threshold the closed form has no upper pair,
        # but the full system may still have one -- that asymmetry is the whole
        # point of the measurement -- so fall back to a fixed window centred on
        # the reduced-model birth frequency rather than declaring failure.
        centre = 1.3010
        lo, hi = centre - 0.016, centre + 0.016
        F_max = 3.0 * max(Fc_of(centre, kappa) or 1.0, 1.0)
    oms = np.linspace(lo, hi, n_om)
    bist = []
    for om in oms:
        f = F_folds(S, K_2dof(kappa), float(om), ds=0.01, F_max=F_max)
        bist.append(len(f) >= 2)
    bist = np.array(bist)
    if not bist.any():
        return float("nan"), None
    idx = np.where(bist)[0]
    # interior runs of non-bistable frequencies, i.e. bounded by bistability
    inner = bist[idx[0]:idx[-1] + 1]
    if inner.all():
        return 0.0, (oms, bist)          # connected band: no gap yet
    runs, cur = [], []
    for j, b in enumerate(inner):
        if not b:
            cur.append(j)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return 0.0, (oms, bist)
    big = max(runs, key=len)
    step = oms[1] - oms[0]
    return (len(big) + 1) * step, (oms, bist)


def main():
    S = make_solver(2, NH, NT, [Z1, Z2], BETA1)
    # bracket: gap present at the upper end, absent at the lower end
    k_lo, k_hi = 0.1170, 0.1225
    print(f"bracketing the projection fold in [{k_lo}, {k_hi}]", flush=True)
    for tag, k in (("lo", k_lo), ("hi", k_hi)):
        t0 = time.time()
        g, _ = gap_width(S, k)
        print(f"  kappa={k:.5f} ({tag}): gap = {g:.6f}  [{time.time()-t0:.0f}s]",
              flush=True)

    hist = []
    for it in range(9):
        k_mid = 0.5 * (k_lo + k_hi)
        t0 = time.time()
        g, _ = gap_width(S, k_mid)
        has_gap = np.isfinite(g) and g > 0
        hist.append(dict(kappa=k_mid, gap=float(g), has_gap=bool(has_gap)))
        print(f"  it{it}: kappa={k_mid:.6f}  gap={g:.6f}  "
              f"{'GAP' if has_gap else 'connected'}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if has_gap:
            k_hi = k_mid
        else:
            k_lo = k_mid
        if k_hi - k_lo < 2e-5:
            break

    kstar_direct = 0.5 * (k_lo + k_hi)
    out = dict(kstar_direct=kstar_direct,
               bracket=[k_lo, k_hi],
               bracket_width=k_hi - k_lo,
               kstar_extrapolated=0.119072,
               kstar_reduced_model=KAP_STAR,
               history=hist)
    print(f"\ndirectly located projection fold: kappa* = {kstar_direct:.6f} "
          f"(bracket {k_hi - k_lo:.1e})")
    print(f"  extrapolated from the sqrt law: 0.119072  "
          f"(difference {abs(kstar_direct - 0.119072):.6f})")
    print(f"  reduced-model prediction:       {KAP_STAR:.6f}  "
          f"(offset {100*(kstar_direct-KAP_STAR)/KAP_STAR:+.2f}%)")
    p = os.path.join(DATA, "fullhb_projection_fold.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
