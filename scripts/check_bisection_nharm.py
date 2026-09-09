"""Is the gap-opening bisection converged in the harmonic truncation?

data/fullhb_projection_fold.json brackets the codimension-two birth of the
two-mode absorber at kappa* = 0.11931, and that number is the Letter's headline
validation of the receptance prediction.  It was produced at N_HARM = 3.  The
Letter describes it as a nine-harmonic result, so the truncation dependence has
to be checked rather than assumed.

The bisection verdict at each coupling is purely topological -- does an
interior run of frequencies carry no bistability? -- so it is enough to
re-evaluate that verdict at the two ends of the published bracket for a range
of truncations.  If the verdict is unchanged, the bracket is truncation
independent and only the wording in the Letter needs fixing.

Output: data/bisection_nharm_check.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_tip_intercept import make_solver, F_folds, K_2dof  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
Z1, Z2, BETA1 = 0.015, 0.02, 0.147

BRACKET = (0.1193095703125, 0.1193203125)   # published: no gap / gap


def main():
    """Re-evaluate the published gap verdict at both bracket ends vs N_HARM.

    gap_width() is imported from locate_fullhb_projection_fold, so this is the
    published estimator with its own frequency window and padding; only the
    harmonic truncation changes.
    """
    from locate_fullhb_projection_fold import gap_width

    out = []
    for nh in (3, 5, 7, 9):
        S = make_solver(2, nh, 256, [Z1, Z2], BETA1)
        row = {"nh": nh}
        for tag, k in (("lo", BRACKET[0]), ("hi", BRACKET[1])):
            t0 = time.time()
            g, _ = gap_width(S, k)
            has = bool(np.isfinite(g) and g > 0)
            row[tag] = has
            row[f"{tag}_width"] = float(g)
            print(f"nh={nh}  kappa={k:.10f} ({tag}): "
                  f"{'GAP w=' + format(g, '.2e') if has else 'connected'}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        row["bracket_holds"] = (row["lo"] is False and row["hi"] is True)
        out.append(row)
        print(f"   -> published bracket "
              f"{'HOLDS' if row['bracket_holds'] else 'FAILS'} at nh={nh}",
              flush=True)

    os.makedirs(DATA, exist_ok=True)
    json.dump(dict(bracket=BRACKET, rows=out),
              open(os.path.join(DATA, "bisection_nharm_check.json"), "w"),
              indent=2)
    print("\nwrote data/bisection_nharm_check.json")


if __name__ == "__main__":
    main()
