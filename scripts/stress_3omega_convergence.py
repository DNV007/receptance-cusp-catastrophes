"""Truncation and fit-window convergence for the 3-Omega stress test.

The stress test deliberately amplifies the third harmonic, so a reference
calculation truncated at five harmonics is exactly the thing a referee should
distrust.  This script re-locates the stress-test cusps at a requested
truncation and, at each one, repeats the (Delta F)^{2/3} intercept over several
fit windows so that the estimator's own systematic is reported rather than
assumed small.  A high r^2 does not bound the intercept bias; the window spread
does.

Usage:  python stress_3omega_convergence.py <nh>
Output: data/stress_3omega_convergence_nh<nh>.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stress_3omega_resonance import (  # noqa: E402
    BETA1, G, bracket_edge, cusps, f_comb, f_fund, make_solver, modes_of,
    tongue_width,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

W3_LIST = [float(x) for x in
           os.environ.get("W3_LIST", "3.545,3.566,3.650").split(",")]
WINDOWS = [(0.0002, 0.0035), (0.0004, 0.0050), (0.0002, 0.0022)]
SIDE = -1


def tip_over_windows(S, edge, windows=WINDOWS, n_pts=10, **kw):
    """(Delta F)^{2/3} intercept for several fit windows below the edge."""
    out = []
    for inner, span in windows:
        oms = edge - np.linspace(inner, span, n_pts)
        dat = []
        for om in oms:
            dF = tongue_width(S, float(om), **kw)
            if np.isfinite(dF) and dF > 0:
                dat.append((om, dF ** (2.0 / 3.0)))
        if len(dat) < 5:
            continue
        x = np.array([d[0] for d in dat])
        y = np.array([d[1] for d in dat])
        A = np.vstack([x, np.ones_like(x)]).T
        (m, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - (res[0] / np.sum((y - y.mean()) ** 2) if len(res) else 0.0)
        out.append(dict(inner=inner, span=span, Om_tip=float(-b / m),
                        r2=float(r2), n=len(dat)))
    return out


def main():
    nh = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = []
    for w3 in W3_LIST:
        md = modes_of(w3)
        Omf = min(cusps(f_fund, md, 1.05, 1.28))
        cc = cusps(f_comb, md, 1.05, 1.28)
        Omc = min(cc, key=lambda o: abs(o - Omf)) if cc else float("nan")
        eta3 = np.sqrt(3) / 9 * abs(G(3 * Omf, md) / G(Omf, md))

        S = make_solver(md, nh, 256, BETA1)
        t0 = time.time()
        pad = 0.008
        Om_in = min([Omf, Omc]) - pad
        Om_out = max([Omf, Omc]) + 0.5 * pad
        edge = bracket_edge(S, Om_in, Om_out, F_max=5.0)
        if edge is None:
            print(f"nh={nh} w3={w3}: no tongue edge", flush=True)
            continue
        fits = tip_over_windows(S, edge, F_max=5.0)
        tips = np.array([f["Om_tip"] for f in fits])
        row = dict(nh=nh, w3=w3, eta3=float(eta3), Om_fund=float(Omf),
                   Om_comb=float(Omc), edge=float(edge),
                   Om_tip_mean=float(tips.mean()),
                   Om_tip_spread=float(tips.max() - tips.min()),
                   fits=fits)
        row["err_fund"] = row["Om_tip_mean"] - Omf
        row["err_comb"] = row["Om_tip_mean"] - Omc
        rows.append(row)
        print(f"nh={nh} w3={w3:6.3f} eta3={eta3:.4f}  edge={edge:.6f}  "
              f"Om_tip={row['Om_tip_mean']:.6f} (window spread "
              f"{row['Om_tip_spread']:.1e}, {len(fits)} windows)  "
              f"err_fund={row['err_fund']:+.2e} err_comb={row['err_comb']:+.2e}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(DATA, exist_ok=True)
    # OUT_TAG lets several ranges of W3_LIST run in parallel without clobbering
    # each other; merge_stress_convergence.py stitches the shards together.
    tag = os.environ.get("OUT_TAG", "")
    out = os.path.join(DATA, f"stress_3omega_convergence_nh{nh}{tag}.json")
    json.dump(dict(nh=nh, windows=WINDOWS, beta1=BETA1, rows=rows),
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
