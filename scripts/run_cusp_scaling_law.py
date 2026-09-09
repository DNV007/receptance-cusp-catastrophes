"""Cusp catastrophe scaling: tongue width DeltaF(Omega) ~ (Omega-Omega_c)^(3/2).

Uses existing data/cusp_both_folds.csv (SN1, SN2 traced via F-continuation
at v8 baseline kappa=0.10, beta_1=0.147). Cusp tip at Om_c in [1.0655, 1.066].

Standard cusp catastrophe x^3 + a x + b = 0 has bifurcation set
    b^2 = (4/27) |a|^3
i.e. fold-pair separation in b scales as |a|^(3/2). Mapping
    a <-> (Omega - Omega_c),   b <-> (F - F_c)
predicts DeltaF(Omega) := F_SN1(Omega) - F_SN2(Omega) ~ (Omega - Omega_c)^(3/2).

Output: log-log fit of DeltaF vs (Omega - Omega_c). Expected slope = 1.5.
Saves data/cusp_scaling_law.csv and figures/cusp_scaling_law.png.
"""
import csv

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _branch(rows, name):
    """{omega: F_fold} for one fold branch, dropping blank entries."""
    out = {}
    for r in rows:
        if r["branch"] != name or not r["F_fold"].strip():
            continue
        out[float(r["omega"])] = float(r["F_fold"])
    return out


with (ROOT / "data" / "cusp_both_folds.csv").open() as fh:
    rows = list(csv.DictReader(fh))
sn1 = _branch(rows, "SN1")
sn2 = _branch(rows, "SN2")

# Frequencies carrying both folds, kept in the order SN1 appears in the file.
common = [o for o in sn1 if o in sn2]
om = np.array(common)
F1 = np.array([sn1[o] for o in common])
F2 = np.array([sn2[o] for o in common])
dF = np.abs(F1 - F2)
# Restrict to well-resolved tongue widths (F-step quantization is 2e-4).
mask = dF > 1e-3
om, dF, F1, F2 = om[mask], dF[mask], F1[mask], F2[mask]

# Find Om_c by minimizing log-log fit residual; cusp tip prior in [1.0640, 1.0670].
def fit(om_c):
    keep = om > om_c
    if keep.sum() < 4: return 0, 0, np.inf
    x = np.log(om[keep] - om_c)
    y = np.log(dF[keep])
    s, c = np.polyfit(x, y, 1)
    resid = y - (s * x + c)
    return s, c, np.sqrt(np.mean(resid**2))

best = min(np.linspace(1.0640, 1.0680, 401), key=lambda c: fit(c)[2])
slope, intercept, rms = fit(best)
print(f"Cusp scaling fit (v8 baseline, lower cusp):")
print(f"  Om_c (best fit)   = {best:.5f}")
print(f"  slope             = {slope:.4f}   (cusp prediction: 1.5)")
print(f"  intercept         = {intercept:.4f}")
print(f"  RMS log-residual  = {rms:.4f}")
print(f"  N points          = {len(om)}")
print(f"  Om range          = [{om.min():.4f}, {om.max():.4f}]")
print(f"  DeltaF range      = [{dF.min():.4e}, {dF.max():.4e}]")

# Save
with (ROOT / "data" / "cusp_scaling_law.csv").open("w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["omega", "F_SN1", "F_SN2", "DeltaF", "Om_minus_Omc"])
    for o, f1, f2, d in zip(om, F1, F2, dF):
        w.writerow([repr(float(o)), repr(float(f1)), repr(float(f2)),
                    repr(float(d)), repr(float(o - best))])

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(om, dF, "o-", ms=4)
ax[0].axvline(best, color="r", ls="--", label=f"$\\Omega_c$={best:.4f}")
ax[0].set(xlabel="$\\Omega$", ylabel="$\\Delta F = F_{SN1}-F_{SN2}$",
          title="Tongue width vs $\\Omega$")
ax[0].legend()
xx = np.log(om - best)
yy = np.log(dF)
ax[1].plot(xx, yy, "o", ms=5, label="full HB")
xs = np.linspace(xx.min(), xx.max(), 50)
ax[1].plot(xs, slope * xs + intercept,
           "r--", label=f"fit: slope={slope:.3f}")
ax[1].plot(xs, 1.5 * xs + (intercept + (slope - 1.5) * xx.mean()),
           "k:", label="cusp law: 3/2")
ax[1].set(xlabel="$\\log(\\Omega-\\Omega_c)$", ylabel="$\\log\\,\\Delta F$",
          title=f"Cusp scaling (slope={slope:.3f}, RMS={rms:.3f})")
ax[1].legend()
plt.tight_layout()
plt.savefig(ROOT / "figures" / "cusp_scaling_law.png", dpi=140)
print(f"\nWrote data/cusp_scaling_law.csv and figures/cusp_scaling_law.png")
