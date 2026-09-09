"""Independent exponent tests: free (C, x_c, p) nonlinear regressions that do NOT
presuppose the exponent, with 1-sigma uncertainty from the covariance and
fitting-window sensitivity. (a) 3/2 cusp closure: DeltaF = C|Om-Om_c|^p from the
full-HB tongue width. (b) 1/2 pair separation: DeltaOmega = C(kappa-kstar)^q."""
import csv, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

ROOT = Path(__file__).parent.parent

def fit3(x, y, p0):
    f = lambda x, C, xc, p: C*np.abs(x-xc)**p
    popt, pcov = curve_fit(f, x, y, p0=p0, maxfev=20000)
    se = np.sqrt(np.diag(pcov))
    return popt, se

print("=== (a) 3/2 cusp closure: free (C, Om_c, p) ===")
rows=list(csv.DictReader(open(ROOT/"data"/"cusp_scaling_law.csv")))
om=np.array([float(r["omega"]) for r in rows])
dF=np.array([float(r["DeltaF"]) for r in rows])
o=np.argsort(om); om,dF=om[o],dF[o]
print(f"  {len(om)} points, Om in [{om.min():.4f},{om.max():.4f}], "
      f"DeltaF in [{dF.min():.4g},{dF.max():.4g}]")
popt,se=fit3(om,dF,[1.0,1.066,1.5])
print(f"  full window: p = {popt[2]:.3f} +/- {se[2]:.3f}, "
      f"Om_c = {popt[1]:.5f} +/- {se[1]:.5f}, C={popt[0]:.3g}")
# window sensitivity: nearest-N-to-tip and farthest
for lab, mask in [("drop 3 nearest tip", np.argsort(np.abs(om-popt[1]))[3:]),
                  ("drop 3 farthest",    np.argsort(np.abs(om-popt[1]))[:-3]),
                  ("inner half",         np.argsort(np.abs(om-popt[1]))[:len(om)//2])]:
    try:
        p2,s2=fit3(om[mask],dF[mask],[popt[0],popt[1],popt[2]])
        print(f"    {lab:18s}: p={p2[2]:.3f}+/-{s2[2]:.3f}, Om_c={p2[1]:.5f}+/-{s2[1]:.5f} ({len(mask)} pts)")
    except Exception as e:
        print(f"    {lab}: fit failed ({e})")

print("\n=== (b) 1/2 pair separation: free (C, kstar, q) ===")
# tab:beaks_hb full-HB pair separations
kap=np.array([0.13,0.14,0.15,0.17,0.20])
Om_m=np.array([1.29719,1.29824,1.29997,1.30436,1.31214])
Om_p=np.array([1.31951,1.32630,1.33303,1.34638,1.36624])
dOm=Om_p-Om_m
print(f"  kappa={list(kap)}")
print(f"  DeltaOmega={[round(x,5) for x in dOm]}")
popt,se=fit3(kap,dOm,[0.16,0.117,0.5])
print(f"  free 3-param: q = {popt[2]:.3f} +/- {se[2]:.3f}, "
      f"kstar = {popt[1]:.5f} +/- {se[1]:.5f}, C={popt[0]:.3g}")
# with kstar fixed at reduced-model value, free q:
f2 = lambda x,C,q: C*np.abs(x-0.1184)**q
p2,pc2=curve_fit(f2,kap,dOm,p0=[0.16,0.5],maxfev=20000)
print(f"  kstar=0.1184 fixed: q = {p2[1]:.3f} +/- {np.sqrt(pc2[1,1]):.3f}")
