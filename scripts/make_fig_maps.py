"""Fig (both topologies): full-HB bistability maps below/at/above a birth.
2x3 grid. Top row beaks (hub), bottom row lips (2-DOF). Each panel: closed-form
bistable region shaded + full-HB fold forcings (from data/hb_bistability_maps.csv)
overlaid. Beaks -> a gap opens in a connected band; lips -> an isolated lens is
born from nothing."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import _prx_style as st

ROOT = Path(__file__).parent.parent
BETA1 = 0.147
C3 = 0.75 * BETA1

with (ROOT / "data" / "hb_bistability_maps.csv").open() as fh:
    _rows = list(csv.DictReader(fh))


def hb_points(config, k):
    om = [float(r["omega"]) for r in _rows
          if r["config"] == config and abs(float(r["kappa"]) - k) < 1e-6]
    Ff = [float(r["F_fold"]) for r in _rows
          if r["config"] == config and abs(float(r["kappa"]) - k) < 1e-6]
    return np.array(om), np.array(Ff)


def hb_widths(config, k):
    """Tongue width max(F_fold) - min(F_fold) at each Omega with >=2 folds."""
    by_om: dict[float, list[float]] = {}
    for r in _rows:
        if r["config"] == config and abs(float(r["kappa"]) - k) < 1e-6:
            by_om.setdefault(float(r["omega"]), []).append(float(r["F_fold"]))
    oms = sorted(o for o, v in by_om.items() if len(v) >= 2)
    return (np.array(oms),
            np.array([max(by_om[o]) - min(by_om[o]) for o in oms]))


def K_hub(om, k):
    N = len(om); K = np.diag(np.asarray(om, float) ** 2).copy()
    for j in range(1, N):
        K[0, 0] += k; K[j, j] += k; K[0, j] -= k; K[j, 0] -= k
    return K


def K_chain(om, k):
    N = len(om); K = np.diag(np.asarray(om, float) ** 2).copy()
    for i in range(N - 1):
        K[i, i] += k; K[i + 1, i + 1] += k; K[i, i + 1] -= k; K[i + 1, i] -= k
    return K


def rs(om, ze, Om, k, Kf):
    N = len(om)
    Z = Kf(om, k) - Om ** 2 * np.eye(N) + 1j * Om * np.diag([2 * z for z in ze])
    inv = 1.0 / np.linalg.solve(Z, np.eye(N)[:, 0])[0]
    return inv.real, inv.imag


def cf_region(om, ze, k, Kf, band, n=600):
    oms = np.linspace(band[0], band[1], n)
    Flo = np.full(n, np.nan); Fhi = np.full(n, np.nan)
    for i, Om in enumerate(oms):
        r, s = rs(om, ze, Om, k, Kf)
        if r < 0 and r ** 2 > 3 * s ** 2:
            d = np.sqrt(r ** 2 - 3 * s ** 2)
            Fs = []
            for y in [(-2 * r + d) / (3 * C3), (-2 * r - d) / (3 * C3)]:
                if y > 0:
                    F2 = y * ((r + C3 * y) ** 2 + s ** 2)
                    if F2 > 0: Fs.append(np.sqrt(F2))
            if len(Fs) == 2:
                Flo[i], Fhi[i] = min(Fs), max(Fs)
    return oms, Flo, Fhi


ROWS = [
    ("beaks", [1.0, 1.15, 1.30], [0.015, 0.02, 0.02], K_hub,
     [0.05, 0.10, 0.14], (1.11, 1.28)),
    # omega_2 = 0.95 (kappa* = 0.2832) keeps eps = 0.30-0.41 at the born
    # cusps, inside the validity window; the earlier omega_2 = 0.60 case sat
    # at eps ~ 0.7 where only part of the lens is realized (Sec. V.D).
    ("lips", [1.0, 0.95], [0.015, 0.02], K_chain,
     [0.263, 0.303, 0.333], (1.00, 1.09)),
]
LABELS = {0.05: r"$\kappa<\kappa^\ast$", 0.10: r"$\kappa\gtrsim\kappa^\ast$",
          0.14: r"$\kappa>\kappa^\ast$", 0.263: r"$\kappa<\kappa^\ast$",
          0.303: r"$\kappa\gtrsim\kappa^\ast$", 0.333: r"$\kappa>\kappa^\ast$"}

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))

# Pre-compute each panel's data extent.  Panels are individually scaled --
# the tongue widths in a row span more than a decade -- but a panel with NO
# bistability has no extent of its own, and defaulting it to an arbitrary
# range made the empty panel look like a different measurement.  Empty panels
# instead borrow the largest scale in their row, so "nothing here" is read
# against the scale on which the lens later appears.
_tops = {}
for ri, (lab, om, ze, Kf, kappas, band) in enumerate(ROWS):
    for ci, k in enumerate(kappas):
        _, Flo, Fhi = cf_region(om, ze, k, Kf, band)
        dF = Fhi - Flo
        _, hdF = hb_widths(lab, k)
        _tops[(ri, ci)] = float(np.nanmax([
            np.nanmax(dF) if np.any(~np.isnan(dF)) else 0.0,
            hdF.max() if hdF.size else 0.0]))
_rowmax = {ri: max(_tops[(ri, c)] for c in range(3)) for ri in (0, 1)}

for ri, (lab, om, ze, Kf, kappas, band) in enumerate(ROWS):
    for ci, k in enumerate(kappas):
        ax = axes[ri, ci]
        # Plot the tongue WIDTH rather than the (Omega, F) region: the lips
        # tongue is only a few percent wide in F, so on a shared F axis the
        # lens is invisible.  The width makes the topology the visible thing --
        # a connected band that develops a zero gap (beaks) versus an isolated
        # bump born from nothing (lips) -- and puts both rows on one quantity.
        oms, Flo, Fhi = cf_region(om, ze, k, Kf, band)
        dF = Fhi - Flo
        ax.fill_between(oms, 0.0, dF, where=~np.isnan(dF),
                        color=st.C_ANALYTIC, alpha=0.25, lw=0,
                        label="exact receptance")
        how, hdF = hb_widths(lab, k)
        ax.plot(how, hdF, "o", ms=2.6, mfc=st.C_HB, mec="white", mew=0.4,
                ls="", label="full HB")
        ax.set_xlim(*band)
        top = _tops[(ri, ci)] or _rowmax[ri]
        ax.set_ylim(0, top * 1.30)
        ax.text(0.04, 0.93, f"{LABELS[k]}\n$\\kappa={k}$", transform=ax.transAxes,
                va="top", fontsize=7.5)
        # An empty panel is the result, not a failure: say so explicitly.
        if len(how) == 0 and not np.any(~np.isnan(dF)):
            ax.text(0.5, 0.45, "no bistability", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color="0.45",
                    style="italic")
        if ci == 0:
            ax.set_ylabel(r"tongue width $\Delta F$")
        if ri == 1:
            ax.set_xlabel(r"frequency $\Omega$")
st.legend(axes[0, 2], loc="upper right", fontsize=6.5, handlelength=1.0)
fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.7)
# Row labels are placed in FIGURE coordinates after tight_layout, clear of the
# tick numbers; anchoring them to the axes put them on top of the y labels.
fig.subplots_adjust(left=0.135)
for ri, lab in ((0, "beaks\n(hub)"), (1, "lips\n(2-DOF)")):
    box = axes[ri, 0].get_position()
    fig.text(0.022, 0.5 * (box.y0 + box.y1), lab, rotation=90,
             va="center", ha="left", fontweight="bold", fontsize=8)
out = ROOT / "figures" / "fig_maps.pdf"
plt.savefig(out)
print(f"wrote {out}")
