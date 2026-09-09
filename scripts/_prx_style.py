"""Shared matplotlib styling and helpers for journal-grade figures.

Sizes: single column 3.4 in (86 mm), double column 7.0 in (178 mm).  Fonts are
set so that text in the figure matches the ~8-9 pt of the printed body text at
final size; figures must therefore be included at their natural width and never
rescaled by \\includegraphics, or the type size will no longer match.

Two helpers exist to remove the two commonest defects in the earlier figures:

  panel(ax, "a")  puts the panel letter OUTSIDE the axes, above the top-left
                  corner.  Panel letters placed inside collided with curves,
                  legends, and insets in four of the seven figures.

  legend(ax, ...) draws a legend with a white, semi-opaque background and a
                  hairline frame, so that a legend which must sit over data is
                  still readable and does not appear to be struck through.

The palette is Okabe-Ito, which is colourblind-safe and stays distinguishable
in greyscale when paired with the linestyle/marker conventions below.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
BLUE = "#0072B2"      # closed form / analytic / reduced model
VERM = "#D55E00"      # full harmonic balance
GREEN = "#009E73"
SKY = "#56B4E9"       # "before the birth" / lighter member of a triple
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHTGREY = "#BBBBBB"

# Semantic aliases used across figures, so colour means the same thing
# everywhere: analytic prediction vs full-HB measurement.
C_ANALYTIC = BLUE
C_HB = VERM
C_BEFORE = SKY
C_AT = BLUE
C_AFTER = VERM

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.6,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.labelpad": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.4,
})

COL_SINGLE = (3.4, 2.6)
COL_DOUBLE = (7.0, 2.8)


def panel(ax, letter, dx=-0.020, dy=0.028, **kw):
    """Panel letter outside the axes, above the top-left corner.

    Keeping it out of the data area is what makes it collision-proof; placing
    it inside is what broke the earlier versions of Figs. 1, 3 and 6.
    """
    ax.text(dx, 1.0 + dy, f"({letter})", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8, **kw)


def legend(ax, *args, over_data=True, **kw):
    """Legend that stays readable when it has to sit over curves."""
    kw.setdefault("fontsize", 7)
    kw.setdefault("handlelength", 1.6)
    if over_data:
        kw.setdefault("frameon", True)
        kw.setdefault("framealpha", 0.92)
        kw.setdefault("facecolor", "white")
        kw.setdefault("edgecolor", "none")
    leg = ax.legend(*args, **kw)
    if over_data and leg is not None:
        leg.get_frame().set_linewidth(0.0)
    return leg


def tight(fig, **kw):
    kw.setdefault("pad", 0.25)
    kw.setdefault("w_pad", 0.6)
    kw.setdefault("h_pad", 0.5)
    fig.tight_layout(**kw)


def save(fig, path, **kw):
    fig.savefig(path, **kw)
    print("wrote", path)
    plt.close(fig)
