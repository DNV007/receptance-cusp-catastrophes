# receptance-cusp-catastrophes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22672005.svg)](https://doi.org/10.5281/zenodo.22672005)

Code and data for locating **cusp-pair births** in a driven resonator network
from its **linear driving-point receptance** alone.

For a passive linear network carrying one localized conservative cubic, the
period-one cusp set lies on a fixed phase contour of the driving-point
receptance `G` at the nonlinear coordinate: `arg G = -150°` for a hardening
cubic, `-30°` for a softening one. Where that contour becomes tangent as a
network parameter is varied, a cusp pair is born, and the curvature of the
contour there decides whether the fold set opens a gap (*beaks*) or closes into
an isolated loop (*lips*).

This repository holds the solvers, the numerical experiments, and the data
behind those results: multi-harmonic balance, pseudo-arclength continuation,
Floquet stability, random-network ensembles, an inverse-design solver, and a
coupled-mode/Josephson device check.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, jax, jaxlib, matplotlib
```

Tested on Python 3.11.5 with the pinned versions. Run every script **from the
repository root**, not from `scripts/`:

```bash
python scripts/<name>.py
```

Some scripts import both `jax` and `scipy`. If your system interpreter breaks
one of them, use the virtual environment above rather than the system Python.

## Quick start

```bash
python scripts/make_fig_prl_construction.py   # the construction, numpy only, seconds
python scripts/ensemble_validity_boundary.py  # 100-network ensemble, ~10 min
python scripts/typeboundary_hb_verify.py      # full HB at the type boundary, ~35 min
```

Long runs print progress and write their results under `data/`.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | one driver per result, plus shared helpers (`ndof_hb.py`, `arclength_fold.py`, `_prx_style.py`) |
| `data/` | outputs the drivers produce and consume |
| `figures/` | where the figure scripts write their PDFs |
| `reproduce.md` | script-to-claim map: which script produces which number |

## Finding the code behind a number

Start from [`reproduce.md`](reproduce.md). Every quantitative result is listed
with the script that regenerates it and the value that script returns, together
with the conventions that are easy to get wrong — how a rung is counted, which
receptance is exact and which is modal, and where an estimator rather than the
physics sets the error.

## Cost

Most scripts finish in seconds. The exceptions are the ensembles and the
full harmonic-balance runs, which take ten minutes to about an hour each on one
core; those are noted in their module docstrings.

## License

MIT — see [`LICENSE`](LICENSE).

## Citing

Cite the archived software by its concept DOI, which always resolves to the
newest version:

> K. Sarkar, *receptance-cusp-catastrophes: code and data for cusp-pair births
> organized by the linear driving-point receptance*. Zenodo.
> [doi:10.5281/zenodo.22672005](https://doi.org/10.5281/zenodo.22672005)

`CITATION.cff` carries the same information in machine-readable form, and
GitHub's *Cite this repository* button reads it. The two manuscripts this
supports are listed at the top; their arXiv identifiers will be added here once
they are posted.
