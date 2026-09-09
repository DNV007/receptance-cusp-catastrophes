"""Find a lips bistable (Om,F) and Floquet-check both outer branches."""
import numpy as np
from ndof_hb import build_K
from floquet_ndof import closed_form_y, spectral_radius, check

# omega_2 = 0.95 at kappa = 0.333: the moderate-anharmonicity lips of
# Sec. V.C (eps = 0.30-0.41), replacing the earlier omega_2 = 0.60 case.
W, Z = [1.0, 0.95], [0.015, 0.02]
BETA1 = 0.147
KAPPA = 0.333
K = build_K(W, KAPPA)
# scan for a 3-root (bistable) point inside the born lens
found = None
for Om in np.linspace(1.020, 1.068, 25):
    for F in np.linspace(0.10, 1.2, 60):
        ys = closed_form_y(W, Z, K, Om, F, BETA1)
        if len(ys) == 3:
            found = (Om, F); break
    if found: break
print("lips bistable point:", found)
if found:
    check("lips born orbit (bistable)", W, Z, build_K, KAPPA,
          found[0], found[1], nh=7)
