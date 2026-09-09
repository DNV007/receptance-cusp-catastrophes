"""Independent check of the Josephson/Kerr device numbers.

The device note supplies J*, the birth detuning, rho* and the cusp occupation
as model estimates and asks for them to be reproduced before use.  This does
that from scratch for the two-mode coupled-mode system

    [[D1 + i k1/2, -J], [-J, D2 + i k2/2]] a  +  (K |a1|^2 a1, 0)^T = drive,

with Dj = fj - fd.  All frequencies are ordinary frequencies in MHz (i.e.
every quantity already divided by 2 pi), so no 2 pi factors appear below.

One point of physics is worth flagging up front.  The Josephson self-Kerr is
NEGATIVE, so this device is a SOFTENING nonlinearity.  The physical branch is
then c*rho < 0 with c = K < 0, i.e. rho > 0, and the cusp condition
rho^2 = 3 sigma^2 selects

    arg G = -/+ 30 degrees,

not the -/+ 150 degrees of the hardening mechanical case.  Using the hardening
contour here would look for the organizer on the wrong branch entirely.

That is not hypothetical: this script previously carried F2 = +50, putting the
auxiliary mode ABOVE the nonlinear one.  For a softening device the organizers
lie BELOW the resonances, so with the auxiliary above there is no physical cusp
at all; fsolve then converged on the mirror (hardening, rho < 0) root and the
script reported arg G = -150 deg, a negative rho, a negative photon number, and
zero cusps above the birth -- none of which contradicts the manuscript, which
places the auxiliary 50 MHz below.  F2 is negative below for that reason.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve

# central parameter set from the device note (MHz)
F1 = 0.0
F2 = -50.0           # auxiliary mode 50 MHz BELOW (softening device)
K1 = 5.0            # nonlinear-mode linewidth
K2 = 0.50           # auxiliary-mode linewidth
K_KERR = -0.080     # Josephson self-Kerr, NEGATIVE (softening)


def Zmat(fd, J, f2=F2, k1=K1, k2=K2):
    D1 = F1 - fd
    D2 = f2 - fd
    return np.array([[D1 + 1j * k1 / 2, -J],
                     [-J, D2 + 1j * k2 / 2]])


def G_of(fd, J, **kw):
    return np.linalg.inv(Zmat(fd, J, **kw))[0, 0]


def rho_sigma(fd, J, **kw):
    inv = 1.0 / G_of(fd, J, **kw)
    return inv.real, inv.imag


def f_disc(fd, J, **kw):
    r, s = rho_sigma(fd, J, **kw)
    return r * r - 3.0 * s * s


def physical(fd, J, **kw):
    """Cusp requires c*rho < 0; K < 0 here, so rho > 0."""
    r, _ = rho_sigma(fd, J, **kw)
    return r > 0


def cusp_freqs(J, lo=F1 - 90.0, hi=F1 - 10.0, n=40001, **kw):
    fds = np.linspace(lo, hi, n)
    fv = np.array([f_disc(x, J, **kw) for x in fds])
    out = []
    for i in np.where(np.diff(np.sign(fv)))[0]:
        x = brentq(lambda t: f_disc(t, J, **kw), fds[i], fds[i + 1], xtol=1e-12)
        if physical(x, J, **kw):
            out.append(float(x))
    return sorted(out)


def birth(J0=2.5, fd0=-50.05, **kw):
    """Solve f = 0 and d f / d fd = 0 simultaneously."""
    h = 1e-6

    def sys(v):
        fd, J = v
        d1 = (f_disc(fd + h, J, **kw) - f_disc(fd - h, J, **kw)) / (2 * h)
        return [f_disc(fd, J, **kw), d1]

    fd, J = fsolve(sys, [fd0, J0], full_output=False)
    return float(fd), float(J)


if __name__ == "__main__":
    print("=== independent reproduction of the device-note numbers ===")
    fd_s, J_s = birth()
    r_s, s_s = rho_sigma(fd_s, J_s)
    argG = np.degrees(np.angle(G_of(fd_s, J_s)))
    n_c = 2.0 * r_s / (3.0 * abs(K_KERR))
    print(f"  birth coupling      J*        = {J_s:.4f} MHz   "
          f"(note: 2.48)")
    print(f"  birth detuning      fd* - f1  = {fd_s:.4f} MHz   "
          f"(note: -50.07)")
    print(f"  inverse receptance  rho*      = {r_s:.3f} MHz    "
          f"(note: 44.0)")
    print(f"                      sigma*    = {s_s:.4f} MHz")
    print(f"  check rho^2 - 3 sigma^2       = {r_s**2 - 3*s_s**2:.3e}")
    print(f"  arg G at the birth            = {argG:+.2f} deg   "
          f"(softening branch: -/+30)")
    print(f"  cusp occupation     n_c       = {n_c:.1f} photons "
          f"(note: 3.7e2)")

    print("\n=== cusp pair above the birth ===")
    for J in (3.0, 4.0, 5.0, 6.0):
        cs = cusp_freqs(J)
        if len(cs) >= 2:
            print(f"  J = {J:.1f} MHz: cusps at "
                  f"{cs[0]:.4f}, {cs[-1]:.4f} MHz   "
                  f"separation {cs[-1]-cs[0]:.4f} MHz")
        else:
            print(f"  J = {J:.1f} MHz: {len(cs)} cusp(s) {cs}")
