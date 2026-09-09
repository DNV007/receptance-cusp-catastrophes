"""Drive port, transfer receptance H, and required input power.

Section II separates the intrinsic organizer, carried by G, from
accessibility through a chosen port, carried by H = e1^T Z^-1 d.  The device
section quoted an occupation but never applied that separation.  This closes
it.

For the two-mode network Z = [[A, -J], [-J, B]] one has

    G = (Z^-1)_11 = B/det,      H_2 = (Z^-1)_12 = J/det,

so driving the nonlinear mode gives H = G and a ratio of one, while driving
through the auxiliary mode gives H/G = J/B.  Neither has a zero at finite J:
a transfer zero of H requires at least three modes.  Accessibility is
therefore automatic here, and the port question reduces to how much power
each choice costs.

Because the birth sits on the auxiliary pole, |B| is small there, and driving
through the auxiliary port is correspondingly cheaper.

Power follows from the drive amplitude by the standard input-output relation.
With every frequency in Hz, F_ang = 2 pi F = sqrt(kappa_ext,ang) s_in and
|s_in|^2 = 2 pi F^2/kappa_ext, so P_in = hbar omega_d |s_in|^2.
"""
from __future__ import annotations

import numpy as np

import kerr_inverse_design as kid

HBAR = 1.054571817e-34
F1_ABS = 5.17e9          # nonlinear-mode frequency, Hz
DF = -50.0               # MHz


def report(kext_MHz, drive_on="nonlinear"):
    Js = kid.birth_J(DF)
    p = kid.pair_near(Js + 1e-4, DF)
    fd = 0.5 * (p[0] + p[1])
    rho, sig = kid.rho_sigma(fd, Js, DF)
    K = kid.K_KERR

    y = -2.0 * rho / (3.0 * K)                      # cusp occupation
    Feff = np.sqrt(y * ((rho + K * y) ** 2 + sig ** 2))   # |F H/G| at the cusp

    B = kid.F1 + DF - fd + 1j * kid.K2 / 2
    ratio = 1.0 if drive_on == "nonlinear" else abs(Js / B)   # |H/G|
    F = Feff / ratio                                # required drive amplitude

    f_d = F1_ABS + fd * 1e6
    kext = kext_MHz * 1e6
    s_in2 = 2 * np.pi * (F * 1e6) ** 2 / kext
    P = HBAR * 2 * np.pi * f_d * s_in2
    return dict(y=y, Feff=Feff, ratio=ratio, F=F, P=P,
                dBm=10 * np.log10(P / 1e-3))


if __name__ == "__main__":
    print("drive on the NONLINEAR mode (d = e1, H = G):")
    for kext in (2.1, 3.0, 4.8):
        r = report(kext, "nonlinear")
        print(f"   kappa_ext/2pi = {kext:.1f} MHz:  n_c = {r['y']:.0f}, "
              f"P_in = {r['P']:.2e} W = {r['dBm']:.1f} dBm")

    print("\ndrive through the AUXILIARY mode (d = e2, H/G = J/B):")
    for kext in (2.1, 3.0, 4.8):
        r = report(kext, "auxiliary")
        print(f"   kappa_ext/2pi = {kext:.1f} MHz:  |H/G| = {r['ratio']:.2f}, "
              f"P_in = {r['P']:.2e} W = {r['dBm']:.1f} dBm")

    a = report(3.0, "nonlinear")
    b = report(3.0, "auxiliary")
    print(f"\nport advantage at kappa_ext/2pi = 3 MHz: "
          f"{a['dBm'] - b['dBm']:.1f} dB in favour of the auxiliary port")
    print(f"  (|H/G| = {b['ratio']:.2f}, so the power ratio is |H/G|^2 = "
          f"{b['ratio']**2:.1f})")
