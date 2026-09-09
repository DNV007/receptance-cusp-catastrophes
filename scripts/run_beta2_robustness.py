"""Robustness of cusp-of-cusps to small beta_2 != 0 (symmetry breaking).

Closed form derived in the paper assumed beta_2 = 0; with beta_2 != 0, the
modal cubic projection gets cross-coupling terms that obstruct the
single-fundamental closed form. We instead use full HB (or numerical
solution of the modal HB system) to locate the upper-pair cusp tips for
small beta_2 and check that:
  (a) the cusp-of-cusps point still exists and drifts continuously;
  (b) the sqrt(mu) law survives.

Strategy: solve the bivariate modal HB at fundamental order including
both beta_1 x_1^3 and beta_2 x_2^3, then track upper cusps numerically.
We use the projected scalar form for beta_2 small by treating the
beta_2 contribution perturbatively: x_2 from leading modal solution
gets cubed and projected back, generating an effective correction to
the scalar Duffing coefficients.
"""
import numpy as np
from scipy.optimize import brentq

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
B1 = 0.147


def modal_admittance(Om, kappa):
    """Return G(Om), and the modal data needed to evaluate beta_2 correction."""
    K = np.array([[W1**2 + kappa, -kappa], [-kappa, W2**2 + kappa]])
    evals, evecs = np.linalg.eigh(K)
    wa2, wb2 = evals
    pa1, pb1 = evecs[0, 0], evecs[0, 1]
    pa2, pb2 = evecs[1, 0], evecs[1, 1]
    pa, pb = evecs[:, 0], evecs[:, 1]
    Cmat = np.diag([2 * Z1, 2 * Z2])
    za = 0.5 * (pa @ Cmat @ pa)
    zb = 0.5 * (pb @ Cmat @ pb)
    Da = wa2 - Om**2 + 2j * za * Om
    Db = wb2 - Om**2 + 2j * zb * Om
    G = pa1**2 / Da + pb1**2 / Db
    # For beta_2 correction we need the "transfer to coordinate 2":
    # X_2 = phi_a2 A_a + phi_b2 A_b with A_m = phi_m1 (F - cubic)/D_m.
    # X_2 / X_1 ratio depends on amplitude. To leading order in beta_2,
    # the cubic contribution from beta_2 x_2^3 projects back to mode m
    # via phi_m2; so the effective scalar equation gets an additional term
    # (3/4) beta_2 phi_{m,2} |X_2|^2 X_2 -> after projection sums to
    # (3/4) beta_2 G_2(Om) |X_2|^2 X_2  (in coordinate 2 units),
    # where G_2(Om) = phi_a2^2/D_a + phi_b2^2/D_b is the response of x_2 to
    # a force on x_2 (transfer for the second coordinate's self-response).
    G_2_self = pa2**2 / Da + pb2**2 / Db
    # Coupling: linear ratio X_2 / X_1 at leading order = T(Om)
    # X_1 = phi_{a1} A_a + phi_{b1} A_b; A_m = phi_{m,1} F / D_m
    # X_2 = phi_{a2} A_a + phi_{b2} A_b
    # So X_2 = sum_m (phi_{m,1} phi_{m,2} / D_m) F = T(Om) F, X_1 = G(Om) F
    T = (pa1 * pa2 / Da + pb1 * pb2 / Db)
    ratio_X2_X1 = T / G  # complex, frequency-dependent
    return G, ratio_X2_X1, G_2_self


def cusp_residual_with_b2(Om, kappa, b2):
    """Effective cusp residual including small beta_2 perturbation.

    The leading-order modal HB with both beta_1, beta_2 in the cubic still
    gives a scalar amplitude equation; with beta_2 small, treat beta_2
    correction perturbatively: replace |X_1|^2 X_1 cubic source by
    (3/4) [beta_1 |X_1|^2 X_1 + beta_2 (|X_2|^2 X_2 projected to driven coord)].

    For the projection, beta_2 x_2^3 generates a force on coord 2 only;
    after modal decomposition this contributes a force on each mode m
    proportional to phi_{m,2}, giving back-reaction on X_1 through G(Om).

    Effective scalar equation (perturbative in b2):
       F = X_1 / G - (3/4) beta_2 (T*(Om)/G(Om)) |X_2|^2 X_2 / X_1 * X_1
                  + (3/4) beta_1 |X_1|^2 X_1
    With X_2 = T X_1 / G (linear), |X_2|^2 = |T/G|^2 |X_1|^2,
    and the back-reaction term reduces to a renormalization of the
    cubic coefficient: beta_eff = beta_1 + beta_2 |T/G|^2 (T*/G).
    Take real part for the cusp condition (imaginary part gives small phase).
    """
    G, R, _ = modal_admittance(Om, kappa)
    H = 1.0 / G
    rho, sig = H.real, H.imag
    # Effective cubic coefficient including beta_2 contribution.
    # The full effective scalar is F = X_1 [H + (3/4) beta_eff |X_1|^2]
    # where beta_eff is *complex* in general. For the cusp condition we
    # treat beta_eff as a perturbation that shifts the locations of folds.
    # Correct perturbative formula: with X_2 ~ (T/G) X_1 = R X_1 at linear order,
    # |X_2|^2 X_2 = R |R|^2 |X_1|^2 X_1, and the source enters scalar equation
    # multiplied by another factor (T/G) = R, giving R^2 |R|^2.
    beta_eff_complex = B1 + b2 * (np.abs(R)**2) * (R * R)
    # The scalar polynomial is F^2 = y |H + (3/4) beta_eff y|^2.
    # Saddle-nodes: dF^2/dy = 0. Cusp: also d^2F^2/dy^2 = 0.
    # Let A = H + c y where c = (3/4) beta_eff (complex).
    # F^2 = y |A|^2. dF^2/dy = |A|^2 + 2y Re(A* c) = 0 = (rho + cR y)^2 + (sig + cI y)^2 + 2y[(rho+cR y) cR + (sig+cI y) cI]
    # d2F^2/dy2 = 4 Re(A* c) + 2y |c|^2.
    # For cusp: solve d2/dy2 = 0 -> Re(A* c) = -y |c|^2 / 2
    cR = 0.75 * beta_eff_complex.real
    cI = 0.75 * beta_eff_complex.imag
    c2 = cR * cR + cI * cI

    def d2F2(y):
        A_re = rho + cR * y
        A_im = sig + cI * y
        return 4 * (A_re * cR + A_im * cI) + 2 * y * c2

    def dF2(y):
        A_re = rho + cR * y
        A_im = sig + cI * y
        return (A_re * A_re + A_im * A_im) + 2 * y * (A_re * cR + A_im * cI)

    ys = np.linspace(1e-6, 5.0, 4001)
    vals = np.array([d2F2(y) for y in ys])
    sc = np.where(np.diff(np.sign(vals)) != 0)[0]
    best = None
    for i in sc:
        try:
            yr = brentq(d2F2, ys[i], ys[i + 1])
        except ValueError:
            continue
        v = dF2(yr)
        if best is None or abs(v) < abs(best):
            best = v
    return best if best is not None else np.nan


def find_upper_cusps(kappa, b2, om_lo=1.20, om_hi=1.45, n=2001):
    oms = np.linspace(om_lo, om_hi, n)
    res = np.array([cusp_residual_with_b2(o, kappa, b2) for o in oms])
    sc = np.where(np.diff(np.sign(res)) != 0)[0]
    out = []
    for i in sc:
        if np.isnan(res[i]) or np.isnan(res[i + 1]):
            continue
        try:
            Om = brentq(lambda o: cusp_residual_with_b2(o, kappa, b2),
                        oms[i], oms[i + 1])
            G, _, _ = modal_admittance(Om, kappa)
            if (1.0 / G).real < 0:
                out.append(Om)
        except ValueError:
            continue
    return sorted(out)


def find_kstar(b2):
    lo, hi = 0.05, 0.20
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n = len(find_upper_cusps(mid, b2, n=801))
        if n >= 2:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-5:
            break
    return 0.5 * (lo + hi)


print(f"  {'beta_2':>10} {'kappa*':>10} {'shift %':>10}")
k0 = find_kstar(0.0)
print(f"  {0.0:10.4f} {k0:10.5f} {0.0:10.3f}")
for b2 in [0.005, 0.010, 0.020, 0.050, 0.100, -0.010, -0.050, -0.100]:
    k = find_kstar(b2)
    print(f"  {b2:10.4f} {k:10.5f} {100*(k-k0)/k0:+10.3f}")

print("\n  At kappa = k* + 0.02, do upper cusps still exist for moderate b2?")
for b2 in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
    cusps = find_upper_cusps(0.13837 + 0.0, b2, n=2001)  # k* roughly 0.1184
    cusps_extra = find_upper_cusps(0.1384, b2, n=2001)
    print(f"  b2={b2:+.3f}: cusps at kappa=0.1384: {[f'{x:.4f}' for x in cusps_extra]}")
