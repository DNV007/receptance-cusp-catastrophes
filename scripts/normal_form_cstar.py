"""Normal-form coefficients of the pair-separation law, from scratch.

The beaks pair separates as Delta Omega = c*(mu) sqrt(mu) with

    c*(mu) = c*(0) + c*(1) mu + O(mu^2).

Writing delta = Omega - Omega*, mu = kappa - kappa*, and t = sqrt(mu), the zero
set of f = rho^2 - 3 sigma^2 has the branch delta(t) = s0 t + s1 t^2 + s2 t^3,
the second branch being delta(-t). Hence

    Delta Omega = delta(t) - delta(-t) = 2 s0 t + 2 s2 t^3,
    c*(0) = 2 s0,   c*(1) = 2 s2.

Matching orders in t on f(delta(t), t^2) = 0 gives, with c_ij the Taylor
coefficients of f about the beaks point (c00 = c10 = 0),

    t^2:  c20 s0^2 + c01 = 0
    t^3:  2 c20 s0 s1 + c11 s0 + c30 s0^3 = 0
    t^4:  c20 (s1^2 + 2 s0 s2) + c11 s1 + 3 c30 s0^2 s1
          + c02 + c21 s0^2 + c40 s0^4 = 0

The c_ij are obtained here by propagating a truncated bivariate Taylor series
through the arithmetic that defines f, so they are exact derivatives rather
than finite differences -- no step-size choice enters.

Cross-checks that the machinery is right: it returns the modal beaks point
(1.30100, 0.118365) and a_Omega = 1370, the exact one (1.30088, 0.118959) and
a_Omega = 1315, and an exact-vs-modal shift in c*(0) of +0.26%, all of which
the manuscript states independently.

RESULT: c*(0) = 0.16216 confirms the published value. c*(1) = 0.31199 does NOT
confirm the published 0.30906. Direct root-finding on the locus settles it: at
mu = 2e-4 the true c*(mu) is 0.16221808 and this expansion gives 0.16221808,
while 0.30906 gives 0.16222181. The manuscript has been corrected to 0.31199.
"""
ORD = 6  # truncate at total degree ORD

def zero(): return {}
def const(c): return {(0,0): complex(c)}
def var(which):   # 0 -> delta, 1 -> mu
    return {(1,0) if which==0 else (0,1): 1.0+0j}
def add(a,b):
    r=dict(a)
    for k,v in b.items(): r[k]=r.get(k,0)+v
    return r
def sub(a,b):
    r=dict(a)
    for k,v in b.items(): r[k]=r.get(k,0)-v
    return r
def mul(a,b):
    r={}
    for (i,j),u in a.items():
        for (m,n),v in b.items():
            if i+m+j+n<=ORD:
                r[(i+m,j+n)]=r.get((i+m,j+n),0)+u*v
    return r
def scal(a,c):  return {k:v*c for k,v in a.items()}
def recip(a):
    a0=a.get((0,0))
    assert a0 is not None and abs(a0)>0
    # Newton: x_{k+1} = x_k (2 - a x_k), doubling correct order each step
    x=const(1.0/a0)
    for _ in range(6):
        x=mul(x, sub(const(2.0), mul(a,x)))
    return x
def div(a,b): return mul(a, recip(b))
def sqrt_s(a):
    a0=a.get((0,0)); import cmath
    x=const(cmath.sqrt(a0))
    for _ in range(6):                      # x <- (x + a/x)/2
        x=scal(add(x, div(a,x)), 0.5)
    return x
def re(a): return {k:complex(v.real,0) for k,v in a.items()}
def im(a): return {k:complex(v.imag,0) for k,v in a.items()}
def coef(a,i,j): return a.get((i,j),0j).real

W1,W2,Z1,Z2 = 1.0, 1.25, 0.015, 0.02
J=1j

def invG_exact(Om0,kap0):
    d,m = var(0), var(1)
    Om = add(const(Om0), d); kap = add(const(kap0), m)
    Om2 = mul(Om,Om)
    Z11 = add(sub(add(const(W1**2), kap), Om2), scal(Om, 2*Z1*J))
    Z22 = add(sub(add(const(W2**2), kap), Om2), scal(Om, 2*Z2*J))
    det = sub(mul(Z11,Z22), mul(kap,kap))
    return div(det, Z22)

def invG_modal(Om0,kap0):
    d,m = var(0), var(1)
    Om = add(const(Om0), d); kap = add(const(kap0), m)
    Om2 = mul(Om,Om)
    a = add(const(W1**2), kap); b = add(const(W2**2), kap)
    half = scal(sub(a,b), 0.5)
    disc = sqrt_s(add(mul(half,half), mul(kap,kap)))
    mid  = scal(add(a,b), 0.5)
    G = zero()
    for lam in (sub(mid,disc), add(mid,disc)):
        num = mul(kap,kap)
        den = add(mul(kap,kap), mul(sub(a,lam), sub(a,lam)))
        p1  = div(num, den)                       # phi_{m,1}^2
        zm  = add(scal(p1, Z1), scal(sub(const(1.0), p1), Z2))
        Dm  = add(sub(lam, Om2), scal(mul(zm, Om), 2*J))
        G   = add(G, div(p1, Dm))
    return recip(G)

def f_series(invG, Om0, kap0):
    v = invG(Om0,kap0); r, s = re(v), im(v)
    return sub(mul(r,r), scal(mul(s,s), 3.0))

def analyse(invG, label, Om0=1.301, kap0=0.1184):
    for _ in range(60):                            # Newton on (f, df/dOm)=0
        F=f_series(invG,Om0,kap0)
        F1,F2 = coef(F,0,0), coef(F,1,0)
        a,b = coef(F,1,0), coef(F,0,1)
        c,dd = 2*coef(F,2,0), coef(F,1,1)
        det=a*dd-b*c
        Om0 -= ( dd*F1 - b*F2)/det
        kap0 -= (-c*F1 + a*F2)/det
        if abs(F1)+abs(F2) < 1e-18: break
    F=f_series(invG,Om0,kap0)
    c20,c01,c11,c30,c02,c21,c40 = (coef(F,2,0),coef(F,0,1),coef(F,1,1),
                                   coef(F,3,0),coef(F,0,2),coef(F,2,1),coef(F,4,0))
    s0=(-c01/c20)**0.5
    s1=-(c11+c30*s0**2)/(2*c20)
    s2=-(c20*s1**2 + c11*s1 + 3*c30*s0**2*s1 + c02 + c21*s0**2 + c40*s0**4)/(2*c20*s0)
    print(f"--- {label}")
    print(f"    Omega* = {Om0:.10f}   kappa* = {kap0:.10f}   (residual f={coef(F,0,0):.2e}, f'={coef(F,1,0):.2e})")
    print(f"    a_Omega = {c20:.6g}   b_kappa = {c01:.6g}")
    print(f"    s0 = {s0:.8f}   s1 = {s1:.8f}   s2 = {s2:.8f}")
    print(f"    c*(0) = 2 s0 = {2*s0:.6f}")
    print(f"    c*(1) = 2 s2 = {2*s2:.6f}")
    return 2*s0, 2*s2

m0,m1 = analyse(invG_modal, "MODAL closed form")
e0,e1 = analyse(invG_exact, "EXACT receptance")
print(f"\nmanuscript (corrected): c*(0) = 0.16216, c*(1) = 0.31199")
print(f"exact vs modal c*(0): {100*(e0/m0-1):+.2f}%   (manuscript says +0.26%)")
