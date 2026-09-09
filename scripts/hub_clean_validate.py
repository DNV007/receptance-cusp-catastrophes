"""CLEAN full-HB validation of a NEW beaks in a 3-resonator hub at accessible
drive (eps~0.2-0.3, matching the regime of the paper's validated 2-DOF beaks).
Config om=[1.0,1.15,1.30] satellites; the upper beaks pair (Om~1.18,1.21) is
born as kappa crosses ~0.07-0.08. Scan Om across the rung at kappa below/above
the birth; a beaks shows as TWO separated bistable sub-bands (gap between)
appearing only above kappa*."""
import numpy as np
from arclength_fold import make_continuation, count_folds

W, Z = [1.0, 1.15, 1.30], [0.015, 0.02, 0.02]
BETA1 = 0.147; C3 = 0.75 * BETA1

def K_hub(om, k):
    N=len(om); K=np.diag(np.asarray(om,float)**2).copy()
    for j in range(1,N):
        K[0,0]+=k;K[j,j]+=k;K[0,j]-=k;K[j,0]-=k
    return K
def rs(Om,k):
    Zm=K_hub(W,k)-Om**2*np.eye(3)+1j*Om*np.diag([2*z for z in Z])
    inv=1/np.linalg.solve(Zm,np.eye(3)[:,0])[0]; return inv.real,inv.imag
def cf_Fc(Om,k):
    r,s=rs(Om,k)
    if r>=0: return None
    yc=-2*r/(3*C3); F2=yc*((r+C3*yc)**2+s**2)
    return np.sqrt(F2) if F2>0 else None
def start_u(nh,Om,F0):
    u=np.zeros(3*nh*2+1); u[0]=F0/max(1e-6,abs(W[0]**2-Om**2)); u[-1]=F0; return u

def cf_tips(k,lo=1.10,hi=1.29,n=4000):
    oms=np.linspace(lo,hi,n); fv=[rs(o,k)[0]**2-3*rs(o,k)[1]**2 for o in oms]; out=[]
    for i in range(n-1):
        if fv[i]*fv[i+1]<0:
            o=oms[i]-fv[i]*(oms[i+1]-oms[i])/(fv[i+1]-fv[i])
            if rs(o,k)[0]<0: out.append(round(o,4))
    return out

NH,NT=5,384
oms=np.linspace(1.11,1.27,17)
for k in [0.05,0.10,0.14]:
    print(f"\nkappa={k:.3f}  closed-form upper cusp tips: {cf_tips(k)}")
    pat=[]
    for Om in oms:
        Fc=cf_Fc(Om,k); Fmax=(1.6*Fc) if Fc else 1.0
        c=make_continuation(W,Z,BETA1,Om,nh=NH,nt=NT)
        traj=c["continue_F"](K_hub(W,k),start_u(NH,Om,0.008),ds=0.01,n_steps=500,F_max=Fmax)
        nf,_=count_folds(traj); m="##" if nf>=2 else ".."; pat.append(m)
        print(f"  Om={Om:.4f}: folds={nf} {m}")
    print(f"  pattern {oms[0]:.2f}->{oms[-1]:.2f}: {''.join(pat)}")
