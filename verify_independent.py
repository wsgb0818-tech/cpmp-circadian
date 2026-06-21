"""
verify_independent.py
=====================
INDEPENDENT replication of the paper's quantitative results, written from the
published equations in the Methods section ALONE. It does NOT import the
project's kjf_engine.py / comparison.py — the model, RK4 integrator, phase
analysis, light PRC and the static-heuristic comparison are all re-derived here
with their own implementation choices (scalar RK4, own event detection, own PRC
read-out). Agreement with the project code therefore corroborates the result
rather than merely reproducing one codebase.

Two parts, one model core:
  Part A — intrinsic + literature validation (paper Results 3.1 & 3.2)
  Part B — static heuristic vs dynamical model across 9 routes (Results 3.3, Table 2)

Run:  python3 verify_independent.py
Deps: numpy only (no matplotlib needed; this is a numeric cross-check).
"""
import numpy as np
from math import ceil
PI = np.pi

# =====================================================================
# MODEL CORE  (paper Table 1; calibrated photic gain G = 38.7)
# =====================================================================
class Par:
    def __init__(self, G=38.7):
        self.mu=0.13; self.taux=24.2; self.k=0.55
        self.alpha0=0.05; self.beta=0.0075; self.p=0.5; self.I0=9500.0
        self.phi_ref=0.8; self.G=G
        self.nat=(24.0/(0.99729*self.taux))**2     # squared natural-freq term

def deriv(x, xc, n, I, P):
    a = P.alpha0*(I/P.I0)**P.p if I>0.0 else 0.0
    Bhat = P.G*(1.0-n)*a
    B = (1.0-0.4*x)*(1.0-0.4*xc)*Bhat
    dx  = (PI/12.0)*(xc+B)
    dxc = (PI/12.0)*(P.mu*(xc-(4.0/3.0)*xc*xc*xc) - x*(P.nat + P.k*B))
    dn  = 60.0*(a*(1.0-n) - P.beta*n)
    return dx, dxc, dn

def integrate(light, t0, t1, dt, y0, P):
    """Scalar fixed-step RK4. Returns (t_grid, x_array, final_state)."""
    N = int(round((t1-t0)/dt)); x, xc, n = y0
    xs = np.empty(N+1); xs[0] = x
    for i in range(N):
        ti = t0 + dt*i
        I1 = light(ti); Im = light(ti+0.5*dt); I2 = light(ti+dt)
        k1x,k1c,k1n = deriv(x,xc,n,I1,P)
        k2x,k2c,k2n = deriv(x+0.5*dt*k1x, xc+0.5*dt*k1c, n+0.5*dt*k1n, Im, P)
        k3x,k3c,k3n = deriv(x+0.5*dt*k2x, xc+0.5*dt*k2c, n+0.5*dt*k2n, Im, P)
        k4x,k4c,k4n = deriv(x+dt*k3x, xc+dt*k3c, n+dt*k3n, I2, P)
        x  += (dt/6.0)*(k1x+2*k2x+2*k3x+k4x)
        xc += (dt/6.0)*(k1c+2*k2c+2*k3c+k4c)
        n  += (dt/6.0)*(k1n+2*k2n+2*k3n+k4n)
        xs[i+1] = x
    return t0 + dt*np.arange(N+1), xs, (x,xc,n)

def xmin_times(t, x, min_sep=16.0):
    ct, cv = [], []
    for i in range(1, len(x)-1):
        if x[i] < x[i-1] and x[i] <= x[i+1]:
            y0,y1,y2 = x[i-1],x[i],x[i+1]; den = y0-2*y1+y2
            d = 0.5*(y0-y2)/den if den != 0 else 0.0
            ct.append(t[i]+d*(t[i+1]-t[i])); cv.append(y1)
    if not ct: return np.array([])
    ct=np.array(ct); cv=np.array(cv); keep=[0]
    for j in range(1,len(ct)):
        if ct[j]-ct[keep[-1]] >= min_sep: keep.append(j)
        elif cv[j] < cv[keep[-1]]: keep[-1]=j
    return ct[keep]

def cbtmin_times(t,x,P): return xmin_times(t,x)+P.phi_ref
def wrap12(d): return (d+12.0)%24.0-12.0
def clamp(v,a,b): return min(max(v,a),b)
def in_wrap(c,a,b):
    a%=24;b%=24;c%=24
    return (a<=c<b) if a<b else (c>=a or c<b)
def settle(P, dt=0.01, nc=18):
    _,_,y = integrate(lambda tt:0.0, 0, nc*24, dt, (1.0,1.0,0.0), P); return y

def prc(P, phis, Ipulse=9500.0, dur=6.7, Ibg=0.0, dt=0.01, read_off=6):
    y0 = settle(P, dt, 18); T = (read_off+5)*24.0
    tc, xc_, _ = integrate(lambda tt:Ibg, 0, T, dt, y0, P)
    cbtc = cbtmin_times(tc, xc_, P); anchor = cbtc[cbtc>1.5*24][0]
    out=[]
    for phi in phis:
        center=anchor+phi; ts=center-dur/2.0
        light = lambda tt, s=ts: Ipulse if (s<=tt<s+dur) else Ibg
        tp, xp_, _ = integrate(light, 0, T, dt, y0, P); cbtp = cbtmin_times(tp, xp_, P)
        tref=center+read_off*24.0
        cc=cbtc[np.argmin(np.abs(cbtc-tref))]; pp=cbtp[np.argmin(np.abs(cbtp-tref))]
        out.append(wrap12(cc-pp))     # + = advance
    return np.array(out)

# =====================================================================
# PART A — intrinsic + literature validation
# =====================================================================
def part_A():
    print("="*66); print("PART A — intrinsic + literature validation (G=38.7)"); print("="*66)
    P = Par(38.7); dt = 0.01
    # period + amplitude in DD
    t, x, _ = integrate(lambda tt:0.0, 0, 60*24, dt, (1.0,1.0,0.0), P)
    xm = xmin_times(t,x); xm = xm[xm>20*24]; period = float(np.mean(np.diff(xm)))
    amp = 0.5*(x[t>50*24].max()-x[t>50*24].min())
    print(f"[3.1] free-running period (DD)   {period:6.3f} h    (paper 24.16)")
    print(f"[3.1] limit-cycle amplitude       {amp:6.3f}      (paper 1.007)")
    t2,x2,_ = integrate(lambda tt:0.0, 0, 60*24, dt, (1.0,1.0,0.0), Par(19.875))
    xm2=xmin_times(t2,x2); xm2=xm2[xm2>20*24]
    print(f"      period at published G=19.875 {np.mean(np.diff(xm2)):6.3f} h   (gain-invariant: B=0 in DD)")
    # LD entrainment
    dawn=6.0
    ld=lambda tt:1000.0 if ((tt-dawn)%24.0)<16.0 else 0.0
    td,xd,_=integrate(ld,0,40*24,0.02,(1.0,1.0,0.0),P)
    cb=cbtmin_times(td,xd,P); cb=cb[cb>30*24]; cbt_clock=float(np.mean(cb%24.0))
    xmd=xmin_times(td,xd); xmd=xmd[xmd>30*24]
    print(f"[3.1] LD 16:8 entrained period   {np.mean(np.diff(xmd)):6.3f} h    (expect 24.00)")
    print(f"[3.1] CBTmin clock {cbt_clock:.2f} -> {wrap12(dawn-cbt_clock):+.2f} h before wake  (paper: wake-2)")
    # PRC
    phis=np.arange(-12,12.0001,1.0); sh=prc(P,phis,dt=0.01)
    cross=None;_b=1e9
    for i in range(len(phis)-1):
        if sh[i]*sh[i+1]<0:
            c=phis[i]-sh[i]*(phis[i+1]-phis[i])/(sh[i+1]-sh[i])
            if abs(c)<_b:_b=abs(c);cross=c
    print(f"[3.2] PRC peak-to-trough          {sh.max()-sh.min():5.2f} h     (paper 5.04)")
    print(f"[3.2] max advance {sh.max():+.2f} h  max delay {sh.min():+.2f} h  (paper +2.2 / -2.8)")
    print(f"[3.2] crossover nearest CBTmin   {cross:+.2f} h    (paper: at CBTmin)")
    # fluence (delay at -5 h, matches the paper's fluence sweep)
    print("[3.2] fluence (|delay| at -5 h re CBTmin):  ", end="")
    print(", ".join(f"{I}lx:{abs(prc(P,[-5.0],Ipulse=float(I),dt=0.01)[0]):.2f}h"
                    for I in [10,100,1000,9500]) + "   (paper 0.35->2.5)")

# =====================================================================
# PART B — static heuristic vs dynamical model (Table 2)
# =====================================================================
def static_plan(tz_o,tz_d,t_wake,t_bed,n_days=14):
    raw=tz_d-tz_o; a=raw%24; d=24-a
    cbtmin_hab=(t_wake-2)%24; sleepDur=clamp((t_wake-t_bed)%24,6,9.5)
    if a<0.5 or a>23.5: mode,amount,rate="none",0.0,1.0
    elif a<=8:          mode,amount,rate="advance",a,1.0
    else:               mode,amount,rate="delay",d,1.5
    b0=(t_bed+raw)%24; target=t_bed
    dist=(b0-target)%24 if mode=="advance" else ((target-b0)%24 if mode=="delay" else 0.0)
    days_to_adjust=max(1,ceil(dist/rate)) if mode!="none" else 0
    days=[]; b=b0
    for n in range(n_days):
        wake=(b+sleepDur)%24; cass=(wake-2)%24
        if mode=="advance": lw=(wake%24,(wake+2.5)%24)
        elif mode=="delay": lw=((b-3)%24,(b-0.5)%24)
        else: lw=None
        days.append(dict(bed=b%24,wake=wake,cbtmin_assumed=cass,light_win=lw))
        if mode=="advance": b=(b-min(rate,(b-target)%24))%24
        elif mode=="delay": b=(b+min(rate,(target-b)%24))%24
    return dict(mode=mode,amount=amount,rate=rate,raw=raw,cbtmin_hab=cbtmin_hab,
                sleepDur=sleepDur,days_to_adjust=days_to_adjust,days=days,
                t_wake=t_wake,t_bed=t_bed)

def _uw(c):
    o=[c[0]]
    for i in range(1,len(c)): o.append(o[-1]+((c[i]-o[-1]+12)%24-12))
    return np.array(o)

def simulate(plan,P,n_days=14,dt=0.02,arrival=0.0,pre_days=20):
    raw=plan["raw"]; bed_o=(plan["t_bed"]+raw)%24; wake_o=(plan["t_wake"]+raw)%24
    days=plan["days"]
    def light(t):
        clk=(arrival+t)%24
        if t<0: return 0.0 if in_wrap(clk,bed_o,wake_o) else 250.0
        nidx=min(max(int(t//24),0),n_days-1); dd=days[nidx]
        if in_wrap(clk,dd["bed"],dd["wake"]): return 0.0
        lw=dd["light_win"]
        if lw is not None and in_wrap(clk,lw[0],lw[1]): return 8000.0
        return 250.0
    y0=settle(P,dt=dt,nc=10)
    t,x,_=integrate(light,-pre_days*24.0,n_days*24.0,dt,y0,P)
    cbt=cbtmin_times(t,x,P); sel=cbt[cbt>=-24.0]
    clk=(arrival+sel)%24.0
    C=np.interp(np.arange(n_days,dtype=float),sel/24.0,_uw(clk),left=np.nan,right=np.nan)%24.0
    return C

def prc_window(P,dur=2.5,Ipulse=8000.0,dt=0.02):
    grid=np.arange(-12,12.0001,1.0); sh=prc(P,grid,Ipulse=Ipulse,dur=dur,Ibg=0.0,dt=dt,read_off=5)
    return lambda phi: float(np.interp(wrap12(phi),grid,sh))

def part_B():
    print("\n"+"="*66); print("PART B — static heuristic vs dynamical model (Table 2)"); print("="*66)
    P=Par(38.7); prc_fn=prc_window(P)
    SCN=[(8,10,7.,23.,"PEK->SYD +2"),(8,4,7.,23.,"PEK->DXB -4"),(8,1,7.,23.,"PEK->LHR -7"),
         (1,8,7.,23.,"LHR->PEK +7"),(8,-4,7.,23.,"PEK->JFK -12"),(8,-7,7.,23.,"PEK->LAX -15"),
         (-7,8,7.,23.,"LAX->PEK +15"),(1,8,6.,22.,"LHR->PEK +7 early"),(1,8,8.5,.5,"LHR->PEK +7 late")]
    PAPER={"PEK->SYD +2":(2,4,1.3),"PEK->DXB -4":(3,5,1.9),"PEK->LHR -7":(5,8,2.7),
           "LHR->PEK +7":(7,10,1.9),"PEK->JFK -12":(8,12,3.5),"PEK->LAX -15":(10,14,3.7),
           "LAX->PEK +15":(6,9,3.2),"LHR->PEK +7 early":(7,10,2.0),"LHR->PEK +7 late":(7,9,1.8)}
    print(f"{'scenario':20}{'mode':8}{'shift':>5}{'stat_d':>7}{'ode_d':>6}{'maxErr':>7}{'wrong':>6}  paper(stat/ode/err)")
    print("-"*92)
    for tz_o,tz_d,w,b,lab in SCN:
        n=16; plan=static_plan(tz_o,tz_d,w,b,n_days=n); C=simulate(plan,P,n_days=n)
        Ca=np.array([d["cbtmin_assumed"] for d in plan["days"][:n]])
        err=wrap12(C-Ca); maxerr=float(np.nanmax(np.abs(err))); target=plan["cbtmin_hab"]
        wrong=0; intended=+1 if plan["mode"]=="advance" else -1
        for di,d in enumerate(plan["days"][:n]):
            lw=d["light_win"]
            if lw is None or np.isnan(C[di]): continue
            span=(lw[1]-lw[0])%24; centre=(lw[0]+span/2.0)%24
            pr=prc_fn(wrap12(centre-C[di]))
            if np.sign(pr)==-intended or abs(pr)<0.1: wrong+=1
        reo=next((k for k in range(n) if not np.isnan(C[k]) and abs(wrap12(C[k]-target))<1.0), None)
        ode_d=(reo+1) if reo is not None else -1
        ps,po,pe=PAPER[lab]
        ok=lambda a,b:"ok" if a==b else f"!{b}"
        print(f"{lab:20}{plan['mode']:8}{plan['amount']:5.0f}{plan['days_to_adjust']:7d}"
              f"{ode_d:6d}{maxerr:7.2f}{wrong:6d}  {ps}/{po}/{pe}"
              f"  [{ok(plan['days_to_adjust'],ps)} {ok(ode_d,po)} err~{pe}]")

if __name__ == "__main__":
    part_A(); part_B()
    print("\n(Independent replication complete. Compare against project step1/2/3 output.)")
