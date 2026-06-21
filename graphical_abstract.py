"""Graphical abstract for the CPMP/JTB manuscript.
Centre panel uses REAL model output (PEK->LAX, -15 zones) from comparison.py.
No data are altered; this is a visualisation of the published computation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

from kjf_engine import calibrated_params
from comparison import static_plan, simulate_followed_plan

P = calibrated_params()           # G = 38.7
N = 14
# PEK (UTC+8) -> LAX (UTC-7): raw = -15  (delay, 15 h)
plan = static_plan(tz_o=8, tz_d=-7, t_wake=7.0, t_bed=23.0, n_days=N)
sim  = simulate_followed_plan(plan, P, n_days=N, dt=0.02, arrival_clock=0.0)

base = sim["baseline_cbtmin"]
C_assumed = np.array([d["cbtmin_assumed"] for d in plan["days"][:N]])
C_actual  = sim["C_actual"]

def unwrap_from(base, seq):
    out=[]; prev=base
    for v in seq:
        step=(v-prev+12)%24-12; prev=prev+step; out.append(prev)
    return np.array(out)

asum_uw = unwrap_from(base, C_assumed)
act_uw  = unwrap_from(base, C_actual)
# phase shift ACHIEVED toward destination (delay => later => +); magnitude
shift_static = np.abs(asum_uw - base)
shift_ode    = np.abs(act_uw  - base)
drift = np.abs(act_uw - asum_uw)              # CBTmin estimation error (h)
days = np.arange(N)

print("days_to_adjust(static):", plan["days_to_adjust"], "mode:", plan["mode"])
print("max drift (h):", round(np.nanmax(drift),2))
print("static reaches ~15h by day:", int(np.argmax(shift_static>=14.5)))
print("ode shift @ last day:", round(shift_ode[-1],2), "static:", round(shift_static[-1],2))

# ---- colours (colourblind-safe) ----
C_STATIC = "#999999"   # grey
C_ODE    = "#0072B2"   # blue
C_GAP    = "#E69F00"   # orange (gap/drift highlight)
INK      = "#1a1a1a"

fig = plt.figure(figsize=(12.6, 5.0))     # ~2.5:1, JTB landscape GA
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.45, 1.15], wspace=0.04,
                      left=0.005, right=0.995, top=0.99, bottom=0.01)

# ---------- ZONE 1 : problem ----------
axL = fig.add_subplot(gs[0,0]); axL.axis("off")
axL.set_xlim(0,1); axL.set_ylim(0,1)
axL.add_patch(FancyBboxPatch((0.02,0.03),0.96,0.94, boxstyle="round,pad=0.02,rounding_size=0.03",
              fc="#F4F6F8", ec="#D4DAE0", lw=1.2, transform=axL.transAxes))
axL.text(0.5,0.90,"Jet-lag light rules\nover-promise the clock",
         ha="center",va="top",fontsize=15.5,fontweight="bold",color=INK)
axL.text(0.5,0.585,"Standard heuristics send light to the\n"
                   "correct side of the human PRC, but\n"
                   "assume the body clock re-entrains at a\n"
                   "fixed rate (e.g. 1.0 h·day⁻¹ advance,\n"
                   "1.5 h·day⁻¹ delay).",
         ha="center",va="center",fontsize=10.2,color=INK,linespacing=1.45)
axL.text(0.5,0.20,"Real re-entrainment is asymptotic,\n"
                  "light-history– and intensity-dependent.",
         ha="center",va="center",fontsize=10.2,color="#555",style="italic",linespacing=1.4)

# ---------- ZONE 2 : real model curve ----------
axM = fig.add_subplot(gs[0,1])
axM.axhline(15, ls=(0,(5,4)), lw=1.3, color="#444", zorder=1)
axM.text(13.6,15.25,"target  −15 h", ha="right",va="bottom",fontsize=9.2,color="#444")
# gap shading
axM.fill_between(days, shift_ode, shift_static, color=C_GAP, alpha=0.22, zorder=1,
                 label="mis-timing gap")
axM.plot(days, shift_static, "-s", color=C_STATIC, lw=2.4, ms=6, mfc="white",
         mec=C_STATIC, mew=1.6, label="static rule (assumed)", zorder=3)
axM.plot(days, shift_ode, "-o", color=C_ODE, lw=2.6, ms=6.5, label="dynamical clock (real)", zorder=4)
# annotate drift at its widest
k = int(np.nanargmax(drift))
axM.annotate("", xy=(k, shift_static[k]), xytext=(k, shift_ode[k]),
             arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=2.0))
axM.text(k+0.25, (shift_static[k]+shift_ode[k])/2,
         f"up to {np.nanmax(drift):.1f} h\nclock drift", fontsize=9.6, color="#8a5a00",
         va="center", ha="left", fontweight="bold")
axM.set_xlim(-0.3, 13.3); axM.set_ylim(0, 17.2)
axM.set_xlabel("days after arrival", fontsize=10.5)
axM.set_ylabel("phase shift achieved (h)", fontsize=10.5)
axM.set_title("Beijing → Los Angeles (15 time zones)", fontsize=11.2, fontweight="bold", pad=7)
axM.tick_params(labelsize=9)
axM.legend(loc="lower right", fontsize=8.8, framealpha=0.92, borderpad=0.5)
for s in ("top","right"): axM.spines[s].set_visible(False)
axM.grid(True, axis="y", ls=":", lw=0.6, color="#cfcfcf", alpha=0.7)

# ---------- ZONE 3 : consequence + fix ----------
axR = fig.add_subplot(gs[0,2]); axR.axis("off")
axR.set_xlim(0,1); axR.set_ylim(0,1)
# consequence box
axR.add_patch(FancyBboxPatch((0.02,0.52),0.96,0.45, boxstyle="round,pad=0.02,rounding_size=0.03",
              fc="#FDF3E6", ec="#F0C987", lw=1.3, transform=axR.transAxes))
axR.text(0.5,0.91,"Consequence",ha="center",va="top",fontsize=12.5,fontweight="bold",color="#8a5a00")
axR.text(0.5,0.70,"The assumed clock lags the true one by\n"
                  "2–4 days and drifts up to 3.7 h, so every\n"
                  "phase-locked light and melatonin window\n"
                  "is mis-timed — the narrow melatonin\n"
                  "window worst of all.",
         ha="center",va="center",fontsize=9.8,color=INK,linespacing=1.4)
# fix box
axR.add_patch(FancyBboxPatch((0.02,0.03),0.96,0.43, boxstyle="round,pad=0.02,rounding_size=0.03",
              fc="#E8F1F8", ec="#9CC3E0", lw=1.3, transform=axR.transAxes))
axR.text(0.5,0.40,"Fix",ha="center",va="top",fontsize=12.5,fontweight="bold",color="#0a4f80")
axR.text(0.5,0.205,"A limit-cycle pacemaker model tracks the\n"
                   "true clock from light history and adds\n"
                   "intensity dependence — same sound\n"
                   "direction, realistic timing & expectations.",
         ha="center",va="center",fontsize=9.8,color=INK,linespacing=1.4)

fig.savefig("Graphical_Abstract.pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig("Graphical_Abstract.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
print("saved GA pdf+png")
