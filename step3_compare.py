"""
step3_compare.py
================
Step 3: systematic comparison of CPMP's STATIC heuristic vs the dynamical KJF
model across a panel of real long-haul routes and chronotypes. Quantifies how
far the static prescription mis-places circadian interventions.

Headline metric: because BOTH light and melatonin windows are anchored to the
static schedule, the error between the heuristic's ASSUMED CBTmin and the ODE's
ACTUAL CBTmin equals the mis-timing (in hours) of every phase-dependent
intervention. We also flag days on which the prescribed BRIGHT-LIGHT window
lands on the wrong side of the true PRC ("wrong-direction light").
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42

from kjf_engine import calibrated_params, wrap_pm12
from comparison import (static_plan, simulate_followed_plan,
                        build_prc_interp, analyse_scenario)

plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.8, "figure.dpi": 130,
                     "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5})
INK="#1c2530"; BLUE="#2f6db5"; AMBER="#d98a2b"; RED="#b8473d"; GREEN="#3f8a5a"; GREY="#8893a0"

P = calibrated_params()
DT = 0.02
NDAYS = 14

# Panel of scenarios: (label, tz_origin, tz_dest, wake, bed)
# standard chronotype = wake 07:00 / bed 23:00 unless noted
SCEN = [
    ("PEK->SYD  +2 E",   8,  10, 7.0, 23.0),
    ("PEK->DXB  -4 W",   8,   4, 7.0, 23.0),
    ("PEK->LHR  -7 W",   8,   1, 7.0, 23.0),
    ("LHR->PEK  +7 E",   1,   8, 7.0, 23.0),
    ("PEK->JFK -12",     8,  -4, 7.0, 23.0),
    ("PEK->LAX -15 (a=9)", 8, -7, 7.0, 23.0),
    ("LAX->PEK +15 (a=15)", -7, 8, 7.0, 23.0),
    ("LHR->PEK +7 EARLY", 1,  8, 6.0, 22.0),
    ("LHR->PEK +7 LATE",  1,  8, 8.5, 0.5),
]

print("=" * 92)
print("STEP 3 - STATIC HEURISTIC vs DYNAMICAL KJF MODEL  (followed-plan comparison)")
print("=" * 92)
print("building reference PRC (once) ...")
prc_fn, prc_grid, prc_shifts = build_prc_interp(P, dt=DT)

rows = []
results = {}
for (label, tzo, tzd, wake, bed) in SCEN:
    plan = static_plan(tzo, tzd, wake, bed, n_days=NDAYS)
    sim = simulate_followed_plan(plan, P, n_days=NDAYS, dt=DT)
    A = analyse_scenario(plan, sim, prc_fn, n_days=NDAYS)
    results[label] = (plan, sim, A)
    re_ode = A["re_ode"]
    re_ode_str = f"{re_ode+1}" if not (re_ode is np.nan or re_ode != re_ode) else ">14"
    rows.append((label, plan["mode"], plan["amount"], plan["days_to_adjust"],
                 re_ode_str, A["max_abs_cbt_err"], A["n_wrong"], A["mean_mistime"]))

# ---- summary table ----
print()
hdr = f'{"scenario":22s}{"mode":8s}{"shift":>6s}{"static_d":>9s}{"ODE_d":>7s}{"maxErr":>8s}{"wrongL":>7s}{"misT":>7s}'
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f'{r[0]:22s}{r[1]:8s}{r[2]:6.1f}{r[3]:9d}{r[4]:>7s}{r[5]:8.2f}{r[6]:7d}{r[7]:7.2f}')
print("\n  static_d = days-to-adjust PREDICTED by heuristic;  ODE_d = days until ODE")
print("  CBTmin within 1 h of target under the followed plan (>14 = not within 2 wk).")
print("  maxErr = max |assumed-actual CBTmin| (h) = worst mis-timing of any")
print("  phase-dependent intervention (light AND melatonin).  wrongL = # days the")
print("  prescribed bright-light window lands on the wrong side of the true PRC.")
print("  misT = mean |prescribed light centre - PRC-optimal centre| (h).")

# ===========================================================================
# FIGURE 5 - CBTmin trajectories (assumed vs actual) for representative cases
# ===========================================================================
def _unwrap(c):
    out = [c[0]]
    for i in range(1, len(c)):
        step = (c[i] - out[-1] + 12) % 24 - 12
        out.append(out[-1] + step)
    return np.array(out)

show = ["LHR->PEK  +7 E", "PEK->LHR  -7 W", "PEK->JFK -12", "PEK->LAX -15 (a=9)"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, lab in zip(axes.ravel(), show):
    plan, sim, A = results[lab]
    d = np.arange(NDAYS)
    s = +1.0 if plan["mode"] == "delay" else -1.0      # direction toward target
    amount = plan["amount"]
    prog_assumed = s * (_unwrap(A["C_assumed"]) - A["C_assumed"][0])
    prog_actual = s * (_unwrap(A["C_actual"]) - A["C_actual"][0])
    ax.axhline(amount, color=GREEN, ls="--", lw=1.0, label=f"target ({amount:.0f} h {plan['mode']})")
    ax.fill_between(d, prog_assumed, prog_actual, color=AMBER, alpha=0.18, lw=0)
    ax.plot(d, prog_assumed, "s-", color=GREY, ms=4, lw=1.3,
            label="static ASSUMES (fixed rate)")
    ax.plot(d, prog_actual, "o-", color=BLUE, ms=4, lw=1.7, label="ODE ACTUAL")
    sd = plan["days_to_adjust"]
    if sd < NDAYS:
        ax.axvline(sd, color=GREY, ls=":", lw=1.2)
        ax.annotate("static says\n'adjusted'", (sd, 0.3), fontsize=7,
                    color=GREY, ha="center", va="bottom")
    # Days-overshoot, consistent with Fig 6 / Table 2. We deliberately do NOT
    # re-annotate an hour figure here: the amber gap is the progress lag, which
    # differs from the absolute CBTmin error by the day-0 baseline; the hour-level
    # error is reported once, in Fig 6, to avoid two numbers for one phenomenon.
    reo = A["re_ode"]
    ode_day = int(reo) + 1 if reo == reo else None        # nan-safe; +1 = day count
    if ode_day is not None and ode_day <= NDAYS:
        ax.annotate(f"ODE reaches target\n+{ode_day - sd} d later",
                    (min(ode_day, NDAYS - 0.5), amount * 0.48),
                    fontsize=7.3, color=RED, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=0.6))
    k = int(np.nanargmax(np.abs(prog_assumed - prog_actual)))
    ax.annotate("interventions\nmis-timed",
                (k, (prog_assumed[k] + prog_actual[k]) / 2),
                fontsize=7, color=AMBER, ha="center", va="center")
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("days after arrival")
    ax.set_ylabel("phase shift accomplished (h)")
    ax.legend(fontsize=7.2, loc="lower right")
fig.suptitle("Re-entrainment progress: the static heuristic assumes a fixed rate (grey, straight); "
             "the dynamical clock lags it (blue)\nThe amber gap = mis-timing of every "
             "phase-locked intervention (light, and especially melatonin)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("Figure_5.pdf", bbox_inches="tight")
plt.close(fig)
print("\n    -> Figure_5.pdf")

# ===========================================================================
# FIGURE 6 - summary across panel
# ===========================================================================
labels = [r[0] for r in rows]
maxerr = [r[5] for r in rows]
stat_d = [r[3] for r in rows]
ode_d = [int(r[4]) if r[4] != ">14" else NDAYS for r in rows]

x = np.arange(len(labels))
fig, ax = plt.subplots(2, 1, figsize=(11, 7.8))
# (top) predicted vs actual days to re-entrain -- the headline overpromise
ax[0].bar(x - 0.2, stat_d, width=0.4, color=GREY, label="static PREDICTS adjusted")
ax[0].bar(x + 0.2, ode_d, width=0.4, color=BLUE, label="ODE actual (followed plan)")
for i in range(len(x)):
    gap = ode_d[i] - stat_d[i]
    if gap > 0:
        ax[0].annotate(f"+{gap}d", (x[i] + 0.2, ode_d[i] + 0.1), fontsize=7.5,
                       color=RED, ha="center")
ax[0].set_ylabel("days to re-entrain (within 1 h)")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
ax[0].set_title("Days to re-entrain: the heuristic's fixed-rate promise vs dynamical reality "
                "(ODE lags by 2-4 days)")
ax[0].legend(fontsize=8, loc="upper left")
# (bottom) max CBTmin estimation error -> melatonin/light mis-timing
bars = ax[1].bar(x, maxerr, width=0.55, color=AMBER)
ax[1].axhline(1.0, color=GREEN, ls="--", lw=1, label="1 h (light tolerance)")
ax[1].axhline(0.5, color=RED, ls=":", lw=1, label="0.5 h (melatonin tolerance)")
for b, v in zip(bars, maxerr):
    ax[1].annotate(f"{v:.1f}", (b.get_x() + b.get_width()/2, v + 0.05),
                   fontsize=7.5, ha="center", color=INK)
ax[1].set_ylabel("max |CBTmin estimation error| (h)")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
ax[1].set_title("How far the open-loop heuristic's clock estimate drifts from truth "
                "(this IS the intervention mis-timing; melatonin's narrow PRC is least forgiving)")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("Figure_6.pdf", bbox_inches="tight")
plt.close(fig)
print("    -> Figure_6.pdf")
print("=" * 92)
print("STEP 3 COMPLETE")
print("=" * 92)
