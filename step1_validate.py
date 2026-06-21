"""
step1_validate.py
=================
Step 1 deliverable: get the KJF engine running and reproduce its BASIC PHASE
BEHAVIOUR. Produces three validation figures and prints quantitative checks.

  Fig 1  Free-running limit cycle in constant darkness (state-space + time series)
  Fig 2  Entrainment to a 24 h light-dark cycle (period locks to 24 h; stable angle)
  Fig 3  Light phase-response curve (type-1; crossover ≈ at x-min, ~0.8-1.1 h before CBTmin) -- the behaviour
         that the static PRC heuristic only approximates.

All outputs are deterministic and reproducible (fixed RK4 step, fixed seeds N/A).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42

from kjf_engine import (KJFParams, integrate, constant_light, ld_cycle,
                        light_pulse, find_minima_times, cbtmin_times,
                        settle_to_limit_cycle, free_running_period,
                        amplitude, wrap_pm12, PI)

plt.rcParams.update({
    "font.size": 10, "axes.linewidth": 0.8,
    "figure.dpi": 130, "savefig.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
INK = "#1c2530"; BLUE = "#2f6db5"; AMBER = "#d98a2b"; RED = "#b8473d"
GREEN = "#3f8a5a"; GREY = "#8893a0"

P = KJFParams()
DT = 0.01

# ===========================================================================
# Numerical checks
# ===========================================================================
print("=" * 64)
print("STEP 1 - KJF ENGINE VALIDATION")
print("=" * 64)
tau, t_dd, Y_dd = free_running_period(P, dt=DT, n_cycles=60, discard=20)
amp = amplitude(P, dt=DT)
print(f"[P] Free-running period (DD)        : {tau:.4f} h")
print(f"[P] Limit-cycle amplitude (x)       : {amp:.4f}")

# ===========================================================================
# FIGURE 1 - limit cycle
# ===========================================================================
y0 = settle_to_limit_cycle(P, dt=DT, n_cycles=40)
t1, Y1, _ = integrate(constant_light(0.0), 0.0, 3 * 24.0, DT, y0, P)
x, xc = Y1[:, 0], Y1[:, 1]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(x, xc, color=BLUE, lw=1.4)
ax[0].plot(x[0], xc[0], "o", color=RED, ms=5)
ax[0].set_xlabel("x  (pacemaker drive)"); ax[0].set_ylabel("x_c  (auxiliary)")
ax[0].set_title("Limit cycle in state space (constant darkness)")
ax[0].set_aspect("equal", adjustable="box")

ax[1].plot(t1, x, color=BLUE, lw=1.4, label="x")
xmins = find_minima_times(t1, x)
for m in xmins:
    ax[1].axvline(m, color=GREY, ls=":", lw=0.8)
    ax[1].axvline(m + P.phi_ref, color=AMBER, ls="--", lw=1.0)
ax[1].plot([], [], color=GREY, ls=":", label="x min")
ax[1].plot([], [], color=AMBER, ls="--", label="CBTmin (=x min + 0.8 h)")
ax[1].set_xlabel("time (h)"); ax[1].set_ylabel("x")
ax[1].set_title(f"Free-run, period = {tau:.3f} h")
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("Figure_1.pdf", bbox_inches="tight")
plt.close(fig)
print("    -> Figure_1.pdf")

# ===========================================================================
# FIGURE 2 - entrainment to a 24 h LD cycle
# ===========================================================================
# Realistic indoor schedule: ~1000 lux for 16 h from 06:00, dark 22:00-06:00.
light = ld_cycle(I_day=1000.0, I_night=0.0, photoperiod=16.0, dawn=6.0)
y0 = (1.0, 1.0, 0.0)
T = 40 * 24.0
t2, Y2, B2 = integrate(light, 0.0, T, DT, y0, P, record_B=True)
cbt = cbtmin_times(t2, Y2, P)
cbt_clock = cbt % 24.0
# entrained period = spacing of CBTmin once locked
locked = cbt[cbt > 25 * 24.0]
ent_period = float(np.mean(np.diff(locked))) if len(locked) > 2 else np.nan
final_phase = cbt_clock[-1]
print(f"[L] Entrained period (24 h LD)      : {ent_period:.4f} h")
print(f"[L] Steady-state CBTmin clock time  : {final_phase:05.2f} h")

fig, ax = plt.subplots(2, 1, figsize=(10, 5.4),
                       gridspec_kw={"height_ratios": [2, 1.1]})
# top: x(t) for first 10 days, dark periods shaded
for d in range(11):
    ax[0].axvspan(d * 24 + 22, d * 24 + 30, color="#dfe5ec", lw=0, zorder=0)
ax[0].plot(t2, Y2[:, 0], color=BLUE, lw=0.8)
ax[0].set_ylabel("x"); ax[0].set_xlim(0, 10 * 24)
ax[0].set_title(
    "Entrainment to a 24 h light-dark cycle (16 h @ 1000 lux, dark 22:00-06:00)")
ax[0].set_xticks(np.arange(0, 10 * 24 + 1, 24))
ax[0].set_xticklabels([str(i) for i in range(11)])
ax[0].set_xlabel("day")
# bottom: CBTmin clock time converging to steady state (UNWRAPPED so the
# transient does not show a spurious 24->0 vertical jump at the midnight branch)
days = cbt / 24.0
def _uw_clock(c):
    o = [c[0]]
    for i in range(1, len(c)):
        o.append(o[-1] + ((c[i] - o[-1] + 12) % 24 - 12))
    return np.array(o)
cbt_uw = _uw_clock(cbt_clock)
ax[1].axhline(cbt_uw[-1], color=GREEN, ls="--", lw=1.0,
              label=f"steady state \u2261 {final_phase:04.1f} h (= wake \u2212 2 h)")
ax[1].plot(days, cbt_uw, "o-", color=AMBER, ms=3.5, lw=0.9)
ax[1].set_xlabel("day"); ax[1].set_ylabel("CBTmin clock\n(unwrapped, h)")
ax[1].set_xlim(0, days.max())
ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("CBTmin converges to a stable phase (\u2248 2 h before 06:00 wake)",
                fontsize=9)
fig.tight_layout(); fig.savefig("Figure_2.pdf", bbox_inches="tight")
plt.close(fig)
print("    -> Figure_2.pdf")

# ===========================================================================
# FIGURE 3 - light phase-response curve (PRC)
# ===========================================================================
def phase_shift_for_pulse(phi_after_cbtmin, I_pulse=9500.0, dur=6.7,
                          I_bg=0.0, settle_days=16, post_days=12):
    """
    Single bright-light pulse on a dim/dark background; measure steady-state
    CBTmin phase shift vs an unperturbed control. phi is hours of the pulse
    CENTRE after CBTmin (negative = before CBTmin).
    Convention: positive shift = phase ADVANCE.
    """
    y0 = settle_to_limit_cycle(P, dt=DT, n_cycles=settle_days)
    # control free-run
    Tc = (post_days + 6) * 24.0
    tc, Yc, _ = integrate(constant_light(I_bg), 0.0, Tc, DT, y0, P)
    cbt_c = cbtmin_times(tc, Yc, P)
    # anchor: first CBTmin after 1.5 days
    t_anchor = cbt_c[cbt_c > 1.5 * 24.0][0]
    t_center = t_anchor + phi_after_cbtmin
    t_start = t_center - dur / 2.0
    # perturbed run
    lp = light_pulse(I_bg, I_pulse, t_start, dur)
    tp, Yp, _ = integrate(lp, 0.0, Tc, DT, y0, P)
    cbt_p = cbtmin_times(tp, Yp, P)
    # compare CBTmin events well after the pulse (steady state)
    tref = t_center + post_days * 24.0
    cc = cbt_c[np.argmin(np.abs(cbt_c - tref))]
    pp = cbt_p[np.argmin(np.abs(cbt_p - tref))]
    return wrap_pm12(cc - pp)

phis = np.arange(-12, 12.01, 1.0)
prc = np.array([phase_shift_for_pulse(ph) for ph in phis])

# locate crossover near CBTmin
sign = np.sign(prc)
crossings = []
for i in range(len(phis) - 1):
    if sign[i] != sign[i + 1] and sign[i] != 0:
        # linear interp zero
        x0 = phis[i] - prc[i] * (phis[i + 1] - phis[i]) / (prc[i + 1] - prc[i])
        crossings.append(x0)
print(f"[L] PRC max advance                 : {prc.max():+.3f} h "
      f"at phi={phis[np.argmax(prc)]:+.0f} h after CBTmin")
print(f"[L] PRC max delay                   : {prc.min():+.3f} h "
      f"at phi={phis[np.argmin(prc)]:+.0f} h after CBTmin")
print(f"[L] PRC zero-crossings (h re CBTmin): "
      f"{', '.join(f'{c:+.2f}' for c in crossings)}")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.axhline(0, color=INK, lw=0.8)
ax.axvline(0, color=AMBER, ls="--", lw=1.2, label="CBTmin")
ax.plot(phis, prc, "o-", color=BLUE, lw=1.6, ms=4)
ax.fill_between(phis, prc, 0, where=(prc > 0), color=GREEN, alpha=0.18)
ax.fill_between(phis, prc, 0, where=(prc < 0), color=RED, alpha=0.18)
ax.set_xlabel("circadian phase of light pulse  (h relative to CBTmin)")
ax.set_ylabel("phase shift (h)\n(+ advance / - delay)")
ax.set_title("Model light PRC: 6.7 h, 9500 lux pulse\n"
             "advances after x-min, delays before -- crossover ≈ at x-min (~0.8-1.1 h before CBTmin)")
ax.annotate("ADVANCE\n(morning light)", (4.5, prc.max() * 0.7),
            color=GREEN, fontsize=8, ha="center")
ax.annotate("DELAY\n(evening light)", (-5.5, prc.min() * 0.7),
            color=RED, fontsize=8, ha="center")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("Figure_3.pdf", bbox_inches="tight")
plt.close(fig)
print("    -> Figure_3.pdf")
print("=" * 64)
print("STEP 1 COMPLETE")
print("=" * 64)
