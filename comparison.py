"""
comparison.py
=============
Step 3 machinery: (a) a faithful port of CPMP's current STATIC heuristic, and
(b) a routine that simulates the dynamical (KJF) clock of a traveller who
FOLLOWS the static plan, so we can measure where the static prescription puts
light/melatonin on the wrong circadian phase.

Static heuristic (ported verbatim from the prototype / 03-数学模型详解 sec.4-5,
02-时型MEQ sec.4-5):
    raw   = TZ_d - TZ_o
    a     = raw mod 24 ;  d = 24 - a
    CBTmin_assumed_habitual = (t_wake - 2) mod 24
    sleepDur = clamp((t_wake - t_bed) mod 24, 6, 9.5)
    mode  = none (a<0.5 or a>23.5) | advance (a<=8, rate 1.0) | delay (else, rate 1.5)
    b0    = (t_bed + raw) mod 24          # first-night body bedtime in dest clock
    target= t_bed                          # pure jet-lag: return to habitual
    each day: move bedtime toward target by <= rate; wake = bed + sleepDur
    light(advance) = [wake, wake+2.5]   ; melatonin at (bed-1.5)
    light(delay)   = [bed-3, bed-0.5]   ; melatonin none
The heuristic's *implicit model of the clock* is CBTmin_assumed(day) = wake-2,
advancing rigidly at the fixed rate.

The dynamical truth is obtained by integrating the KJF oscillator (entrained to
the origin schedule, then exposed to exactly the light the static plan tells the
traveller to get) and reading the actual CBTmin each day.
"""
from __future__ import annotations
import numpy as np
from math import ceil
from kjf_engine import (integrate, cbtmin_times, settle_to_limit_cycle,
                        single_pulse_prc, wrap_pm12)


def clamp(x, a, b):
    return min(max(x, a), b)


def in_wrap(c, a, b):
    """Is clock c (0..24) inside the [a,b) interval, allowing midnight wrap?"""
    a %= 24; b %= 24; c %= 24
    return (a <= c < b) if a < b else (c >= a or c < b)


# ---------------------------------------------------------------------------
# (a) the static heuristic
# ---------------------------------------------------------------------------
def static_plan(tz_o, tz_d, t_wake, t_bed, n_days=14):
    raw = tz_d - tz_o
    a = raw % 24
    d = 24 - a
    cbtmin_hab = (t_wake - 2) % 24
    sleepDur = clamp((t_wake - t_bed) % 24, 6, 9.5)

    if a < 0.5 or a > 23.5:
        mode, amount, rate = "none", 0.0, 1.0
    elif a <= 8:
        mode, amount, rate = "advance", a, 1.0
    else:
        mode, amount, rate = "delay", d, 1.5

    b0 = (t_bed + raw) % 24
    target = t_bed
    if mode == "advance":
        dist = (b0 - target) % 24
    elif mode == "delay":
        dist = (target - b0) % 24
    else:
        dist = 0.0
    days_to_adjust = max(1, ceil(dist / rate)) if mode != "none" else 0

    days = []
    b = b0
    for n in range(n_days):
        wake = (b + sleepDur) % 24
        cbtmin_assumed = (wake - 2) % 24
        if mode == "advance":
            light_win = (wake % 24, (wake + 2.5) % 24)
            mel = (b - 1.5) % 24
        elif mode == "delay":
            light_win = ((b - 3) % 24, (b - 0.5) % 24)
            mel = None
        else:
            light_win, mel = None, None
        days.append(dict(day=n, bed=b % 24, wake=wake,
                         cbtmin_assumed=cbtmin_assumed,
                         light_win=light_win, melatonin=mel))
        if mode == "advance":
            rem = (b - target) % 24
            b = (b - min(rate, rem)) % 24
        elif mode == "delay":
            rem = (target - b) % 24
            b = (b + min(rate, rem)) % 24
    return dict(mode=mode, amount=amount, rate=rate, raw=raw,
                cbtmin_hab=cbtmin_hab, sleepDur=sleepDur,
                days_to_adjust=days_to_adjust, days=days,
                tz_o=tz_o, tz_d=tz_d, t_wake=t_wake, t_bed=t_bed)


# ---------------------------------------------------------------------------
# (b) dynamical truth: simulate a traveller who follows the static plan
# ---------------------------------------------------------------------------
def followed_plan_light(plan, P, n_days, arrival_clock=0.0,
                        I_sleep=0.0, I_indoor=250.0, I_bright=8000.0,
                        pre_days=30):
    """
    Build the light(t) the traveller actually receives.
    t<0 : pre-arrival, entrained to the ORIGIN schedule expressed in dest clock
          (origin bed/wake shifted by raw), with daytime indoor light.
    t>=0: follows the static plan -- dark asleep, bright in the prescribed light
          window, indoor otherwise.
    arrival_clock = destination clock time at t=0.
    """
    raw = plan["raw"]; t_wake = plan["t_wake"]; t_bed = plan["t_bed"]
    sleepDur = plan["sleepDur"]
    bed_o = (t_bed + raw) % 24
    wake_o = (t_wake + raw) % 24
    days = plan["days"]

    def light(t):
        clk = (arrival_clock + t) % 24
        if t < 0:
            return I_sleep if in_wrap(clk, bed_o, wake_o) else I_indoor
        n = int(t // 24)
        n = min(max(n, 0), n_days - 1)
        dd = days[n]
        if in_wrap(clk, dd["bed"], dd["wake"]):
            return I_sleep
        lw = dd["light_win"]
        if lw is not None and in_wrap(clk, lw[0], lw[1]):
            return I_bright
        return I_indoor
    return light


def _unwrap_clock(clk):
    out = [clk[0]]
    for i in range(1, len(clk)):
        step = (clk[i] - out[-1] + 12) % 24 - 12
        out.append(out[-1] + step)
    return np.array(out)


def simulate_followed_plan(plan, P, n_days=14, dt=0.02, arrival_clock=0.0,
                           pre_days=30, **light_kw):
    """
    Integrate the KJF clock through pre-arrival entrainment + n_days of the plan.
    Returns the actual CBTmin clock time per plan-day (interpolated from the
    CBTmin event sequence, robust to >24 h event spacing during delays).
    """
    light = followed_plan_light(plan, P, n_days, arrival_clock=arrival_clock,
                                pre_days=pre_days, **light_kw)
    y0 = settle_to_limit_cycle(P, dt=dt, n_cycles=10)
    t, Y, _ = integrate(light, -pre_days * 24.0, n_days * 24.0, dt, y0, P)
    cbt = cbtmin_times(t, Y, P)                       # absolute hours
    # use events from the last pre-arrival day onward to anchor day 0
    sel = cbt[cbt >= -24.0]
    day_idx = sel / 24.0
    clk = (arrival_clock + sel) % 24.0
    clk_uw = _unwrap_clock(clk)
    grid = np.arange(n_days, dtype=float)
    C_unwrap = np.interp(grid, day_idx, clk_uw, left=np.nan, right=np.nan)
    C_actual = C_unwrap % 24.0
    cbt_pre = cbt[cbt < 0]
    base = (arrival_clock + cbt_pre[-1]) % 24 if len(cbt_pre) else np.nan
    return dict(C_actual=C_actual, C_unwrap=C_unwrap, baseline_cbtmin=base,
                t=t, Y=Y, cbt_abs=cbt, light=light)


# ---------------------------------------------------------------------------
# metrics: where does the static plan mis-time light?
# ---------------------------------------------------------------------------
def build_prc_interp(P, dur=2.5, I_pulse=8000.0, dt=0.02):
    """Reference PRC (phase relative to CBTmin -> shift, +advance) for the
    prescribed light-window geometry, as an interpolator."""
    grid = np.arange(-12, 12.01, 1.0)
    shifts = single_pulse_prc(P, grid, I_pulse=I_pulse, dur=dur, dt=dt,
                              settle_cycles=12, read_cycle_offset=5)
    def f(phi):
        return float(np.interp(wrap_pm12(phi), grid, shifts))
    return f, grid, shifts


def analyse_scenario(plan, sim, prc_fn, n_days=14):
    """
    Compare static-assumed vs ODE-actual CBTmin and classify each prescribed
    light window. Returns per-day arrays and summary scalars.
    """
    days = plan["days"]; mode = plan["mode"]
    C_assumed = np.array([d["cbtmin_assumed"] for d in days[:n_days]])
    C_actual = sim["C_actual"]
    target = plan["cbtmin_hab"]

    cbt_err = wrap_pm12(C_actual - C_assumed)            # actual lags(+)/leads(-)
    # light-window centre, phase vs actual & assumed CBTmin, and PRC sign
    psi_actual = np.full(n_days, np.nan)
    psi_assumed = np.full(n_days, np.nan)
    prc_actual = np.full(n_days, np.nan)
    wrong_dir = np.zeros(n_days, dtype=bool)
    intended = +1 if mode == "advance" else (-1 if mode == "delay" else 0)
    for n, d in enumerate(days[:n_days]):
        lw = d["light_win"]
        if lw is None or np.isnan(C_actual[n]):
            continue
        # window centre clock (handle wrap)
        a_, b_ = lw
        span = (b_ - a_) % 24
        centre = (a_ + span / 2.0) % 24
        psi_actual[n] = wrap_pm12(centre - C_actual[n])
        psi_assumed[n] = wrap_pm12(centre - C_assumed[n])
        prc_actual[n] = prc_fn(psi_actual[n])
        if intended != 0:
            # wrong direction if the PRC the light actually lands on opposes
            # the intended shift (or is essentially a dead-zone, |shift|<0.1)
            if np.sign(prc_actual[n]) == -intended or abs(prc_actual[n]) < 0.1:
                wrong_dir[n] = True

    # days-to-reentrain (ODE) under the followed plan: |actual - target|<1h
    re_ode = np.nan
    for n in range(n_days):
        if not np.isnan(C_actual[n]) and abs(wrap_pm12(C_actual[n] - target)) < 1.0:
            re_ode = n
            break
    # light mis-timing in hours: prescribed window centre vs PRC-optimal centre
    # PRC-optimal phase (rel. actual CBTmin) for the intended direction:
    if intended == +1:
        phi_opt = _argopt(prc_fn, sign=+1)
    elif intended == -1:
        phi_opt = _argopt(prc_fn, sign=-1)
    else:
        phi_opt = 0.0
    mistime = np.array([abs(wrap_pm12(psi_actual[n] - phi_opt))
                        if not np.isnan(psi_actual[n]) else np.nan
                        for n in range(n_days)])

    return dict(C_assumed=C_assumed, C_actual=C_actual, cbt_err=cbt_err,
                psi_actual=psi_actual, psi_assumed=psi_assumed,
                prc_actual=prc_actual, wrong_dir=wrong_dir, mistime=mistime,
                re_ode=re_ode, phi_opt=phi_opt,
                max_abs_cbt_err=float(np.nanmax(np.abs(cbt_err))),
                n_wrong=int(np.nansum(wrong_dir)),
                mean_mistime=float(np.nanmean(mistime)))


def _argopt(prc_fn, sign=+1, grid=None):
    if grid is None:
        grid = np.arange(-12, 12.01, 0.25)
    vals = np.array([sign * prc_fn(p) for p in grid])
    return float(grid[np.argmax(vals)])
