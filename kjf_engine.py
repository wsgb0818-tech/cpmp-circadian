"""
kjf_engine.py
=============
Faithful implementation of the Kronauer-Jewett-Forger (KJF / FJK) limit-cycle
oscillator model of the human circadian pacemaker, with an explicit Runge-Kutta
(RK4) integrator and the phase-analysis tools needed to reproduce its basic
phase behaviour.

This is Step 1 of the CPMP "upgrade route #1": replace the static phase-response
heuristic with a dynamical model that integrates a continuous light history.

MODEL
-----
The version implemented here is the *cubic* two-state van der Pol oscillator
(Process P) driven through a one-state photoreceptor model (Process L). This is
the model of:

    Forger DB, Jewett ME, Kronauer RE. A simpler model of the human circadian
    pacemaker. J Biol Rhythms. 1999;14(6):532-537.

which is the companion to (and the field-standard simplification of):

    Jewett ME, Forger DB, Kronauer RE. Revised limit cycle oscillator model of
    the human circadian pacemaker. J Biol Rhythms. 1999;14(6):493-499.

The cubic form is the one carried forward by essentially all subsequent applied
work, including (in our target journal) St Hilaire MA et al., J Theor Biol.
2007;247(4):583-599, and Serkh & Forger, PLoS Comput Biol. 2014.

EQUATIONS  (time t in hours; light intensity I(t) in lux)
---------------------------------------------------------
Process L (phototransduction):
    alpha(I) = alpha0 * (I / I0) ** p
    dn/dt    = 60 * ( alpha(I) * (1 - n) - beta * n )
    Bhat     = G * (1 - n) * alpha(I)

Circadian modulation of photic sensitivity:
    B        = (1 - 0.4 * x) * (1 - 0.4 * xc) * Bhat

Process P (pacemaker):
    dx/dt    = (pi/12) * ( xc + B )
    dxc/dt   = (pi/12) * ( mu*(xc - (4/3)*xc**3)
                           - x*( (24/(0.99729*taux))**2 + k*B ) )

State variables
    x   : pacemaker drive (CBTmin occurs PHI_REF = 0.8 h after x reaches its min)
    xc  : complementary (auxiliary) variable
    n   : fraction of "activated" photic elements in Process L  (0..1)

PARAMETER PROVENANCE
--------------------
mu     = 0.13     oscillator stiffness          (Forger et al. 1999 cubic model)
taux   = 24.2 h   intrinsic period              (Kronauer et al. 1999; Czeisler 1999: 24.18 h)
k      = 0.55     light->period coupling (Aschoff's rule)            (Kronauer 1999)
G      = 19.875   gain of Process L drive        (Kronauer, Forger & Jewett 1999, text)
alpha0 = 0.05     forward-rate scale of Process L                    (Kronauer 1999)
beta   = 0.0075   recovery rate of Process L                         (Kronauer 1999)
p      = 0.5      compressive exponent on light                      (Kronauer 1999)
I0     = 9500 lux half-saturation-like light scale                   (Kronauer 1999)
PHI_REF= 0.8 h    CBTmin lag behind x_min        (Kronauer, Forger & Jewett 1999, text)

NOTE ON CONVENTIONS (to be locked against the original PDFs before submission):
  * Some reproductions use (mu=0.23, G=33.75); that is a rescaling of the same
    cubic model. We use the original (mu=0.13, G=19.875) pairing, for which we
    have direct textual confirmation of G and phi_ref from Kronauer et al. 1999.
  * The light response alpha(I) is the plain power law alpha0*(I/I0)^p. A
    low-light saturation variant alpha0*(I/I0)^p * I/(I+I1) appears in some
    code; at the high intensities used for PRC validation it changes alpha by
    <10% and does not affect the qualitative phase behaviour. It is available
    via the `saturate` flag for sensitivity checks.

All numbers are validated downstream by (a) the free-running period in constant
darkness and (b) the shape/crossover of the light phase-response curve.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

PI = np.pi


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class KJFParams:
    mu: float = 0.13
    taux: float = 24.2
    k: float = 0.55
    G: float = 19.875
    alpha0: float = 0.05
    beta: float = 0.0075
    p: float = 0.5
    I0: float = 9500.0
    phi_ref: float = 0.8          # CBTmin = t(x_min) + phi_ref   [hours]
    saturate: bool = False        # add I/(I+I1) low-light saturation to alpha
    I1: float = 100.0


def calibrated_params(**overrides) -> "KJFParams":
    """
    KJF parameters with the photic gain G calibrated so the model's single
    6.7 h / 9500 lux pulse PRC matches the peak-to-trough amplitude of the
    canonical human light PRC of Khalsa et al. 2003 (J Physiol 549:945-952),
    which is 5.02 h.

    G is the ONLY parameter changed from the published values:
        G = 38.7   (published reference value is 19.875)
    With G=38.7 the model gives peak-to-trough = 5.04 h (0.5 h grid; 5.00 h on a coarser 1 h grid), max advance +2.2 h,
    max delay -2.8 h. All other constants remain at their published values.
    The free-running period (24.16 h) is independent of G, and -- importantly
    for the jet-lag comparison -- the PRC crossover is NOT locked at CBTmin and is
    NOT gain-independent (it sits ~0.4-1.1 h before CBTmin, approximately at the
    x-minimum, and drifts toward CBTmin as G rises; see sensitivity_gain.py /
    Table S1). Timing conclusions are robust across the tested gains per that
    sensitivity analysis, not by invariance.

    The single-pulse magnitude underprediction at the published G is a known
    property of the 1999 cubic model, which was calibrated primarily to the
    three-cycle resetting protocol; later refinements (St Hilaire et al. 2007)
    further sharpen the delay-dominant asymmetry of the human PRC.
    """
    p = KJFParams(G=38.7)
    for key, val in overrides.items():
        setattr(p, key, val)
    return p


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------
def alpha_of_I(I: float, P: KJFParams) -> float:
    """Process-L forward rate as a function of instantaneous light (lux)."""
    if I <= 0.0:
        return 0.0
    a = P.alpha0 * (I / P.I0) ** P.p
    if P.saturate:
        a *= I / (I + P.I1)
    return a


def rhs(state, I, P: KJFParams):
    """
    Time-derivative of [x, xc, n] given instantaneous light I (lux).

    Returns
    -------
    (dx, dxc, dn) : tuple of floats
    B             : the photic drive actually felt by the oscillator (for diag.)
    """
    x, xc, n = state
    a = alpha_of_I(I, P)

    Bhat = P.G * (1.0 - n) * a
    B = (1.0 - 0.4 * x) * (1.0 - 0.4 * xc) * Bhat

    dx = (PI / 12.0) * (xc + B)
    nat = (24.0 / (0.99729 * P.taux)) ** 2          # squared natural frequency term
    dxc = (PI / 12.0) * (P.mu * (xc - (4.0 / 3.0) * xc ** 3)
                         - x * (nat + P.k * B))
    dn = 60.0 * (a * (1.0 - n) - P.beta * n)
    return (dx, dxc, dn), B


# ---------------------------------------------------------------------------
# RK4 integrator (fixed step), with a time-varying light input
# ---------------------------------------------------------------------------
def integrate(light_fn, t0, t1, dt, y0, P: KJFParams, record_B=False):
    """
    Classic 4th-order Runge-Kutta with fixed step `dt` (hours).

    Parameters
    ----------
    light_fn : callable t(hours) -> I(lux)
    t0, t1   : start / end time (hours)
    dt       : step (hours), e.g. 0.02 (=72 s)
    y0       : initial [x, xc, n]
    P        : KJFParams

    Returns
    -------
    t   : (N,) time grid
    Y   : (N,3) states
    B   : (N,) photic drive felt by oscillator (only if record_B else zeros)
    """
    n_steps = int(round((t1 - t0) / dt))
    t = t0 + dt * np.arange(n_steps + 1)
    Y = np.empty((n_steps + 1, 3), dtype=float)
    B = np.zeros(n_steps + 1, dtype=float)
    Y[0] = y0
    if record_B:
        (_, _, _), b0 = rhs(y0, light_fn(t0), P)
        B[0] = b0

    y = np.array(y0, dtype=float)
    for i in range(n_steps):
        ti = t[i]
        Ii = light_fn(ti)
        Imid = light_fn(ti + 0.5 * dt)
        Iend = light_fn(ti + dt)

        d1, _ = rhs(y, Ii, P)
        k1 = np.array(d1)
        d2, _ = rhs(y + 0.5 * dt * k1, Imid, P)
        k2 = np.array(d2)
        d3, _ = rhs(y + 0.5 * dt * k2, Imid, P)
        k3 = np.array(d3)
        d4, _ = rhs(y + dt * k3, Iend, P)
        k4 = np.array(d4)

        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = y
        if record_B:
            (_, _, _), b = rhs(y, Iend, P)
            B[i + 1] = b
    return t, Y, B


# ---------------------------------------------------------------------------
# Light schedules
# ---------------------------------------------------------------------------
def constant_light(I):
    """Constant illuminance (use I=0 for constant darkness, DD)."""
    return lambda t: float(I)


def ld_cycle(I_day, I_night, photoperiod=16.0, period=24.0, dawn=6.0):
    """
    Square-wave light-dark cycle.
    Lights ON (I_day) for `photoperiod` hours starting at clock time `dawn`,
    OFF (I_night) otherwise. `t` is absolute hours; clock = t mod period.
    """
    def fn(t):
        ph = (t - dawn) % period
        return float(I_day) if ph < photoperiod else float(I_night)
    return fn


def light_pulse(I_bg, I_pulse, t_start, duration):
    """Background I_bg everywhere, replaced by I_pulse on [t_start, t_start+duration]."""
    def fn(t):
        if t_start <= t < t_start + duration:
            return float(I_pulse)
        return float(I_bg)
    return fn


def sleep_wake_light(wake_clock, sleep_clock, I_wake, I_sleep=0.0,
                     period=24.0, tz_shift=0.0):
    """
    Self-selected light: bright-ish during the wake episode, dark during sleep.
    wake_clock / sleep_clock are LOCAL clock hours of the destination.
    tz_shift lets the *imposed* schedule sit at a different absolute time
    (used to model arrival in a new time zone).

    The wake interval is [wake_clock, sleep_clock) modulo 24 (handles overnight).
    """
    def fn(t):
        clk = (t - tz_shift) % period
        w, s = wake_clock % period, sleep_clock % period
        if w < s:
            awake = (w <= clk < s)
        else:  # wake interval wraps past midnight
            awake = (clk >= w) or (clk < s)
        return float(I_wake) if awake else float(I_sleep)
    return fn


# ---------------------------------------------------------------------------
# Phase analysis
# ---------------------------------------------------------------------------
def find_minima_times(t, x, min_sep=16.0):
    """
    Times at which x attains local minima, via parabolic interpolation of each
    interior sample below both neighbours. Returns array of times (h).

    `min_sep` (hours) enforces a refractory spacing: circadian x-minima are ~24 h
    apart, so any two detected minima closer than `min_sep` are de-duplicated,
    keeping the deeper one. This removes spurious half-cycle/shoulder minima that
    can appear during large entrainment transients.
    """
    cand_t, cand_v = [], []
    for i in range(1, len(x) - 1):
        if x[i] < x[i - 1] and x[i] <= x[i + 1]:
            y0, y1, y2 = x[i - 1], x[i], x[i + 1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            cand_t.append(t[i] + delta * (t[i + 1] - t[i]))
            cand_v.append(y1)
    if not cand_t:
        return np.array([])
    cand_t = np.array(cand_t); cand_v = np.array(cand_v)
    keep = [0]
    for j in range(1, len(cand_t)):
        if cand_t[j] - cand_t[keep[-1]] >= min_sep:
            keep.append(j)
        elif cand_v[j] < cand_v[keep[-1]]:   # closer & deeper -> replace
            keep[-1] = j
    return cand_t[keep]


def cbtmin_times(t, Y, P: KJFParams):
    """CBTmin event times = x-minimum times + phi_ref."""
    return find_minima_times(t, Y[:, 0]) + P.phi_ref


def free_running_period(P: KJFParams, dt=0.01, n_cycles=60, discard=20,
                        y0=(1.0, 1.0, 0.0)):
    """
    Free-running period (hours) in constant darkness (DD).
    Integrate for n_cycles*~24h, discard the first `discard` cycles as transient,
    then average successive x-minimum intervals.
    """
    T = n_cycles * 24.0
    t, Y, _ = integrate(constant_light(0.0), 0.0, T, dt, y0, P)
    xmins = find_minima_times(t, Y[:, 0])
    xmins = xmins[xmins > discard * 24.0]
    if len(xmins) < 3:
        return np.nan, t, Y
    periods = np.diff(xmins)
    return float(np.mean(periods)), t, Y


def settle_to_limit_cycle(P: KJFParams, dt=0.01, n_cycles=40,
                          y0=(1.0, 1.0, 0.0)):
    """Integrate in DD to relax onto the limit cycle; return final state."""
    T = n_cycles * 24.0
    t, Y, _ = integrate(constant_light(0.0), 0.0, T, dt, y0, P)
    return Y[-1].copy()


def amplitude(P: KJFParams, dt=0.01):
    """Limit-cycle amplitude of x in DD (half peak-to-peak of the last cycle)."""
    T = 60 * 24.0
    t, Y, _ = integrate(constant_light(0.0), 0.0, T, dt, (1.0, 1.0, 0.0), P)
    mask = t > 50 * 24.0
    x = Y[mask, 0]
    return 0.5 * (x.max() - x.min())


def wrap_pm12(dphi):
    """Wrap a phase difference into (-12, 12] hours."""
    return (dphi + 12.0) % 24.0 - 12.0


def single_pulse_prc(P, phis, I_pulse=9500.0, dur=6.7, I_bg=0.0,
                     dt=0.01, settle_cycles=18, read_cycle_offset=6):
    """
    Phase-response curve to a single light pulse on a (dim/dark) background.

    Parameters
    ----------
    phis : iterable of pulse-centre phases, in hours relative to CBTmin
           (negative = before CBTmin / biological evening).
    I_pulse, dur : pulse intensity (lux) and duration (h).
    I_bg : background illuminance during the rest of the run (lux).
    read_cycle_offset : how many ~24 h cycles after the pulse to read the
           steady-state phase (shift is stable from ~1 cycle on).

    Returns
    -------
    shifts : np.ndarray, same length as phis. Positive = phase ADVANCE.
    """
    phis = np.asarray(phis, dtype=float)
    shifts = np.empty_like(phis)
    for i, phi in enumerate(phis):
        y0 = settle_to_limit_cycle(P, dt=dt, n_cycles=settle_cycles)
        T = (read_cycle_offset + 5) * 24.0
        tc, Yc, _ = integrate(constant_light(I_bg), 0.0, T, dt, y0, P)
        cbt_c = cbtmin_times(tc, Yc, P)
        t_anchor = cbt_c[cbt_c > 1.5 * 24.0][0]
        t_center = t_anchor + phi
        t_start = t_center - dur / 2.0
        lp = light_pulse(I_bg, I_pulse, t_start, dur)
        tp, Yp, _ = integrate(lp, 0.0, T, dt, y0, P)
        cbt_p = cbtmin_times(tp, Yp, P)
        tref = t_center + read_cycle_offset * 24.0
        cc = cbt_c[np.argmin(np.abs(cbt_c - tref))]
        pp = cbt_p[np.argmin(np.abs(cbt_p - tref))]
        shifts[i] = wrap_pm12(cc - pp)
    return shifts
