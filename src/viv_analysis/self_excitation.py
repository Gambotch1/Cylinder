#!/usr/bin/env python3
"""
Self-excitation / basin-of-attraction seed for the coupled VIV loop.
====================================================================

Point 3(B): instead of warm-starting the coupled loop from a CFD window AT
the limit-cycle amplitude, seed it with a small analytic oscillation and let
the loop run FREE (no ongoing forcing). If the noise-trained model has learned
a genuine ATTRACTOR, trajectories converge to the CFD limit-cycle amplitude:
  - seeded BELOW  -> grows up to it
  - seeded ABOVE  -> decays down to it
Both reaching the same A/D at the same frequency = attractor demonstration.

CRUCIAL design point — this is a BASIN test, not a "start-from-zero" test.
The training manifold is a RING; the sigma=0.05 noise thickened it into a
TUBE around the ring. The center of the ring (tiny amplitude) is the HOLE,
which is off-manifold and NOT covered by the tube. So:
  * Primary experiment: seed WITHIN tube reach (0.4-0.7x and 1.3-1.6x the
    limit-cycle A/D). Convergence from both sides = attractor. This is the
    defensible claim.
  * Separate sweep: progressively smaller seeds to find how far the basin
    extends (the smallest seed that still converges). That edge quantifies
    basin size; do NOT read "tiny seed fails" as "no attractor" -- it may
    just be outside the trained tube.

The seed is ONLY the initial condition. After t=0 the loop is free -- so this
is NOT the tuning-fork problem (no sustained drive). The seed's frequency
primes the GRU; convergence to the CFD A/D at the CFD frequency is what the
model must produce on its own.

This module builds the seed; it does not run the loop (call your existing
run_coupled_viv with e_forcing=None). numpy + scipy.
"""

import numpy as np

# a_ref is the PEAK amplitude convention (sqrt(2)*RMS of the steady tail),
# never RMS itself -- see measure_cfd_amplitude. Shared by coupled_inference.py
# and analyze_bridge_cfd_campaign.py so the receipt/CSV metadata can't drift
# from the value actually used in the VdP closure.
#
# The definition has NOT changed; only the label was clarified so it reads
# as "RMS-equivalent peak" rather than an ambiguous "peak/RMS" pairing.
A_REF_CONVENTION = "rms_equivalent_peak_sqrt2_std_tail"
# Older receipts (written before this rename) used this string for the exact
# same quantity -- kept so any code reading back a saved receipt can resolve
# both spellings to the current canonical name.
A_REF_CONVENTION_LEGACY_ALIASES = {"peak_sqrt2_rms_tail"}


def normalize_a_ref_convention(value: str) -> str:
    """Map a legacy a_ref_convention string (from an older receipt) to the
    current canonical name. Unknown/current values pass through unchanged."""
    if value in A_REF_CONVENTION_LEGACY_ALIASES:
        return A_REF_CONVENTION
    return value


def to_model_coords(kin: np.ndarray, nd_inputs: bool, D: float, U: float) -> np.ndarray:
    """Twin of coupled_inference.to_model_coords; kept local to avoid a circular import."""
    if not nd_inputs:
        return kin
    if D <= 0 or U <= 0:
        raise ValueError(f"to_model_coords requires D>0, U>0 (got D={D}, U={U})")
    return kin / np.array([D, U, U * U / D], dtype=np.float32)


def build_seed_history(
    ad_seed,          # seed amplitude as A/D (dimensionless)
    seq_len,
    dt,
    D,
    nd_inputs,
    U,
    fn,               # structural frequency [Hz] -> seed frequency
    x_scaler,
    use_ur_context,
    ur_value,
    ur_stats,
    freq=None,        # seed oscillation freq [Hz]; default fn
    phase=0.0,
):
    """
    Returns (history_scaled, initial_state) shaped like warmup_history's output,
    so it is a DROP-IN replacement for the CFD warm-start in coupled_inference.

    disp/vel/acc are the analytic sinusoid and its exact derivatives, so the
    (disp, vel, acc) triple is self-consistent by construction (verified below).
    """
    f = float(fn if freq is None else freq)
    w = 2.0 * np.pi * f
    amp = float(ad_seed) * float(D)             # physical disp amplitude

    # window at t = [-seq_len*dt ... -dt], initial state at t = 0
    j = np.arange(seq_len)
    t_win = (j - seq_len) * dt
    disp = amp * np.sin(w * t_win + phase)
    vel = amp * w * np.cos(w * t_win + phase)
    acc = -amp * w * w * np.sin(w * t_win + phase)

    # self-consistency guard: acc must be the 2nd derivative of disp.
    # (analytic, so this only catches edits/bugs, but cheap and important --
    #  an inconsistent seed is off-manifold and confounds the whole test.)
    if seq_len > 4:
        d2 = np.gradient(np.gradient(disp, dt), dt)
        interior = slice(2, -2)
        denom = np.max(np.abs(acc[interior])) + 1e-30
        rel = np.max(np.abs(d2[interior] - acc[interior])) / denom
        if rel > 0.02:
            raise ValueError(
                f"seed disp/vel/acc not self-consistent (rel err {rel:.3f}); "
                f"w*dt={w*dt:.4f} may be too large for this dt.")

    kin = np.column_stack([disp, vel, acc]).astype(np.float32)
    kin = to_model_coords(kin, nd_inputs, float(D), float(U))
    kin_scaled = x_scaler.transform(kin)

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (float(ur_value) - float(ur_mean)) / ur_std_safe
        ur_col = np.full((seq_len, 1), ur_scaled, dtype=np.float32)
        history = np.hstack([kin_scaled, ur_col]).astype(np.float32)
    else:
        history = kin_scaled.astype(np.float32)

    initial_state = {
        "h": float(amp * np.sin(phase)),
        "h_dot": float(amp * w * np.cos(phase)),
        "h_ddot": float(-amp * w * w * np.sin(phase)),
    }
    return history, initial_state


def amplitude_envelope(h, D, dt, fn, win_cycles=3.0, step_frac=0.5):
    """Sliding-window A/D envelope (sqrt(2)*RMS/D ~ sinusoid amplitude)."""
    h = np.asarray(h, float)
    win = max(int(win_cycles / fn / dt), 8)
    step = max(int(step_frac * win), 1)
    centers, env = [], []
    for s in range(0, len(h) - win, step):
        seg = h[s:s + win]
        centers.append((s + win / 2) * dt)
        env.append(np.sqrt(2.0) * np.std(seg) / D)
    return np.asarray(centers), np.asarray(env)


def dominant_freq(h, dt, fn, band=(0.3, 3.0)):
    """FFT peak of the steady tail (last 40%), in a band around fn (x fn)."""
    h = np.asarray(h, float)
    tail = h[int(0.6 * len(h)):]
    tail = tail - tail.mean()
    F = np.fft.rfft(tail * np.hanning(len(tail)))
    f = np.fft.rfftfreq(len(tail), dt)
    lo, hi = band[0] * fn, band[1] * fn
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return float("nan")
    return float(f[m][np.argmax(np.abs(F[m]))])


def analyze_run(h, D, dt, fn, ad_cfd_ref, f_cfd_ref=None,
    ad_tol=0.10, stat_tol=0.10, freq_bins_tol=1.0):

    
    tc, env = amplitude_envelope(h, D, dt, fn)
    tail = env[int(0.7 * len(env)):]

    ad_final = float(np.mean(tail))
    stationary = bool(
        np.std(tail) / (abs(ad_final) + 1e-30) < stat_tol
    )

    fdom = dominant_freq(h, dt, fn)

    ad_match = bool(
        abs(ad_final - ad_cfd_ref) /
        (abs(ad_cfd_ref) + 1e-30)
        <= ad_tol
    )

    f_ref = float(fn if f_cfd_ref is None else f_cfd_ref)

    fft_tail_n = len(np.asarray(h)[int(0.6 * len(h)):])
    fft_bin = 1.0 / (fft_tail_n * dt)

    f_match = bool(
        np.isfinite(fdom)
        and abs(fdom - f_ref) <= freq_bins_tol * fft_bin + 1e-12
    )

    return {
        "ad_final": ad_final,
        "ad_cfd_ref": float(ad_cfd_ref),
        "ad_match": ad_match,
        "stationary": stationary,
        "f_dominant": float(fdom),
        "f_cfd_ref": f_ref,
        "fft_bin": float(fft_bin),
        "f_error": float(abs(fdom - f_ref)),
        "f_match": f_match,
        "converged": bool(ad_match and stationary and f_match),
        "env_t": tc,
        "env_ad": env,
    }

def find_contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True-run (start_idx, end_idx) pairs in a boolean mask,
    end_idx exclusive. Used to audit whether a growth-rate mask is one
    connected interval or several disconnected ones (see measure_growth_rate
    and contiguous_first_growth_fit)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return [(int(idx[s]), int(idx[e]) + 1) for s, e in zip(starts, ends)]


def measure_growth_rate(h, dt, fn, D, floor_frac=0.05, hi_frac=0.45):
    """
    Fits an exponential growth rate (lambda) to the initial transient of the CFD limit cycle.
    Uses a log-linear regression on the amplitude envelope.

    PRODUCTION ESTIMATOR -- lam/r2/n/t0/t1 selection logic is unchanged.
    The mask is NOT required to be contiguous: every envelope sample in
    (floor_frac*max_env, hi_frac*max_env) is fit in one global regression,
    which can span several disconnected segments if the envelope re-enters
    the band after an early excursion. mask/tc/env/b are additionally
    returned (audit-only) so callers can inspect that without duplicating
    this masking logic -- see analyze_bridge_cfd_campaign.py's
    growth_mask_n_segments/_largest_segment_cycles/_time_span columns and
    contiguous_first_growth_fit (a diagnostic-only alternative estimator).
    """
    # amplitude_envelope is already defined earlier in your self_excitation.py
    tc, env = amplitude_envelope(h, D, dt, fn)
    max_env = np.max(env) if len(env) else 0.0

    # Isolate the clean exponential growth region (escaping the noise floor, before saturation bends)
    mask = (env > floor_frac * max_env) & (env < hi_frac * max_env)

    if not np.any(mask):
        return dict(lam=0.0, r2=0.0, n=0, t0=0.0, t1=0.0, b=0.0,
                     mask=mask, tc=tc, env=env, max_env=float(max_env))

    t_fit = tc[mask]
    y_fit = np.log(env[mask])

    # Linear regression: y = lam * t + b
    A = np.vstack([t_fit, np.ones(len(t_fit))]).T
    lam, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    # Calculate R^2 for the honesty check
    y_pred = lam * t_fit + b
    ss_res = np.sum((y_fit - y_pred)**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-30))

    return dict(lam=float(lam), r2=float(r2), n=len(t_fit), t0=float(t_fit[0]), t1=float(t_fit[-1]),
                 b=float(b), mask=mask, tc=tc, env=env, max_env=float(max_env))


def contiguous_first_growth_fit(h, dt, fn, D, floor_frac=0.05, hi_frac=0.45,
                                  min_cycles=5.0, r2_min=0.80, decay_run_len=3):
    """
    DIAGNOSTIC-ONLY alternative to measure_growth_rate.

    measure_growth_rate fits every sample across the whole floor_frac..hi_frac
    band in one global regression, which the Phase-A audit found is often
    SEVERAL disconnected segments (see growth_mask_n_segments) rather than one
    clean growth interval -- e.g. small post-saturation fluctuations can
    re-enter the band well after the real transient ended, dragging r2 down
    and contaminating lam with unrelated samples.

    This instead: (1) takes only the FIRST contiguous run of the same mask,
    (2) stops that run early if `decay_run_len` consecutive non-increasing
    envelope points appear inside it (a sustained decay/saturation onset),
    then (3) fits log-linear regression on that sub-interval only, and
    (4) applies the SAME minimum-cycle / R2 checks as the production
    eligibility screen (min_cycles, r2_min -- pass the same threshold values
    used elsewhere; this function does not define its own).

    Same masking band as measure_growth_rate (do not silently redefine the
    lock-in band). NOT called by measure_mu_from_cfd or coupled_inference.py
    -- reporting only, per project instruction. Never used to auto-select a
    "better" fit; both legacy and alternative metrics are recorded side by
    side for human review.
    """
    tc, env = amplitude_envelope(h, D, dt, fn)
    max_env = np.max(env) if len(env) else 0.0
    mask = (env > floor_frac * max_env) & (env < hi_frac * max_env)
    segments = find_contiguous_segments(mask)

    if not segments:
        return dict(lam=0.0, r2=0.0, n=0, t0=0.0, t1=0.0, b=0.0, valid=False,
                     n_segments_considered=0, stopped_reason="no_mask_samples",
                     mask=mask, tc=tc, env=env)

    s, e = segments[0]
    seg_env = env[s:e]
    stop_len = len(seg_env)
    if decay_run_len > 0 and len(seg_env) > decay_run_len:
        non_increasing = np.diff(seg_env) <= 0.0
        run = 0
        for i, ni in enumerate(non_increasing):
            run = run + 1 if ni else 0
            if run >= decay_run_len:
                stop_len = i - decay_run_len + 2  # index just before the sustained decay began
                break
    e_eff = s + max(int(stop_len), 1)

    t_fit = tc[s:e_eff]
    y_fit = np.log(env[s:e_eff])
    if len(t_fit) < 2:
        return dict(lam=0.0, r2=0.0, n=len(t_fit),
                     t0=float(t_fit[0]) if len(t_fit) else 0.0,
                     t1=float(t_fit[-1]) if len(t_fit) else 0.0,
                     b=0.0, valid=False, n_segments_considered=len(segments),
                     stopped_reason="segment_too_short", mask=mask, tc=tc, env=env)

    A = np.vstack([t_fit, np.ones(len(t_fit))]).T
    lam, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    y_pred = lam * t_fit + b
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-30))

    cycles = (t_fit[-1] - t_fit[0]) * fn
    valid = bool(lam > 0 and cycles >= min_cycles and r2 >= r2_min)

    return dict(lam=float(lam), r2=float(r2), n=len(t_fit), t0=float(t_fit[0]), t1=float(t_fit[-1]),
                 b=float(b), valid=valid, n_segments_considered=len(segments),
                 stopped_reason=("ok" if valid else "below_thresholds"),
                 mask=mask, tc=tc, env=env)

def measure_cfd_amplitude(case_df, handoff_idx, D):
    """Saturated CFD amplitude. Returns BOTH conventions - do not confuse them.
       a_ref for the VdP MUST be PEAK (matches a_env = sqrt(h^2+(v/w)^2), a peak
       measure), i.e. sqrt(2)*RMS. The harness reports RMS/D. Mixing them puts
       a_ref low by 1.41x and the cycle saturates ~30% under CFD."""
    ordered = case_df.sort_values("time").reset_index(drop=True)
    h = ordered["disp"].to_numpy(float)[handoff_idx:]
    tail = h[int(0.6 * len(h)):]
    rms = float(np.std(tail))                 # RMS displacement [m]
    return dict(rms_AD=rms / D,               # RMS A/D  (matches harness "rms")
                a_ref_peak=float(np.sqrt(2.0) * rms))   # PEAK amplitude [m] -> a_ref

def measure_mu_from_cfd(case_df, t_release, m, c, D, dt, fn, a_ref, qD):
    """Derive the VdP negative-damping strength mu from the CFD growth transient.
       mu sets the GROWTH RATE (a prediction); a_ref sets the amplitude (prescribed)."""
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(float)
    rel = int(np.searchsorted(times, t_release))
    h = ordered["disp"].to_numpy(float)[rel:]
    g = measure_growth_rate(h, dt, fn, D)     # the tested function above
    omega_n = 2.0 * np.pi * fn
    beta_true = c + 2.0 * m * g["lam"]        # aero negative-damping coeff [N.s/m]
    mu = beta_true * omega_n * a_ref / qD     # dimensionless CL-scale
    return dict(mu=float(mu), beta_true=float(beta_true), **g)


def _selftest():
    fs = 1 / 0.005
    dt = 0.005
    fn = 0.2
    w = 2 * np.pi * fn
    D = 0.4
    n = 40000
    t = np.arange(n) * dt
    A_lim = 0.05 * D
    tau = 30.0

    grow = A_lim * (1 - np.exp(-t / tau)) * np.sin(w * t)
    A_hi = 0.10 * D
    decay = (A_lim + (A_hi - A_lim) * np.exp(-t / tau)) * np.sin(w * t)

    rg = analyze_run(grow, D, dt, fn, ad_cfd_ref=0.05)
    rd = analyze_run(decay, D, dt, fn, ad_cfd_ref=0.05)
    print("self-test")
    print(f"  grow : A/D_final={rg['ad_final']:.4f}  fdom={rg['f_dominant']:.3f}  "
          f"converged={rg['converged']}")
    print(f"  decay: A/D_final={rd['ad_final']:.4f}  fdom={rd['f_dominant']:.3f}  "
          f"converged={rd['converged']}")

    class _S:
        mean_ = np.zeros(3); scale_ = np.array([0.02, 0.05, 0.1])
        def transform(self, x): return (x - self.mean_) / self.scale_
    hist, st = build_seed_history(0.5 * 0.05, 1000, dt, D, False, 1.0, fn, _S(),
                                  use_ur_context=True, ur_value=6.0,
                                  ur_stats=(6.0, 1.0))
    print(f"  seed history shape={hist.shape}  init h_dot={st['h_dot']:.5f}")
    assert rg["converged"] and rd["converged"]
    assert hist.shape == (1000, 4)
    print("  PASS")


if __name__ == "__main__":
    _selftest()