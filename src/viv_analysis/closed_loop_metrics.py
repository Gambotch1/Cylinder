#!/usr/bin/env python3
"""
Closed-loop coupled-inference evaluation metrics (thesis Sec 2.4.2/2.4.4).

Steady-state A/D (peak-to-peak over a time window) is the only closed-loop
metric evaluate_all.py reports. This module adds the four the examiner
actually wants, computed from data already present in coupled_inference.py's
saved .npz files (t, h, cl for the surrogate; h_cfd, cl_cfd for CFD, added
alongside this module):

  (a) Energy error   eps_E = (E_f_hat - E_f) / |E_f|,  E_f = oint C_L dh
      Cycle-based: for each full cycle (successive same-direction
      zero-crossings of h_dot), E_f_cycle = trapz(C_L * h_dot, t); average
      over cycles in the analysis window. This is the projection of C_L
      onto velocity -- the component that sets growth/decay -- not an
      RMSE, which weights every direction equally.

  (b) Phase error   phi_surrogate - phi_CFD
      From E_f = pi * A * F1 * sin(phi):  sin(phi) = E_f / (pi * A * F1),
      A from (c), F1 = |C_L| at the response frequency via a single-bin
      DFT over the same window. No Hilbert transform needed.

  (c) LCO amplitude and frequency error per case
      Cycle-based half-range (max-min)/2 per cycle, averaged -- NOT
      mean(|h|), which is a materially different (and smaller) number for
      a non-sinusoidal limit cycle. Frequency = 1/(mean cycle period).

  (d) Stability outcome: stationary_lco / decay_to_rest / divergence
      Exponential fit (via the Hilbert-envelope log-slope) over the last
      window_frac of the trajectory; classified by the sign and relative
      magnitude of the fitted growth rate. This is the actual pass/fail
      RMSE cannot express.

All four share one `window_frac` (default 0.5, i.e. the last half of the
trajectory) so the analysis window is identical across all four metrics
and between surrogate and CFD -- required for (a)/(b) to be a meaningful
comparison ("computed over the same cycles for surrogate and CFD").
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


# ── Cycle detection ──────────────────────────────────────────────────────

def _same_direction_crossings(x: np.ndarray) -> np.ndarray:
    """Indices of UPWARD zero-crossings of x (negative -> non-negative),
    i.e. one full period apart for a periodic-ish signal. Used on h_dot to
    mark cycle boundaries (h_dot=0 upward = trough of h)."""
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    return np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0] + 1


_FLAT_SIGNAL_REL_TOL = 1e-5
DEFAULT_A_STAR_NOISE_FLOOR = 1e-4  # min (peak-to-peak/2)/D to trust cycle detection


def cycles_from_displacement(t: np.ndarray, h: np.ndarray, D: float | None = None,
                             a_star_floor: float = DEFAULT_A_STAR_NOISE_FLOOR
                             ) -> list[tuple[int, int]]:
    """Full-cycle (start_idx, end_idx) index pairs from upward zero-crossings
    of dh/dt (central-difference velocity) within the given (t, h) window.

    Guards against a signal with no real oscillation to find cycles in:
    np.gradient of pure noise crosses zero constantly, producing spurious
    sub-timestep "cycles" and nonsense frequencies (seen for real: 58 Hz on
    a bridge response settled to a static equilibrium, and separately
    12-40 Hz on responses that had decayed to a large DC offset (h~0.08m)
    with only ~1e-4 m of residual numerical wobble on top -- comparing
    range to h's OWN max there is misleading, since the huge DC offset
    dilutes a self-relative threshold; the physically meaningful comparison
    is against D). Two guards, applied in order:
      1. If D is given: (peak-to-peak/2)/D below a_star_floor -> no cycles.
      2. Always: peak-to-peak range below _FLAT_SIGNAL_REL_TOL of h's own
         magnitude (catches pure float noise even when D is unavailable/0).
    """
    if len(t) < 2:
        return []
    h_range = float(h.max() - h.min())
    if D is not None and D > 0 and (h_range / 2.0) / D < a_star_floor:
        return []
    h_scale = float(np.abs(h).max()) + 1e-30
    if h_range < _FLAT_SIGNAL_REL_TOL * h_scale:
        return []
    h_dot = np.gradient(h, t)
    idx = _same_direction_crossings(h_dot)
    return list(zip(idx[:-1], idx[1:]))


def _windowed(t: np.ndarray, *arrays: np.ndarray, window_frac: float):
    n = len(t)
    start = int((1.0 - window_frac) * n)
    return (t[start:],) + tuple(a[start:] for a in arrays)


def _trustworthy_cycles(t_w: np.ndarray, h_w: np.ndarray, cycles: list[tuple[int, int]],
                        D: float | None, a_star_floor: float = DEFAULT_A_STAR_NOISE_FLOOR,
                        min_period_frac_of_median: float = 0.5
                        ) -> list[tuple[int, int]]:
    """Filter individual cycles by their OWN amplitude against the noise
    floor, not just the window as a whole -- and drop cycles whose period
    is anomalously short relative to the rest (truncated/partial cycles).

    cycles_from_displacement's gate only checks whether the WINDOW contains
    any real signal anywhere -- a window that starts with a real decaying
    transient and settles into noise for the rest passes that gate (the
    early cycles are real), but then thousands of noise-driven micro-cycles
    from the settled tail dominate a plain average by sheer count (seen for
    real: 7819 "cycles" averaging to A_star~3e-7 and f_osc~23 Hz, when only
    the first handful of cycles, at the true ~0.29 Hz, were real).

    Amplitude filtering alone isn't enough: the LAST surviving cycle at a
    transient-to-noise (or window-edge) boundary can have a real amplitude
    (it's a half-real, half-noise span) but a badly truncated period --
    seen for real: 0.19s vs a true ~3.45s period, once the noise tail was
    otherwise correctly excluded. Dropping cycles shorter than
    min_period_frac_of_median of the survivors' median period catches this
    without needing to know the true frequency in advance.
    """
    if D is None or D <= 0:
        return cycles
    amp_ok = []
    for i0, i1 in cycles:
        seg = h_w[i0:i1 + 1]
        amp = (float(seg.max()) - float(seg.min())) / 2.0
        if amp / D >= a_star_floor:
            amp_ok.append((i0, i1))
    if len(amp_ok) < 2:
        return amp_ok
    periods = np.array([t_w[i1] - t_w[i0] for i0, i1 in amp_ok])
    median_T = float(np.median(periods))
    return [c for c, T in zip(amp_ok, periods) if T >= min_period_frac_of_median * median_T]


# ── (c) Cycle-based amplitude and frequency ──────────────────────────────

def cycle_amplitude_and_frequency(t: np.ndarray, h: np.ndarray, D: float,
                                  window_frac: float = 0.5) -> dict:
    """Cycle-based half-range LCO amplitude (A_star = A/D) and oscillation
    frequency, averaged over full cycles within the last window_frac of the
    trajectory -- NOT mean(|h|), which understates a non-sinusoidal LCO's
    true peak-to-peak amplitude."""
    t_w, h_w = _windowed(t, h, window_frac=window_frac)
    cycles = _trustworthy_cycles(t_w, h_w, cycles_from_displacement(t_w, h_w, D=D), D=D)
    if not cycles:
        return {"A_phys": float("nan"), "A_star": float("nan"),
                "f_osc": float("nan"), "n_cycles": 0}
    amps, periods = [], []
    for i0, i1 in cycles:
        seg = h_w[i0:i1 + 1]
        amps.append((float(seg.max()) - float(seg.min())) / 2.0)
        periods.append(float(t_w[i1] - t_w[i0]))
    A = float(np.mean(amps))
    T = float(np.mean(periods))
    return {
        "A_phys": A,
        "A_star": A / D if D else float("nan"),
        "f_osc": 1.0 / T if T > 0 else float("nan"),
        "n_cycles": len(cycles),
    }


# ── (a) Cycle-based energy ────────────────────────────────────────────────

def cycle_average_energy(t: np.ndarray, h: np.ndarray, cl: np.ndarray,
                         window_frac: float = 0.5, D: float | None = None
                         ) -> dict:
    """E_f = oint C_L dh, averaged over full cycles in the analysis window.
    dh = h_dot*dt, so each cycle's integral is trapz(C_L * h_dot, t).

    Reports E_f_std (and coefficient of variation) alongside the mean:
    E_f is only a physically clean number for a genuine limit cycle: where
    the response's phase wanders cycle-to-cycle (e.g. against a
    phase-randomised residual forcing), individual cycles' energy transfer
    varies and the mean alone overstates how precise that number is. A
    small n_cycles makes E_f_std itself noisy -- read it as "spread", not
    a tight error bar, when n_cycles is small.
    """
    t_w, h_w, cl_w = _windowed(t, h, cl, window_frac=window_frac)
    cycles = _trustworthy_cycles(t_w, h_w, cycles_from_displacement(t_w, h_w, D=D), D=D)
    if not cycles:
        return {"E_f_mean": float("nan"), "E_f_std": float("nan"),
                "E_f_cv": float("nan"), "n_cycles": 0}
    h_dot_w = np.gradient(h_w, t_w)
    energies = np.array([
        float(np.trapezoid(cl_w[i0:i1 + 1] * h_dot_w[i0:i1 + 1], t_w[i0:i1 + 1]))
        for i0, i1 in cycles
    ])
    E_f_mean = float(np.mean(energies))
    E_f_std = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
    E_f_cv = E_f_std / abs(E_f_mean) if E_f_mean else float("nan")
    return {"E_f_mean": E_f_mean, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
            "n_cycles": len(cycles)}


# ── (b) Single-bin DFT amplitude + phase ─────────────────────────────────

def single_bin_dft_amplitude(t: np.ndarray, x: np.ndarray, f: float) -> float:
    """|C_L| at frequency f (Hz) via direct-summation DFT (tolerant of a
    non-uniform grid, unlike FFT). Amplitude convention: for x(t) = F*sin
    (2*pi*f*t + phase) sampled over ~integer periods, this recovers F."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return float("nan")
    kernel = np.exp(-2j * np.pi * f * np.asarray(t, dtype=float))
    return float(2.0 / n * np.abs(np.sum(x * kernel)))


def energy_and_phase(t: np.ndarray, h: np.ndarray, cl: np.ndarray, D: float,
                     window_frac: float = 0.5) -> dict:
    """(a) E_f (mean +- cycle-to-cycle spread) and (b) phi from
    E_f = pi * A * F1 * sin(phi). E_f_std/E_f_cv report how much E_f varies
    cycle-to-cycle -- phi (derived from the mean E_f) is only as precise as
    that spread allows, which matters wherever the response's phase wanders
    (e.g. against a phase-randomised residual) rather than sitting on a
    clean limit cycle."""
    energy = cycle_average_energy(t, h, cl, window_frac, D=D)
    E_f, E_f_std, E_f_cv, n_cycles = (
        energy["E_f_mean"], energy["E_f_std"], energy["E_f_cv"], energy["n_cycles"]
    )
    amp = cycle_amplitude_and_frequency(t, h, D, window_frac)
    A_phys, f_osc = amp["A_phys"], amp["f_osc"]

    if n_cycles == 0 or not np.isfinite(f_osc) or A_phys <= 0:
        return {"E_f": E_f, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
                "A_phys": A_phys, "f_osc": f_osc,
                "F1": float("nan"), "sin_phi": float("nan"),
                "sin_phi_clipped": float("nan"), "phi_rad": float("nan"),
                "n_cycles": n_cycles}

    t_w, cl_w = _windowed(t, cl, window_frac=window_frac)
    F1 = single_bin_dft_amplitude(t_w, cl_w, f_osc)
    denom = np.pi * A_phys * F1
    sin_phi = E_f / denom if denom > 0 else float("nan")
    sin_phi_clipped = float(np.clip(sin_phi, -1.0, 1.0)) if np.isfinite(sin_phi) else float("nan")
    phi_rad = float(np.arcsin(sin_phi_clipped)) if np.isfinite(sin_phi_clipped) else float("nan")

    return {
        "E_f": E_f, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
        "A_phys": A_phys, "f_osc": f_osc, "F1": F1,
        "sin_phi": float(sin_phi) if np.isfinite(sin_phi) else float("nan"),
        "sin_phi_clipped": sin_phi_clipped, "phi_rad": phi_rad,
        "n_cycles": n_cycles,
    }


# ── (d) Stability classification ─────────────────────────────────────────

def classify_stability(t: np.ndarray, h: np.ndarray,
                       window_frac: float = 0.5,
                       stationary_frac_threshold: float = 0.10) -> dict:
    """Fit an exponential to the Hilbert-envelope of h over the last
    window_frac of the trajectory; label by the fitted growth rate.

    stationary_frac_threshold: the fractional envelope change over the
    window (exp(growth_rate * T) - 1) must exceed this in magnitude to be
    called anything other than 'stationary_lco'. Otherwise sign of the
    growth rate selects 'divergence' (still growing) or 'decay_to_rest'
    (shrinking toward zero).

    fractional_envelope_change is self-relative -- relative to the FITTED
    envelope level at the window's own start, not to D or any
    case-independent scale -- so it can be large purely because the
    starting level is tiny (a case already collapsed near-zero by the
    start of the analysis window). envelope_change_abs (same physical
    units as h) is reported alongside it for exactly that reason: read
    fractional_envelope_change together with envelope_change_abs, not
    alone, whenever the case's amplitude is small.

    No minimum-cycle-count guard is applied here (unlike
    cycle_amplitude_and_frequency/cycle_average_energy, which filter
    through _trustworthy_cycles): the only length check is on raw sample
    count (len(t_w) < 4), not on how many real oscillation cycles are
    present. A signal that has decayed to pure numerical noise can still
    receive a confident label from that noise's own Hilbert-envelope
    trend -- there is no noise-floor gate here.
    """
    t_w, h_w = _windowed(t, h, window_frac=window_frac)
    if len(t_w) < 4:
        return {"label": "insufficient_data", "growth_rate_per_s": float("nan"),
                "fractional_envelope_change": float("nan"),
                "envelope_change_abs": float("nan"),
                "envelope_start": float("nan"), "envelope_end": float("nan"),
                "envelope_start_fitted": float("nan"), "envelope_end_fitted": float("nan")}

    h_centered = np.asarray(h_w, dtype=float) - float(np.mean(h_w))
    envelope = np.abs(hilbert(h_centered))
    envelope_safe = np.clip(envelope, 1e-12, None)

    b, log_a = np.polyfit(t_w, np.log(envelope_safe), 1)
    T = float(t_w[-1] - t_w[0])
    frac_change = float(np.exp(b * T) - 1.0) if np.isfinite(b) else float("nan")

    # Fitted (regression-line) envelope level at the window's start/end --
    # robust to single-sample endpoint noise, and the basis frac_change
    # itself is derived from: frac_change = (env_end_fitted - env_start_fitted)
    # / env_start_fitted, so env_change_abs below is exactly consistent
    # with the reported fractional value (not the noisier raw endpoints).
    if np.isfinite(b) and np.isfinite(log_a):
        env_start_fitted = float(np.exp(log_a + b * t_w[0]))
        env_end_fitted = float(np.exp(log_a + b * t_w[-1]))
        env_change_abs = env_end_fitted - env_start_fitted
    else:
        env_start_fitted = env_end_fitted = env_change_abs = float("nan")

    if not np.isfinite(frac_change):
        label = "insufficient_data"
    elif abs(frac_change) < stationary_frac_threshold:
        label = "stationary_lco"
    elif b > 0:
        label = "divergence"
    else:
        label = "decay_to_rest"

    return {
        "label": label,
        "growth_rate_per_s": float(b),
        "fractional_envelope_change": frac_change,
        "envelope_change_abs": env_change_abs,
        "envelope_start": float(envelope[0]),
        "envelope_end": float(envelope[-1]),
        "envelope_start_fitted": env_start_fitted,
        "envelope_end_fitted": env_end_fitted,
    }


# ── Per-case driver ───────────────────────────────────────────────────────

def compute_signal_metrics(t: np.ndarray, h: np.ndarray, cl: np.ndarray, D: float,
                           window_frac: float = 0.5) -> dict:
    """All four metrics for ONE signal (surrogate or CFD)."""
    amp = cycle_amplitude_and_frequency(t, h, D, window_frac)
    energy = energy_and_phase(t, h, cl, D, window_frac)
    stab = classify_stability(t, h, window_frac)
    return {**amp, **energy, **stab}


def compare_surrogate_vs_cfd(surrogate: dict, cfd: dict) -> dict:
    """(a)/(b)/(c) comparison quantities from two compute_signal_metrics() dicts."""
    out = {}
    out["A_star_error"] = surrogate["A_star"] - cfd["A_star"]
    out["A_star_rel_error"] = (out["A_star_error"] / cfd["A_star"]
                               if cfd.get("A_star") else float("nan"))
    out["f_osc_error"] = surrogate["f_osc"] - cfd["f_osc"]
    out["f_osc_rel_error"] = (out["f_osc_error"] / cfd["f_osc"]
                              if cfd.get("f_osc") else float("nan"))
    Ef_cfd = cfd.get("E_f")
    out["energy_error_eps_E"] = ((surrogate["E_f"] - Ef_cfd) / abs(Ef_cfd)
                                  if Ef_cfd not in (None, 0) and np.isfinite(Ef_cfd)
                                  else float("nan"))
    if np.isfinite(surrogate.get("phi_rad", float("nan"))) and np.isfinite(cfd.get("phi_rad", float("nan"))):
        out["phase_error_rad"] = surrogate["phi_rad"] - cfd["phi_rad"]
    else:
        out["phase_error_rad"] = float("nan")
    return out


def compute_case_metrics(npz_path: str | Path, window_frac: float = 0.5) -> dict:
    """Load one coupled_*.npz and compute all four metrics for the
    surrogate, and (if cl_cfd is present -- added to coupled_inference.py's
    save call alongside this module) for CFD plus the comparison errors.

    The surrogate free-runs autonomously for the full requested duration,
    but the CFD reference is capped by however long that Ur case was
    actually simulated -- h_cfd/cl_cfd can come out SHORTER than t/h/cl.
    `surrogate_*` columns always reflect the surrogate's FULL trajectory
    (its own long-run behavior). The CFD comparison (cfd_*, and the
    *_error columns) is computed on both signals truncated to their common
    overlap, so "same cycles for surrogate and CFD" actually holds -- a
    truncated comparison is flagged via cfd_window_truncated_to_n rather
    than silently comparing mismatched windows (which used to crash).
    """
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    t, h, cl = d["t"], d["h"], d["cl"]
    D = float(d["D"])
    Ur = float(d["Ur"])

    row = {"npz": str(npz_path), "Ur": Ur}
    surrogate = compute_signal_metrics(t, h, cl, D, window_frac)
    row.update({f"surrogate_{k}": v for k, v in surrogate.items()})

    if "cl_cfd" in d.files and "h_cfd" in d.files:
        h_cfd, cl_cfd = d["h_cfd"], d["cl_cfd"]
        n_common = min(len(t), len(h_cfd), len(cl_cfd))
        if n_common < len(t):
            row["cfd_window_truncated_to_n"] = n_common
        t_c, h_c, cl_c = t[:n_common], h[:n_common], cl[:n_common]
        h_cfd_c, cl_cfd_c = h_cfd[:n_common], cl_cfd[:n_common]

        surrogate_matched = compute_signal_metrics(t_c, h_c, cl_c, D, window_frac)
        cfd = compute_signal_metrics(t_c, h_cfd_c, cl_cfd_c, D, window_frac)
        row.update({f"cfd_{k}": v for k, v in cfd.items()})
        row.update(compare_surrogate_vs_cfd(surrogate_matched, cfd))
    else:
        row["_missing_cfd_cl"] = True

    return row


# ── Batch CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_dir", required=True,
                   help="Directory of coupled_*.npz files (an evaluate_all.py "
                        "--output_dir sweep directory)")
    p.add_argument("--window_frac", type=float, default=0.5,
                   help="Fraction of the trajectory (from the end) treated as "
                        "the analysis window for all four metrics (default 0.5)")
    p.add_argument("--out_csv", default=None,
                   help="Output CSV path (default: <npz_dir>/closed_loop_metrics.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    npz_files = sorted(npz_dir.glob("coupled_*.npz"))
    if not npz_files:
        raise SystemExit(f"No coupled_*.npz files found in {npz_dir}")

    rows = []
    n_missing_cfd = 0
    for npz_path in npz_files:
        try:
            row = compute_case_metrics(npz_path, window_frac=args.window_frac)
        except Exception as e:
            print(f"  FAILED {npz_path.name}: {e}")
            continue
        if row.pop("_missing_cfd_cl", False):
            n_missing_cfd += 1
        rows.append(row)
        stab = row.get("surrogate_label", "?")
        print(f"  Ur={row['Ur']:.4f}  stability={stab:14s}  "
              f"A*={row.get('surrogate_A_star', float('nan')):.4f}  "
              f"f_osc={row.get('surrogate_f_osc', float('nan')):.4f}")

    if n_missing_cfd:
        print(f"\nWARNING: {n_missing_cfd}/{len(rows)} npz file(s) have no cl_cfd "
              f"(pre-date this module's addition to coupled_inference.py's save "
              f"call) -- surrogate-only metrics were computed for those; "
              f"re-run evaluate_all.py for that model to get comparison metrics.")

    out_csv = Path(args.out_csv) if args.out_csv else npz_dir / "closed_loop_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    # stable, readable column order: identity/label columns first
    front = ["npz", "Ur", "surrogate_label", "cfd_label"]
    fieldnames = [c for c in front if c in fieldnames] + \
                 [c for c in fieldnames if c not in front]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} case(s) to {out_csv}")


if __name__ == "__main__":
    main()
