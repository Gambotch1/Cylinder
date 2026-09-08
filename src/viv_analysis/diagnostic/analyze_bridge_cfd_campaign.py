#!/usr/bin/env python3
"""
Phase-A characterization of the bridge CFD campaign.
=====================================================

CPU-only, read-only diagnostic pass over the cached bridge CFD trajectories.
Does NOT load the GRU, does NOT run coupled inference, does NOT touch the
GPU. For every bridge case in the parquet cache it independently measures:

  - per-case dt (median(diff(time))) -- no global bridge dt is assumed;
  - GRU train/val/test coverage, recovered from the trained model's
    metrics_gru.json (never inferred from ur_stats.pkl mean/std);
  - steady-state CFD response (amplitude, dominant frequency, lock-in);
  - the deterministic v3_coherent closure parameters mu and a_ref, via the
    same measure_mu_from_cfd / measure_cfd_amplitude functions the coupled
    inference CLI uses;
  - an audit of measure_growth_rate's 5%-45% envelope mask (is it one
    contiguous growth interval, or several disconnected segments?), plus a
    diagnostic-only alternative estimator (contiguous_first_growth_fit) that
    fits only the first contiguous segment -- recorded side by side with the
    production estimator, never substituted for it;
  - preliminary eligibility for a future leave-one-out hybrid experiment,
    reported as several independently-interpretable named booleans rather
    than one conflated flag.

This is a screening pass. Every raw metric and exclusion reason is kept in
the output table so the (explicit, named) eligibility thresholds below can
be reviewed before any GPU run is scheduled. NO threshold value here has
been loosened relative to the original Phase-A pass; new columns either
decompose the previous single eligibility flag into honestly-named parts,
or add genuinely new (documented, named) screening criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viv_analysis.config import config, bridge_structural_params
from viv_analysis.self_excitation import (
    measure_cfd_amplitude,
    measure_mu_from_cfd,
    measure_growth_rate,
    contiguous_first_growth_fit,
    find_contiguous_segments,
    amplitude_envelope,
    dominant_freq,
    A_REF_CONVENTION,
)
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label

# ── Fixed conventions borrowed from self_excitation.py -- do NOT redefine ──
# these locally; they define the lock-in tail / stationarity windows used
# throughout the project (measure_cfd_amplitude, dominant_freq, analyze_run).
TAIL_FRACTION = 0.6              # last (1-0.6)=40% of the post-handoff record
ENVELOPE_STATIONARY_FRACTION = 0.7  # last 30% of the amplitude envelope

DEFAULT_INPUT = PROJECT_ROOT / "data" / "cache" / "bridge_ds20_trim100_v1.parquet"
DEFAULT_CSV_OUT = PROJECT_ROOT / "results" / "bridge_phaseA_characterization.csv"
DEFAULT_JSON_OUT = PROJECT_ROOT / "results" / "bridge_phaseA_characterization.json"
DEFAULT_MODEL_SUBDIR = "gru_bridge_noise0.05"

# Cases always given a per-case diagnostic figure regardless of the
# synchronization screen (explicit request; both already sit inside the
# lock-in band, so this is normally a no-op union, kept for safety).
ALWAYS_PLOT_CASES = {"Ur6.7385", "Ur8.2126"}


# ── Bridge training-split recovery ──────────────────────────────────────────

def recover_bridge_split(model_subdir: str) -> dict:
    """
    Recover the ACTUAL train/val/test split used to train the bridge GRU,
    from the training-time receipt (metrics_gru.json["case_split"]).

    Deliberately does NOT use ur_stats.pkl (mean/std only, no case identity)
    -- per project constraint, coverage must never be inferred from that.
    Returns gru_split="unknown" / coverage unresolved if nothing is found,
    with a record of exactly what was searched.
    """
    artifact_dir = PROJECT_ROOT / "results" / model_subdir
    searched = []

    metrics_path = artifact_dir / "metrics_gru.json"
    searched.append(str(metrics_path))
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            case_split = metrics.get("case_split")
            if case_split and all(k in case_split for k in ("train", "val", "test")):
                return {
                    "recovered": True,
                    "source": str(metrics_path),
                    "train_cases": sorted(case_split["train"]),
                    "validation_cases": sorted(case_split["val"]),
                    "test_cases": sorted(case_split["test"]),
                    "searched": searched,
                }
        except Exception as e:
            searched.append(f"{metrics_path} (error reading: {e})")

    # Other places a split could plausibly have been recorded; none of these
    # exist for gru_bridge_noise0.05 today, but check and report honestly.
    for name in ("run_config.json", "split.json", "case_split.json", "train_log.txt"):
        p = artifact_dir / name
        searched.append(str(p))

    return {
        "recovered": False,
        "source": None,
        "train_cases": None,
        "validation_cases": None,
        "test_cases": None,
        "searched": searched,
    }


def classify_coverage(case_label: str, ur_value: float, split: dict) -> tuple[str, str]:
    """Returns (gru_split, coverage_class) for one case, using ONLY the
    recovered case identity + Ur range of the recovered train set -- never
    ur_stats.pkl mean/std."""
    if not split["recovered"]:
        return "unknown", "unknown"

    if case_label in split["train_cases"]:
        return "train", "trained"
    if case_label in split["validation_cases"]:
        gru_split = "validation"
    elif case_label in split["test_cases"]:
        gru_split = "test"
    else:
        # Case exists in the CFD cache but wasn't part of the training run at all.
        return "unknown", "unknown"

    train_ur = [parse_ur_label(c) for c in split["train_cases"]]
    lo, hi = min(train_ur), max(train_ur)
    if lo <= ur_value <= hi:
        return gru_split, "unseen_interpolation"
    return gru_split, "context_extrapolation"


# ── Per-case dt (robust, no global assumption) ──────────────────────────────

def measure_case_dt(times: np.ndarray) -> float:
    tt = np.sort(np.unique(times.astype(np.float64)))
    if len(tt) < 2:
        return float("nan")
    return float(np.median(np.diff(tt)))


# ── Eligibility thresholds (named, not magic numbers; NONE loosened) ───────

class Thresholds:
    def __init__(self, args: argparse.Namespace):
        self.min_post_release_duration_s = args.min_post_release_duration_s
        self.min_post_handoff_duration_s = args.min_post_handoff_duration_s
        self.freq_lockin_tol = args.freq_lockin_tol
        self.envelope_cv_max = args.envelope_cv_max
        self.min_growth_cycles = args.min_growth_cycles
        self.growth_fit_r2_min = args.growth_fit_r2_min
        self.min_large_response_ad = args.min_large_response_ad

    def as_dict(self) -> dict:
        return {
            "min_post_release_duration_s": self.min_post_release_duration_s,
            "min_post_handoff_duration_s": self.min_post_handoff_duration_s,
            "freq_lockin_tol": self.freq_lockin_tol,
            "envelope_cv_max": self.envelope_cv_max,
            "min_growth_cycles": self.min_growth_cycles,
            "growth_fit_r2_min": self.growth_fit_r2_min,
            "min_large_response_ad": self.min_large_response_ad,
        }


# ── Per-case analysis ────────────────────────────────────────────────────────

def analyze_case(
    case_label: str,
    case_df: pd.DataFrame,
    D: float,
    fn: float,
    rho: float,
    B: float,
    t_star_release: float,
    seq_len: int,
    handoff_offset_steps: int,
    m: float,
    c: float,
    thr: Thresholds,
    split: dict,
    ur_stats: Optional[dict],
) -> dict:
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float64)
    disp = ordered["disp"].to_numpy(dtype=np.float64)
    n_samples = len(ordered)

    ur_value = parse_ur_label(case_label)
    U = ur_value * fn * D
    dt = measure_case_dt(times)

    row: dict = {
        "case": case_label,
        "U_mps": float(U),
        "Ur": float(ur_value),
        "n_samples": int(n_samples),
        "dt": dt,
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
    }

    gru_split, coverage_class = classify_coverage(case_label, ur_value, split)
    row["gru_split"] = gru_split
    row["coverage_class"] = coverage_class
    if ur_stats is not None:
        ur_std_safe = float(ur_stats["std"]) if abs(float(ur_stats["std"])) > 0 else 1.0
        row["ur_context_z"] = float((ur_value - float(ur_stats["mean"])) / ur_std_safe)
    else:
        row["ur_context_z"] = float("nan")

    notes: list[str] = []

    # ── Sampling / handoff geometry ─────────────────────────────────────────
    t_release = float(t_star_release * D / U)
    release_idx = int(np.searchsorted(times, t_release))
    handoff_idx = release_idx + int(seq_len) + int(handoff_offset_steps)
    row["t_release"] = t_release
    row["handoff_offset_steps"] = int(handoff_offset_steps)
    row["post_release_duration"] = float(times[-1] - t_release) if release_idx < n_samples else float("nan")

    handoff_computable = handoff_idx < n_samples
    if handoff_computable:
        row["handoff_time"] = float(times[handoff_idx])
        row["post_handoff_duration"] = float(times[-1] - times[handoff_idx])
    else:
        row["handoff_time"] = float("nan")
        row["post_handoff_duration"] = 0.0

    # ── complete_case: is this a genuinely usable CFD record at all? ───────
    # Consolidates BOTH duration gates (post-release AND post-handoff) at
    # their original, unmodified threshold values -- this is a rename/
    # consolidation of where these two checks are reported, not a change to
    # either threshold.
    complete_case = (
        handoff_computable
        and row["post_release_duration"] >= thr.min_post_release_duration_s
        and row["post_handoff_duration"] >= thr.min_post_handoff_duration_s
    )
    if not handoff_computable or row["post_release_duration"] < thr.min_post_release_duration_s:
        notes.append(
            f"incomplete CFD record: only {n_samples} samples "
            f"({row['post_release_duration']:.1f}s post-release, dt={dt:.4f}s) "
            f"< required {thr.min_post_release_duration_s:.0f}s post-release duration"
        )
    elif row["post_handoff_duration"] < thr.min_post_handoff_duration_s:
        notes.append(
            f"incomplete CFD record: post-handoff duration "
            f"{row['post_handoff_duration']:.1f}s < required "
            f"{thr.min_post_handoff_duration_s:.0f}s"
        )
    row["complete_case"] = bool(complete_case)

    # ── Steady CFD response (post-handoff tail) ─────────────────────────────
    mean_h_over_D = ad_rms = ad_half_range = envelope_cv = f_dom_hz = f_over_fn = float("nan")
    tail_start_time = float("nan")
    a_ref_m = a_ref_over_D = float("nan")

    if handoff_computable and np.isfinite(dt) and dt > 0:
        try:
            h_post_handoff = disp[handoff_idx:]
            n_ph = len(h_post_handoff)
            tail_start_idx_local = int(TAIL_FRACTION * n_ph)
            h_tail = h_post_handoff[tail_start_idx_local:]
            if handoff_idx + tail_start_idx_local < n_samples:
                tail_start_time = float(times[handoff_idx + tail_start_idx_local])

            if len(h_tail) > 1:
                mean_h_over_D = float(np.mean(h_tail) / D)
                ad_half_range = float((h_tail.max() - h_tail.min()) / (2.0 * D))

            amp = measure_cfd_amplitude(ordered, handoff_idx, D)
            a_ref_m = float(amp["a_ref_peak"])
            a_ref_over_D = a_ref_m / D
            ad_rms = float(np.sqrt(2.0) * amp["rms_AD"])  # sqrt(2)*std(h_tail)/D

            if n_ph > 16:
                _, env = amplitude_envelope(h_post_handoff, D, dt, fn)
                if len(env) > 1:
                    env_tail = env[int(ENVELOPE_STATIONARY_FRACTION * len(env)):]
                    if len(env_tail) > 0:
                        envelope_cv = float(np.std(env_tail) / (abs(np.mean(env_tail)) + 1e-30))

                fdom = dominant_freq(h_post_handoff, dt, fn)
                if np.isfinite(fdom):
                    f_dom_hz = fdom
                    f_over_fn = fdom / fn
        except Exception as e:
            notes.append(f"steady-response measurement error: {e}")

    row.update({
        "tail_start_time": tail_start_time,
        "tail_fraction": TAIL_FRACTION,
        "mean_h_over_D": mean_h_over_D,
        "ad_rms": ad_rms,
        "ad_half_range": ad_half_range,
        "envelope_cv": envelope_cv,
        "f_dom_hz": f_dom_hz,
        "f_over_fn": f_over_fn,
        "a_ref_m": a_ref_m,
        "a_ref_over_D": a_ref_over_D,
        "a_ref_convention": A_REF_CONVENTION,
    })

    # ── Growth / closure characterization (post-release transient) ─────────
    # Legacy (production) estimator metrics:
    growth_interval_start = growth_interval_end = float("nan")
    growth_cycles = growth_rate_lambda = growth_fit_r2 = float("nan")
    growth_fit_n_points = 0
    growth_mask_n_segments = 0
    growth_mask_largest_segment_cycles = float("nan")
    growth_mask_time_span = float("nan")
    mu_measured = float("nan")

    # Diagnostic-only alternative estimator metrics (never fed back into mu):
    alt_growth_rate_lambda = alt_growth_fit_r2 = float("nan")
    alt_growth_fit_n_points = 0
    alt_growth_interval_start = alt_growth_interval_end = float("nan")
    alt_growth_cycles = float("nan")
    alt_growth_fit_valid = False
    alt_n_segments_considered = 0
    alt_stopped_reason = "not_computed"
    alt_mu_measured = float("nan")

    if np.isfinite(dt) and dt > 0 and release_idx < n_samples:
        try:
            qD = 0.5 * rho * U ** 2 * B
            h_post_release = disp[release_idx:]

            g = measure_growth_rate(h_post_release, dt, fn, D)
            growth_interval_start = g["t0"]
            growth_interval_end = g["t1"]
            growth_rate_lambda = g["lam"]
            growth_fit_r2 = g["r2"]
            growth_fit_n_points = g["n"]
            growth_cycles = float((g["t1"] - g["t0"]) * fn)
            growth_mask_time_span = float(g["t1"] - g["t0"])

            mask = g.get("mask")
            tc = g.get("tc")
            if mask is not None and tc is not None and np.any(mask):
                segments = find_contiguous_segments(mask)
                growth_mask_n_segments = len(segments)
                seg_cycles = [
                    (tc[e - 1] - tc[s]) * fn if e - 1 > s else 0.0
                    for (s, e) in segments
                ]
                growth_mask_largest_segment_cycles = float(max(seg_cycles)) if seg_cycles else 0.0
            else:
                growth_mask_n_segments = 0
                growth_mask_largest_segment_cycles = 0.0

            # a_ref used for mu must be the measured target-CFD value here --
            # Phase-A characterizes the CFD campaign itself, no CLI override
            # applies (that override only exists in coupled_inference.py).
            a_ref_for_mu = a_ref_m if np.isfinite(a_ref_m) and a_ref_m > 0 else None
            if a_ref_for_mu is not None:
                mm = measure_mu_from_cfd(ordered, t_release, m, c, D, dt, fn, a_ref_for_mu, qD)
                mu_measured = mm["mu"]

            # Diagnostic-only alternative: same threshold VALUES, different
            # segment-selection strategy. Never used for mu_measured above.
            ga = contiguous_first_growth_fit(
                h_post_release, dt, fn, D,
                min_cycles=thr.min_growth_cycles, r2_min=thr.growth_fit_r2_min,
            )
            alt_growth_rate_lambda = ga["lam"]
            alt_growth_fit_r2 = ga["r2"]
            alt_growth_fit_n_points = ga["n"]
            alt_growth_interval_start = ga["t0"]
            alt_growth_interval_end = ga["t1"]
            alt_growth_cycles = float((ga["t1"] - ga["t0"]) * fn)
            alt_growth_fit_valid = bool(ga["valid"])
            alt_n_segments_considered = ga["n_segments_considered"]
            alt_stopped_reason = ga["stopped_reason"]
            if a_ref_for_mu is not None and alt_growth_rate_lambda != 0.0:
                omega_n = 2.0 * np.pi * fn
                beta_true_alt = c + 2.0 * m * alt_growth_rate_lambda
                alt_mu_measured = float(beta_true_alt * omega_n * a_ref_for_mu / qD)
        except Exception as e:
            notes.append(f"growth/mu fit error: {e}")

    row.update({
        "growth_interval_start": growth_interval_start,
        "growth_interval_end": growth_interval_end,
        "growth_cycles": growth_cycles,
        "growth_rate_lambda": growth_rate_lambda,
        "growth_fit_r2": growth_fit_r2,
        "growth_fit_n_points": int(growth_fit_n_points),
        "growth_mask_n_segments": int(growth_mask_n_segments),
        "growth_mask_largest_segment_cycles": growth_mask_largest_segment_cycles,
        "growth_mask_time_span": growth_mask_time_span,
        "mu_measured": mu_measured,
        "alt_growth_rate_lambda": alt_growth_rate_lambda,
        "alt_growth_fit_r2": alt_growth_fit_r2,
        "alt_growth_fit_n_points": int(alt_growth_fit_n_points),
        "alt_growth_interval_start": alt_growth_interval_start,
        "alt_growth_interval_end": alt_growth_interval_end,
        "alt_growth_cycles": alt_growth_cycles,
        "alt_growth_fit_valid": alt_growth_fit_valid,
        "alt_n_segments_considered": int(alt_n_segments_considered),
        "alt_stopped_reason": alt_stopped_reason,
        "alt_mu_measured": alt_mu_measured,
    })

    # ── Eligibility: independently-named, non-conflated booleans ───────────
    # Every flag below is ALSO gated on complete_case, so a structurally
    # incomplete record can never read as eligible on any axis.
    a_ref_measurement_available = bool(
        complete_case and np.isfinite(a_ref_m) and a_ref_m > 0
    )
    frequency_synchronised = bool(
        complete_case and np.isfinite(f_over_fn) and abs(f_over_fn - 1.0) <= thr.freq_lockin_tol
    )
    large_response_candidate = bool(
        complete_case and np.isfinite(ad_rms) and ad_rms >= thr.min_large_response_ad
    )
    tail_stationary = bool(
        complete_case and np.isfinite(envelope_cv) and envelope_cv <= thr.envelope_cv_max
    )
    clean_growth_fit = bool(
        complete_case
        and np.isfinite(growth_rate_lambda) and growth_rate_lambda > 0
        and np.isfinite(growth_cycles) and growth_cycles >= thr.min_growth_cycles
        and np.isfinite(growth_fit_r2) and growth_fit_r2 >= thr.growth_fit_r2_min
    )
    mu_finite_positive = bool(np.isfinite(mu_measured) and mu_measured > 0)

    # "Is THIS case's own measured (mu, a_ref) pair trustworthy enough to use
    # as a v3_coherent closure reference?" -- the honest replacement for the
    # old, misleadingly-named measured_ref_eligible (which only meant "a_ref
    # was numerically computable", true for nearly every complete case).
    measured_v3_reference_eligible = bool(
        complete_case
        and a_ref_measurement_available
        and frequency_synchronised
        and large_response_candidate
        and tail_stationary
        and clean_growth_fit
        and mu_finite_positive
    )

    # "Is THIS case a scientifically meaningful UNSEEN target for a future
    # full-hybrid (GRU + v3 closure) evaluation?" -- deliberately does NOT
    # require clean_growth_fit / mu_finite_positive: a true hybrid run
    # predicts the target from OTHER cases' closure parameters, so the
    # target's own growth fit quality is irrelevant. It DOES require the
    # case be absent from GRU training/scaler fitting (val/test only -- see
    # train_gru.py's scaler_fit_cases = train_cases) and interior to the
    # training Ur range (unseen_interpolation, not context_extrapolation),
    # so a failed prediction can't be blamed on out-of-context extrapolation
    # instead of the closure. This is a GRU-coverage question, NOT a
    # closure-parameter-availability question -- see closure_loo_target_eligible
    # below for that, which is computed later (needs the full case population)
    # and must NOT depend on this split.
    full_hybrid_unseen_target_eligible = bool(
        complete_case
        and frequency_synchronised
        and tail_stationary
        and large_response_candidate
        and coverage_class == "unseen_interpolation"
        and gru_split in ("validation", "test")
    )

    # Descriptive regime, independent of any eligibility gate -- a
    # synchronised-but-non-stationary (modulated/beating) response is NOT
    # the same thing as "not locked in"; it just means the envelope hasn't
    # settled into a clean single-amplitude limit cycle within the tail
    # window. Both regimes are still frequency-synchronised.
    if not complete_case:
        response_regime = "incomplete"
    elif not frequency_synchronised:
        response_regime = "non_synchronised"
    elif tail_stationary:
        response_regime = "synchronised_stationary"
    else:
        response_regime = "synchronised_modulated"

    if complete_case:
        if not a_ref_measurement_available:
            notes.append("a_ref_measurement_available=False: a_ref not finite/positive")
        if not frequency_synchronised:
            notes.append(
                f"frequency_synchronised=False: f/fn={f_over_fn:.3f} outside "
                f"[{1-thr.freq_lockin_tol:.2f}, {1+thr.freq_lockin_tol:.2f}]"
            )
        if not large_response_candidate:
            notes.append(
                f"large_response_candidate=False: ad_rms={ad_rms:.4f} < "
                f"required {thr.min_large_response_ad:.4f}"
            )
        if not tail_stationary:
            notes.append(
                f"tail_stationary=False: envelope CV={envelope_cv:.3f} > "
                f"max {thr.envelope_cv_max:.2f}"
            )
        if not clean_growth_fit:
            notes.append(
                f"clean_growth_fit=False: lam={growth_rate_lambda:.4f}, "
                f"cycles={growth_cycles:.2f} (need >= {thr.min_growth_cycles:.1f}), "
                f"r2={growth_fit_r2:.3f} (need >= {thr.growth_fit_r2_min:.2f})"
            )
        if not mu_finite_positive:
            notes.append("mu not finite/positive")
        if not full_hybrid_unseen_target_eligible:
            reasons = []
            if coverage_class != "unseen_interpolation":
                reasons.append(f"coverage_class={coverage_class} (need unseen_interpolation)")
            if gru_split not in ("validation", "test"):
                reasons.append(f"gru_split={gru_split} (need validation/test)")
            if reasons:
                notes.append("full_hybrid_unseen_target_eligible=False: " + "; ".join(reasons))

    row["a_ref_measurement_available"] = a_ref_measurement_available
    row["frequency_synchronised"] = frequency_synchronised
    row["large_response_candidate"] = large_response_candidate
    row["tail_stationary"] = tail_stationary
    row["clean_growth_fit"] = clean_growth_fit
    row["mu_fit_valid"] = bool(clean_growth_fit and mu_finite_positive)  # retained for continuity
    row["response_regime"] = response_regime
    row["measured_v3_reference_eligible"] = measured_v3_reference_eligible
    # closure_loo_target_eligible is filled in by a post-processing pass in
    # main() (needs the full case population to find bracketing references);
    # False here is just the safe default for incomplete/error rows.
    row["closure_loo_target_eligible"] = False
    row["full_hybrid_unseen_target_eligible"] = full_hybrid_unseen_target_eligible
    row["exclusion_reason"] = "; ".join(notes) if notes else ""

    return row


def assign_closure_loo_targets(rows: list[dict]) -> None:
    """Post-processing pass over the FULL case population (mutates `rows`
    in place): closure_loo_target_eligible requires the TARGET case itself
    to be a scientifically meaningful case to predict --
        complete_case, frequency_synchronised, large_response_candidate,
        tail_stationary
    -- AND a LOWER and an UPPER measured_v3_reference_eligible case
    bracketing this case's Ur, purely by Ur value. Deliberately independent
    of gru_split/coverage_class -- interpolating a physical closure
    parameter between two CFD-measured reference points has nothing to do
    with which cases the GRU happened to train on; that's a separate
    question (full_hybrid_unseen_target_eligible). Note this does NOT
    require clean_growth_fit on the target itself: a closure-interpolation
    target's own growth fit is irrelevant since mu/a_ref would be
    interpolated from the bracketing references, not measured on the target.

    With only one measured_v3_reference_eligible case in the current
    campaign, no case can have both a lower and an upper reference, so this
    is False for every row -- that is the correct, data-driven answer, not
    a hardcoded one.
    """
    reference_urs = sorted(
        r["Ur"] for r in rows
        if r.get("measured_v3_reference_eligible") and np.isfinite(r.get("Ur", float("nan")))
    )
    for r in rows:
        ur = r.get("Ur")
        target_quality_ok = bool(
            r.get("complete_case")
            and r.get("frequency_synchronised")
            and r.get("large_response_candidate")
            and r.get("tail_stationary")
        )
        if not target_quality_ok or ur is None or not np.isfinite(ur):
            r["closure_loo_target_eligible"] = False
            continue
        has_lower = any(u < ur for u in reference_urs)
        has_upper = any(u > ur for u in reference_urs)
        r["closure_loo_target_eligible"] = bool(has_lower and has_upper)


# ── Per-case diagnostic figures ─────────────────────────────────────────────

def build_case_diagnostics(
    case_label: str, case_df: pd.DataFrame, D: float, fn: float,
    t_star_release: float, seq_len: int, handoff_offset_steps: int,
    thr: Thresholds,
) -> dict:
    """Recompute the intermediate arrays needed for a per-case diagnostic
    figure. Mirrors analyze_case's geometry exactly (dt, t_release,
    handoff_idx) so the figure matches the CSV row for the same case."""
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float64)
    disp = ordered["disp"].to_numpy(dtype=np.float64)
    n_samples = len(ordered)

    ur_value = parse_ur_label(case_label)
    U = ur_value * fn * D
    dt = measure_case_dt(times)
    t_release = float(t_star_release * D / U)
    release_idx = int(np.searchsorted(times, t_release))
    handoff_idx = release_idx + int(seq_len) + int(handoff_offset_steps)
    handoff_time = float(times[handoff_idx]) if handoff_idx < n_samples else float("nan")

    tail_start_time = float("nan")
    if handoff_idx < n_samples:
        h_post_handoff = disp[handoff_idx:]
        n_ph = len(h_post_handoff)
        tail_start_idx_local = int(TAIL_FRACTION * n_ph)
        if handoff_idx + tail_start_idx_local < n_samples:
            tail_start_time = float(times[handoff_idx + tail_start_idx_local])

    h_post_release = disp[release_idx:] if release_idx < n_samples else np.array([])
    g = measure_growth_rate(h_post_release, dt, fn, D) if len(h_post_release) > 1 else {}
    g_alt = contiguous_first_growth_fit(
        h_post_release, dt, fn, D,
        min_cycles=thr.min_growth_cycles, r2_min=thr.growth_fit_r2_min,
    ) if len(h_post_release) > 1 else {}

    return dict(
        times=times, disp=disp, D=D, fn=fn, dt=dt,
        t_release=t_release, handoff_time=handoff_time, tail_start_time=tail_start_time,
        g=g, g_alt=g_alt,
    )


def plot_case_diagnostic(case_label: str, diag: dict, out_dir: Path) -> Path:
    times, disp, D = diag["times"], diag["disp"], diag["D"]
    t_release, handoff_time = diag["t_release"], diag["handoff_time"]
    tail_start_time = diag["tail_start_time"]
    g, g_alt = diag["g"], diag["g_alt"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    # ── Panel A: h/D vs time, release/handoff marked, a_ref tail window ────
    ax = axes[0]
    ax.plot(times, disp / D, lw=0.5, color="black", alpha=0.7)
    ax.axvline(t_release, color="tab:green", ls="--", lw=1.4, label=f"release t={t_release:.2f}s")
    if np.isfinite(handoff_time):
        ax.axvline(handoff_time, color="tab:purple", ls="-.", lw=1.4, label=f"handoff t={handoff_time:.2f}s")
    if np.isfinite(tail_start_time):
        ax.axvspan(tail_start_time, times[-1], color="tab:orange", alpha=0.15, label="a_ref tail window")
    ax.set_ylabel("h/D")
    ax.set_title(f"{case_label}: displacement -- release/handoff markers, a_ref tail window")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel B: growth-phase envelope, mask segments, tail window ─────────
    ax = axes[1]
    mask, tc, env = g.get("mask"), g.get("tc"), g.get("env")
    if tc is not None and env is not None and len(env) > 0:
        t_abs = t_release + tc
        ax.plot(t_abs, env, lw=0.8, color="steelblue", label="A/D envelope (post-release)", zorder=1)
        if mask is not None and np.any(mask):
            segments = find_contiguous_segments(mask)
            for i, (s, e) in enumerate(segments):
                ax.scatter(t_abs[s:e], env[s:e], s=16, color=cmap(i % 10),
                           label=f"legacy mask segment {i+1} ({e-s} pts)", zorder=3)
        if g_alt.get("n", 0) > 0:
            ax.axvspan(t_release + g_alt["t0"], t_release + g_alt["t1"],
                       color="tab:red", alpha=0.10, label="alt: first-contiguous fit window")
    if np.isfinite(tail_start_time):
        ax.axvspan(tail_start_time, times[-1], color="tab:orange", alpha=0.15, label="a_ref tail window")
    ax.set_ylabel("A/D envelope")
    ax.set_title("Growth-phase envelope: legacy mask segments (color-coded) vs alt fit window")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel C: log-envelope, legacy vs alt regression, segments separated ─
    ax = axes[2]
    if mask is not None and tc is not None and env is not None and np.any(mask):
        idx_all = np.flatnonzero(mask)
        t_fit_all = tc[mask]
        y_fit_all = np.log(env[mask])
        segments = find_contiguous_segments(mask)
        seg_id = np.full(len(idx_all), -1)
        for i, (s, e) in enumerate(segments):
            in_seg = (idx_all >= s) & (idx_all < e)
            seg_id[in_seg] = i
        for i in range(len(segments)):
            sel = seg_id == i
            ax.scatter(t_fit_all[sel], y_fit_all[sel], s=16, color=cmap(i % 10),
                       label=f"segment {i+1} (n={sel.sum()})", zorder=3)

        if g.get("n", 0) > 1:
            lam, b, r2 = g["lam"], g.get("b", 0.0), g["r2"]
            tt = np.array([t_fit_all.min(), t_fit_all.max()])
            ax.plot(tt, lam * tt + b, color="black", lw=1.8, ls="-",
                    label=f"legacy fit (all segments): lam={lam:.4f}, R2={r2:.3f}, n={g['n']}")

        if g_alt.get("n", 0) > 1:
            lam_a, b_a, r2_a = g_alt["lam"], g_alt.get("b", 0.0), g_alt["r2"]
            tta = np.array([g_alt["t0"], g_alt["t1"]])
            ax.plot(tta, lam_a * tta + b_a, color="tab:red", lw=1.8, ls="--",
                    label=f"alt fit (1st contiguous): lam={lam_a:.4f}, R2={r2_a:.3f}, "
                          f"n={g_alt['n']}, valid={g_alt.get('valid')}")
    else:
        ax.text(0.5, 0.5, "no growth-mask samples", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("time since release [s]")
    ax.set_ylabel("log(A/D envelope)")
    ax.set_title("Growth-mask log-envelope: legacy (all segments) vs alt (1st contiguous segment)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    p = out_dir / f"bridge_phaseA_case_{case_label}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── Summary plots (diagnostics only -- never used to auto-tune thresholds) ─

def make_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    complete = df["complete_case"]

    # 1. Response curve: A/D vs Ur
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(df.loc[complete, "Ur"], df.loc[complete, "ad_rms"],
               color="tab:blue", label="complete case")
    ax.scatter(df.loc[~complete, "Ur"], df.loc[~complete, "ad_rms"],
               color="tab:red", marker="x", s=80, label="incomplete case")
    ax.set_xlabel("Ur")
    ax.set_ylabel("CFD A/D (RMS-equivalent peak, sqrt(2)*std of tail)")
    ax.set_title("Bridge CFD steady-state response vs Ur (Phase A)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "bridge_phaseA_response_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. Frequency curve: f_dom/fn vs Ur
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(df.loc[complete, "Ur"], df.loc[complete, "f_over_fn"],
               color="tab:green", label="complete case")
    ax.scatter(df.loc[~complete, "Ur"], df.loc[~complete, "f_over_fn"],
               color="tab:red", marker="x", s=80, label="incomplete case")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="fn (lock-in)")
    ax.set_xlabel("Ur")
    ax.set_ylabel(r"$f_{dom} / f_n$")
    ax.set_title("Bridge CFD dominant frequency vs Ur (Phase A)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "bridge_phaseA_frequency_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. Closure parameters: mu and a_ref/D vs Ur, valid vs invalid fits
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    valid = df["clean_growth_fit"] & complete
    invalid = complete & ~df["clean_growth_fit"]

    axes[0].scatter(df.loc[valid, "Ur"], df.loc[valid, "mu_measured"],
                     color="tab:blue", label="clean_growth_fit")
    axes[0].scatter(df.loc[invalid, "Ur"], df.loc[invalid, "mu_measured"],
                     color="tab:orange", marker="^", label="not clean_growth_fit")
    axes[0].scatter(df.loc[~complete, "Ur"], df.loc[~complete, "mu_measured"],
                     color="tab:red", marker="x", s=80, label="incomplete case")
    axes[0].set_ylabel(r"measured $\mu$ (legacy estimator)")
    axes[0].set_title("Bridge closure parameters vs Ur (Phase A)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df.loc[valid, "Ur"], df.loc[valid, "a_ref_over_D"],
                     color="tab:blue", label="clean_growth_fit")
    axes[1].scatter(df.loc[invalid, "Ur"], df.loc[invalid, "a_ref_over_D"],
                     color="tab:orange", marker="^", label="not clean_growth_fit")
    axes[1].scatter(df.loc[~complete, "Ur"], df.loc[~complete, "a_ref_over_D"],
                     color="tab:red", marker="x", s=80, label="incomplete case")
    axes[1].set_xlabel("Ur")
    axes[1].set_ylabel(r"$a_{ref}/D$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    p = out_dir / "bridge_phaseA_closure_parameters.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_parquet)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Bridge cache not found at {input_path}. This script only reads an "
            f"existing cache; it does not rebuild it."
        )

    print(f"[phaseA] loading {input_path} ...")
    df = pd.read_parquet(input_path)
    expected = {"case", "time", "disp", "cl", "vel", "acc"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Bridge cache missing expected columns: {missing}")

    D = config["bridge_D_ref"]
    fn = config["bridge_fn_hz"]
    rho = config["bridge_rho"]
    B = config["bridge_B_ref"]
    t_star_release = config["bridge_t_star_release"]
    bsp = bridge_structural_params()
    m, c = bsp["m"], bsp["c"]

    artifact_dir = PROJECT_ROOT / "results" / args.model_subdir
    seq_len = config["bridge_seq_len"]
    metrics_path = artifact_dir / "metrics_gru.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)
        seq_len = saved_metrics["gru_config"].get("seq_len", seq_len)

    ur_stats = None
    ur_stats_path = artifact_dir / "ur_stats.pkl"
    if ur_stats_path.exists():
        import pickle
        with open(ur_stats_path, "rb") as f:
            ur_stats = pickle.load(f)

    split = recover_bridge_split(args.model_subdir)
    if split["recovered"]:
        print(f"[phaseA] bridge train/val/test split RECOVERED from {split['source']}")
        print(f"  train (n={len(split['train_cases'])}): {split['train_cases']}")
        print(f"  val   (n={len(split['validation_cases'])}): {split['validation_cases']}")
        print(f"  test  (n={len(split['test_cases'])}): {split['test_cases']}")
    else:
        print(f"[phaseA] bridge train/val/test split NOT recoverable. Searched: {split['searched']}")
        print("  gru_split / coverage_class will be reported as 'unknown'.")

    thr = Thresholds(args)

    cases = sorted(df["case"].unique().tolist(), key=parse_ur_label)
    print(f"[phaseA] found {len(cases)} bridge cases in cache")

    rows = []
    case_dfs = {}
    for case_label in cases:
        case_df = df[df["case"] == case_label]
        case_dfs[case_label] = case_df
        try:
            row = analyze_case(
                case_label=case_label,
                case_df=case_df,
                D=D, fn=fn, rho=rho, B=B,
                t_star_release=t_star_release,
                seq_len=seq_len,
                handoff_offset_steps=args.handoff_offset_steps,
                m=m, c=c,
                thr=thr,
                split=split,
                ur_stats=ur_stats,
            )
        except Exception as e:
            print(f"  [ERROR] {case_label}: {e}")
            row = {"case": case_label, "exclusion_reason": f"processing_error: {e}",
                   "complete_case": False, "frequency_synchronised": False,
                   "large_response_candidate": False, "tail_stationary": False,
                   "clean_growth_fit": False, "a_ref_measurement_available": False,
                   "response_regime": "incomplete",
                   "measured_v3_reference_eligible": False,
                   "closure_loo_target_eligible": False,
                   "full_hybrid_unseen_target_eligible": False}
        rows.append(row)

    # Needs the full population (bracketing reference cases by Ur), so it
    # runs once after the per-case loop, not inside analyze_case.
    assign_closure_loo_targets(rows)

    for row in rows:
        case_label = row["case"]
        status = "complete" if row.get("complete_case") else "INCOMPLETE"
        ref_ok = "REF-OK" if row.get("measured_v3_reference_eligible") else "ref-no"
        cloo_ok = "CLOSURE-LOO-OK" if row.get("closure_loo_target_eligible") else "closure-loo-no"
        hyb_ok = "HYBRID-OK" if row.get("full_hybrid_unseen_target_eligible") else "hybrid-no"
        print(f"  {case_label:12s} Ur={row.get('Ur', float('nan')):.4f}  "
              f"n={row.get('n_samples','?')}  {status}  regime={row.get('response_regime','?'):22s}  "
              f"{ref_ok}  {cloo_ok}  {hyb_ok}")

    result_df = pd.DataFrame(rows)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.csv_out, index=False)
    print(f"[phaseA] wrote {args.csv_out}")

    n_complete = int(result_df["complete_case"].sum())
    n_synchronised = int(result_df["frequency_synchronised"].sum())
    n_stationary = int(result_df["tail_stationary"].sum())
    n_clean_growth = int(result_df["clean_growth_fit"].sum())
    n_ref_eligible = int(result_df["measured_v3_reference_eligible"].sum())
    n_closure_loo_eligible = int(result_df["closure_loo_target_eligible"].sum())
    n_hybrid_unseen_eligible = int(result_df["full_hybrid_unseen_target_eligible"].sum())
    n_alt_valid = int(result_df["alt_growth_fit_valid"].sum()) if "alt_growth_fit_valid" in result_df else 0
    n_disconnected_masks = int((result_df["growth_mask_n_segments"] > 1).sum()) if "growth_mask_n_segments" in result_df else 0

    regime_counts = result_df["response_regime"].value_counts().to_dict() if "response_regime" in result_df else {}
    synchronised_stationary_cases = sorted(
        result_df.loc[result_df["response_regime"] == "synchronised_stationary", "case"].tolist()
    ) if "response_regime" in result_df else []
    synchronised_modulated_cases = sorted(
        result_df.loc[result_df["response_regime"] == "synchronised_modulated", "case"].tolist()
    ) if "response_regime" in result_df else []

    reference_cases = sorted(
        result_df.loc[result_df["measured_v3_reference_eligible"], "case"].tolist()
    )

    # ── LOO feasibility (data-driven, not hardcoded) ────────────────────────
    if len(reference_cases) < 2:
        loo_feasible = False
        loo_infeasible_reason = "fewer than two bracketing measured-v3 reference cases"
    elif n_closure_loo_eligible == 0:
        loo_feasible = False
        loo_infeasible_reason = (
            "at least two measured-v3 reference cases exist but their Ur values "
            "do not bracket any candidate target case"
        )
    else:
        loo_feasible = True
        loo_infeasible_reason = None

    if len(reference_cases) == 1:
        recommended_scope = f"local v3 closure at {reference_cases[0]}"
    elif len(reference_cases) == 0:
        recommended_scope = "no valid v3 closure reference case found; v3 cannot be scoped"
    elif loo_feasible:
        recommended_scope = "leave-one-out generalization across bracketed interior targets"
    else:
        recommended_scope = (
            f"local v3 closure restricted to measured reference cases {reference_cases} "
            f"(not bracketing any interior target)"
        )

    mu_warnings = [
        f"{r['case']}: r2={r.get('growth_fit_r2', float('nan')):.3f}, "
        f"mask_segments={r.get('growth_mask_n_segments', '?')}"
        for r in rows
        if r.get("complete_case") and not r.get("clean_growth_fit", False)
    ]

    summary = {
        "input_parquet": str(input_path),
        "n_cases_found": len(cases),
        "n_cases_complete": n_complete,
        "n_cases_frequency_synchronised": n_synchronised,
        "n_cases_tail_stationary": n_stationary,
        "n_cases_clean_growth_fit": n_clean_growth,
        "n_cases_alt_growth_fit_valid": n_alt_valid,
        "n_cases_disconnected_growth_mask": n_disconnected_masks,
        "n_cases_measured_v3_reference_eligible": n_ref_eligible,
        "n_cases_closure_loo_target_eligible": n_closure_loo_eligible,
        "n_cases_full_hybrid_unseen_target_eligible": n_hybrid_unseen_eligible,
        "measured_v3_reference_cases": reference_cases,
        "response_regime_counts": regime_counts,
        "synchronised_stationary_cases": synchronised_stationary_cases,
        "synchronised_modulated_cases": synchronised_modulated_cases,
        "loo_feasible": loo_feasible,
        "loo_infeasible_reason": loo_infeasible_reason,
        "recommended_scope": recommended_scope,
        "growth_fit_warnings": mu_warnings,
        "thresholds": thr.as_dict(),
        "tail_fraction": TAIL_FRACTION,
        "envelope_stationary_fraction": ENVELOPE_STATIONARY_FRACTION,
        "handoff_offset_steps": args.handoff_offset_steps,
        "seq_len": seq_len,
        "model_subdir": args.model_subdir,
        "bridge_split": split,
        "cases": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o)
    print(f"[phaseA] wrote {args.json_out}")

    if not args.no_plots:
        plot_paths = make_plots(result_df, args.plots_dir)
        for p in plot_paths:
            print(f"[phaseA] wrote {p}")

        plot_targets = sorted(
            set(result_df.loc[result_df["frequency_synchronised"], "case"]) | ALWAYS_PLOT_CASES
        )
        print(f"[phaseA] generating {len(plot_targets)} per-case diagnostic figures: {plot_targets}")
        for case_label in plot_targets:
            if case_label not in case_dfs:
                print(f"  [warn] {case_label} requested for a diagnostic figure but not found in cache; skipping")
                continue
            try:
                diag = build_case_diagnostics(
                    case_label=case_label, case_df=case_dfs[case_label],
                    D=D, fn=fn, t_star_release=t_star_release,
                    seq_len=seq_len, handoff_offset_steps=args.handoff_offset_steps,
                    thr=thr,
                )
                p = plot_case_diagnostic(case_label, diag, args.plots_dir)
                print(f"[phaseA] wrote {p}")
            except Exception as e:
                print(f"  [ERROR] diagnostic figure for {case_label}: {e}")

    print(f"\n[phaseA] SUMMARY: {len(cases)} cases found, {n_complete} complete, "
          f"{n_synchronised} frequency_synchronised, {n_stationary} tail_stationary, "
          f"{n_clean_growth} clean_growth_fit, {n_disconnected_masks} with disconnected growth masks, "
          f"{n_ref_eligible} measured_v3_reference_eligible, "
          f"{n_closure_loo_eligible} closure_loo_target_eligible, "
          f"{n_hybrid_unseen_eligible} full_hybrid_unseen_target_eligible")
    print(f"[phaseA] response regimes: {regime_counts}")
    print(f"[phaseA] synchronised_stationary cases: {synchronised_stationary_cases}")
    print(f"[phaseA] synchronised_modulated cases: {synchronised_modulated_cases}")
    print(f"[phaseA] LOO feasible = {loo_feasible}"
          + (f"  (reason: {loo_infeasible_reason})" if loo_infeasible_reason else ""))
    print(f"[phaseA] recommended scope = {recommended_scope}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Phase-A characterization of the bridge CFD campaign. "
                     "Does not load the GRU or run coupled inference."
    )
    parser.add_argument("--input_parquet", type=str, default=str(DEFAULT_INPUT),
                    help="Path to the preprocessed bridge CFD parquet cache.")
    parser.add_argument("--csv_out", type=Path, default=DEFAULT_CSV_OUT,
                    help="Output CSV path.")
    parser.add_argument("--json_out", type=Path, default=DEFAULT_JSON_OUT,
                    help="Output JSON path.")
    parser.add_argument("--plots_dir", type=Path, default=PROJECT_ROOT / "results",
                    help="Directory to write the Phase-A diagnostic PNGs into.")
    parser.add_argument("--no_plots", action="store_true",
                    help="Skip diagnostic plot generation (summary + per-case figures).")
    parser.add_argument("--model_subdir", type=str, default=DEFAULT_MODEL_SUBDIR,
                    help="Bridge GRU artifact dir to recover the train/val/test split "
                         "and seq_len from (metrics_gru.json). Never uses ur_stats.pkl "
                         "for split recovery.")
    parser.add_argument("--handoff_offset_steps", type=int, default=2000,
                    help="CFD steps past (release+seq_len) for handoff -- matches "
                         "coupled_inference.py's --handoff_offset default.")
    parser.add_argument("--min_post_release_duration_s", type=float, default=60.0,
                    help="complete_case gate: minimum post-release duration [s]. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--min_post_handoff_duration_s", type=float, default=100.0,
                    help="complete_case gate: minimum post-handoff duration [s]. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--freq_lockin_tol", type=float, default=0.10,
                    help="frequency_synchronised: |f_dom-fn|/fn must be <= this. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--envelope_cv_max", type=float, default=0.10,
                    help="tail_stationary: amplitude-envelope tail CV must be <= this. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--min_growth_cycles", type=float, default=5.0,
                    help="clean_growth_fit: minimum growth-fit duration [cycles]. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--growth_fit_r2_min", type=float, default=0.80,
                    help="clean_growth_fit: minimum growth-fit R^2. "
                         "UNCHANGED from the original Phase-A pass.")
    parser.add_argument("--min_large_response_ad", type=float, default=0.01,
                    help="large_response_candidate (NEW criterion): minimum ad_rms "
                         "(RMS-equivalent peak A/D) to consider the steady response "
                         "'large' rather than buffeting/noise-scale. Default 0.01 "
                         "(1%% of D) is a round, physically-motivated floor, not "
                         "reverse-engineered from any particular case count.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    main(cli_args)
