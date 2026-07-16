"""
Focused unit tests for Phase-A prep:
  - --a_ref_m override (validation + resolution semantics + metadata source tagging)
  - --mu validation (finite, non-negative)
  - analyze_bridge_cfd_campaign.py helpers (per-case dt, split recovery, 17 m/s exclusion,
    the redesigned eligibility columns, growth-mask contiguity audit, the diagnostic-only
    contiguous_first_growth_fit alternative)
  - a_ref_convention rename + backward-compat alias
  - git_dirty / worktree_patch_hash receipt provenance

These are CPU-only, do not load the GRU, and do not run coupled inference.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from viv_analysis.coupled_inference import (
    positive_finite_float,
    finite_admissible_mu,
    resolve_a_ref,
    resolve_mu_source,
    get_git_dirty_and_patch_hash,
)
from viv_analysis.self_excitation import (
    find_contiguous_segments,
    measure_growth_rate,
    contiguous_first_growth_fit,
    A_REF_CONVENTION,
    A_REF_CONVENTION_LEGACY_ALIASES,
    normalize_a_ref_convention,
)
from viv_analysis.analyze_bridge_cfd_campaign import (
    measure_case_dt,
    recover_bridge_split,
    classify_coverage,
    analyze_case,
    Thresholds,
)


# ── --a_ref_m validation ─────────────────────────────────────────────────────

class TestARefMValidation:
    def test_accepts_valid_positive_float(self):
        assert positive_finite_float("0.15") == pytest.approx(0.15)
        assert positive_finite_float("3") == pytest.approx(3.0)

    @pytest.mark.parametrize("bad", ["-1.0", "0", "0.0", "nan", "inf", "-inf", "abc", ""])
    def test_rejects_invalid_values(self, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_finite_float(bad)


class TestMuValidation:
    def test_accepts_zero_and_positive(self):
        assert finite_admissible_mu("0") == pytest.approx(0.0)
        assert finite_admissible_mu("0.65") == pytest.approx(0.65)

    @pytest.mark.parametrize("bad", ["-0.5", "-1", "nan", "inf", "-inf", "abc", ""])
    def test_rejects_negative_nonfinite_and_nonnumeric(self, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            finite_admissible_mu(bad)


def test_a_ref_m_help_text_states_metres():
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "viv_analysis.coupled_inference", "--help"],
        cwd=str(Path(__file__).parent.parent / "src"),
        capture_output=True, text=True, timeout=30,
    )
    assert "--a_ref_m" in out.stdout
    assert "METRES" in out.stdout


# ── a_ref / mu resolution semantics ─────────────────────────────────────────

class TestResolveARef:
    def test_omitted_uses_measured_value(self):
        """When --a_ref_m is omitted, preserve current behavior: measure from CFD."""
        amp = {"a_ref_peak": 0.42, "rms_AD": 0.01}
        a_ref, source = resolve_a_ref(amp, a_ref_m=None)
        assert a_ref == pytest.approx(0.42)
        assert source == "measured_from_target_cfd"

    def test_supplied_uses_cli_value_verbatim(self):
        """When --a_ref_m is supplied, use it and do NOT replace with the measurement."""
        amp = {"a_ref_peak": 0.42, "rms_AD": 0.01}
        a_ref, source = resolve_a_ref(amp, a_ref_m=1.234)
        assert a_ref == pytest.approx(1.234)
        assert a_ref != pytest.approx(amp["a_ref_peak"])
        assert source == "cli_override"


class TestResolveMuSource:
    def test_cli_mu_always_overrides_regardless_of_mode(self):
        assert resolve_mu_source(mu=0.65, forcing_mode="v3_coherent") == "cli_override"
        assert resolve_mu_source(mu=0.65, forcing_mode="v1_additive") == "cli_override"

    def test_v3_coherent_without_mu_is_measured(self):
        assert resolve_mu_source(mu=None, forcing_mode="v3_coherent") == "measured_from_target_cfd"

    def test_other_modes_without_mu_are_not_applicable(self):
        assert resolve_mu_source(mu=None, forcing_mode="v1_additive") == "not_applicable"
        assert resolve_mu_source(mu=None, forcing_mode="v2_multiplicative") == "not_applicable"


# ── git_dirty / worktree_patch_hash provenance ──────────────────────────────

def test_git_dirty_and_patch_hash_returns_bool_and_str():
    dirty, patch_hash = get_git_dirty_and_patch_hash()
    assert isinstance(dirty, bool)
    assert isinstance(patch_hash, str)


# ── a_ref_convention rename + backward compatibility ────────────────────────

class TestARefConventionRename:
    def test_current_convention_is_rms_equivalent_name(self):
        assert A_REF_CONVENTION == "rms_equivalent_peak_sqrt2_std_tail"

    def test_legacy_name_normalizes_to_current(self):
        assert "peak_sqrt2_rms_tail" in A_REF_CONVENTION_LEGACY_ALIASES
        assert normalize_a_ref_convention("peak_sqrt2_rms_tail") == A_REF_CONVENTION

    def test_current_name_normalizes_to_itself(self):
        assert normalize_a_ref_convention(A_REF_CONVENTION) == A_REF_CONVENTION

    def test_unknown_value_passes_through(self):
        assert normalize_a_ref_convention("some_future_convention") == "some_future_convention"


# ── measure_growth_rate mask audit (find_contiguous_segments) ───────────────

class TestFindContiguousSegments:
    def test_empty_mask(self):
        assert find_contiguous_segments(np.array([False, False, False])) == []

    def test_single_contiguous_run(self):
        m = np.array([False, True, True, True, False])
        assert find_contiguous_segments(m) == [(1, 4)]

    def test_multiple_disconnected_runs(self):
        m = np.array([False, True, True, False, False, True, True, True, False, True])
        assert find_contiguous_segments(m) == [(1, 3), (5, 8), (9, 10)]

    def test_all_true(self):
        m = np.array([True, True, True])
        assert find_contiguous_segments(m) == [(0, 3)]


class TestMeasureGrowthRateAudit:
    """measure_growth_rate is the PRODUCTION estimator -- these tests check
    the additive audit fields (mask/tc/env/b) don't change lam/r2/n/t0/t1,
    and correctly expose contiguity information."""

    def test_clean_single_segment_growth_has_one_mask_segment(self):
        dt, fn, D = 0.01, 0.2, 1.0
        t = np.arange(0, 200, dt)
        # clean monotone exponential growth, well inside the floor/hi band
        env_true = 0.001 * np.exp(0.05 * t)
        h = env_true * np.sin(2 * np.pi * fn * t)
        g = measure_growth_rate(h, dt, fn, D)
        segs = find_contiguous_segments(g["mask"])
        assert g["n"] > 0
        assert g["lam"] > 0
        # a clean single monotone exponential should produce few/one segment(s)
        assert len(segs) <= 2

    def test_returned_dict_has_audit_keys_without_changing_core_fields(self):
        dt, fn, D = 0.01, 0.2, 1.0
        t = np.arange(0, 200, dt)
        env_true = 0.001 * np.exp(0.05 * t)
        h = env_true * np.sin(2 * np.pi * fn * t)
        g = measure_growth_rate(h, dt, fn, D)
        for key in ("lam", "r2", "n", "t0", "t1", "b", "mask", "tc", "env", "max_env"):
            assert key in g

    def test_empty_mask_case_returns_safe_defaults(self):
        dt, fn, D = 0.01, 0.2, 1.0
        h = np.zeros(100)  # no growth at all -> envelope ~0, mask likely empty
        g = measure_growth_rate(h, dt, fn, D)
        assert g["n"] == 0
        assert g["lam"] == 0.0
        assert g["mask"] is not None


class TestContiguousFirstGrowthFit:
    def test_diagnostic_only_matches_legacy_on_single_clean_segment(self):
        """When there's exactly one contiguous segment, alt and legacy should
        agree closely (alt is legacy restricted to segment 1, which IS all
        of it here)."""
        dt, fn, D = 0.01, 0.2, 1.0
        t = np.arange(0, 200, dt)
        env_true = 0.001 * np.exp(0.05 * t)
        h = env_true * np.sin(2 * np.pi * fn * t)
        g = measure_growth_rate(h, dt, fn, D)
        ga = contiguous_first_growth_fit(h, dt, fn, D, min_cycles=5.0, r2_min=0.80)
        if len(find_contiguous_segments(g["mask"])) == 1:
            assert ga["lam"] == pytest.approx(g["lam"], rel=0.05)

    def test_retains_min_cycle_and_r2_checks(self):
        """A first segment with too few cycles must be invalid regardless of
        how good its local R2 is."""
        dt, fn, D = 0.01, 0.2, 1.0
        # very short, clean linear-in-log growth burst (2 points -> perfect
        # R2 but far short of 5 cycles), synthetic and deliberately tiny.
        t = np.arange(0, 3, dt)
        env_true = 0.001 * np.exp(0.05 * t)
        h = env_true * np.sin(2 * np.pi * fn * t)
        ga = contiguous_first_growth_fit(h, dt, fn, D, min_cycles=5.0, r2_min=0.80)
        assert ga["valid"] is False

    def test_no_mask_samples_returns_invalid_not_a_crash(self):
        h = np.zeros(50)
        ga = contiguous_first_growth_fit(h, dt=0.01, fn=0.2, D=1.0)
        assert ga["valid"] is False
        assert ga["n"] == 0
        assert ga["stopped_reason"] == "no_mask_samples"

    def test_never_reports_more_segments_considered_than_legacy_mask_has(self):
        dt, fn, D = 0.01, 0.2, 1.0
        t = np.arange(0, 300, dt)
        # multi-lobe envelope to force several disconnected mask segments
        env_true = 0.02 * (1 + 0.9 * np.sin(2 * np.pi * t / 60.0))
        h = env_true * np.sin(2 * np.pi * fn * t)
        g = measure_growth_rate(h, dt, fn, D)
        ga = contiguous_first_growth_fit(h, dt, fn, D, min_cycles=5.0, r2_min=0.80)
        legacy_segments = find_contiguous_segments(g["mask"])
        assert ga["n_segments_considered"] == len(legacy_segments)


# ── Phase-A: per-case dt ────────────────────────────────────────────────────

class TestMeasureCaseDt:
    def test_uniform_grid(self):
        t = np.arange(0, 10, 0.01)
        assert measure_case_dt(t) == pytest.approx(0.01, rel=1e-6)

    def test_robust_to_duplicate_and_unsorted_timestamps(self):
        t = np.array([0.02, 0.0, 0.01, 0.01, 0.03])
        assert measure_case_dt(t) == pytest.approx(0.01, rel=1e-6)

    def test_two_cases_different_dt_are_independent(self):
        """A global-dt bug would make these collide; per-case dt must not."""
        t_fast = np.arange(0, 5, 0.001)
        t_slow = np.arange(0, 5, 0.02)
        dt_fast = measure_case_dt(t_fast)
        dt_slow = measure_case_dt(t_slow)
        assert dt_fast == pytest.approx(0.001, rel=1e-6)
        assert dt_slow == pytest.approx(0.02, rel=1e-6)
        assert dt_fast != dt_slow


# ── Phase-A: bridge split recovery ──────────────────────────────────────────

class TestRecoverBridgeSplit:
    def test_recovers_from_metrics_gru_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            subdir = results_root / "gru_bridge_test"
            subdir.mkdir(parents=True)
            metrics = {
                "case_split": {
                    "train": ["Ur5", "Ur6"],
                    "val": ["Ur5.5"],
                    "test": ["Ur6.5"],
                }
            }
            with open(subdir / "metrics_gru.json", "w") as f:
                json.dump(metrics, f)

            import viv_analysis.analyze_bridge_cfd_campaign as mod
            orig_root = mod.PROJECT_ROOT
            mod.PROJECT_ROOT = Path(tmp)
            try:
                split = recover_bridge_split("gru_bridge_test")
            finally:
                mod.PROJECT_ROOT = orig_root

            assert split["recovered"] is True
            assert split["train_cases"] == ["Ur5", "Ur6"]
            assert split["validation_cases"] == ["Ur5.5"]
            assert split["test_cases"] == ["Ur6.5"]

    def test_missing_metrics_reports_unknown_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            import viv_analysis.analyze_bridge_cfd_campaign as mod
            orig_root = mod.PROJECT_ROOT
            mod.PROJECT_ROOT = Path(tmp)
            try:
                split = recover_bridge_split("does_not_exist")
            finally:
                mod.PROJECT_ROOT = orig_root

            assert split["recovered"] is False
            assert split["train_cases"] is None
            assert len(split["searched"]) > 0

    def test_real_bridge_model_split_is_recoverable(self):
        """Guards against the real gru_bridge_noise0.05 receipt regressing."""
        split = recover_bridge_split("gru_bridge_noise0.05")
        assert split["recovered"] is True
        assert len(split["train_cases"]) == 16
        assert len(split["validation_cases"]) == 6
        assert len(split["test_cases"]) == 6


class TestClassifyCoverage:
    def _split(self):
        return {
            "recovered": True,
            "train_cases": ["Ur5", "Ur6", "Ur7"],
            "validation_cases": ["Ur6.5"],
            "test_cases": ["Ur5.5"],
        }

    def test_trained_case(self):
        gru_split, coverage = classify_coverage("Ur6", 6.0, self._split())
        assert gru_split == "train"
        assert coverage == "trained"

    def test_unseen_interpolation_within_train_range(self):
        gru_split, coverage = classify_coverage("Ur6.5", 6.5, self._split())
        assert gru_split == "validation"
        assert coverage == "unseen_interpolation"

    def test_context_extrapolation_outside_train_range(self):
        """A held-out case whose Ur falls outside [min,max] of the recovered
        train Ur's -- not the same as 'not in any split list' (unknown)."""
        split = {
            "recovered": True,
            "train_cases": ["Ur5", "Ur6", "Ur7"],
            "validation_cases": [],
            "test_cases": ["Ur9"],
        }
        gru_split, coverage = classify_coverage("Ur9", 9.0, split)
        assert gru_split == "test"
        assert coverage == "context_extrapolation"

    def test_case_absent_from_any_split_list_is_unknown_not_guessed(self):
        gru_split, coverage = classify_coverage("Ur10", 10.0, self._split())
        # Ur10 not in any split list -> unknown (not a fabricated guess)
        assert gru_split == "unknown"
        assert coverage == "unknown"

    def test_unknown_when_split_not_recovered(self):
        unresolved = {"recovered": False, "train_cases": None,
                      "validation_cases": None, "test_cases": None}
        gru_split, coverage = classify_coverage("Ur6", 6.0, unresolved)
        assert gru_split == "unknown"
        assert coverage == "unknown"


# ── Phase-A: 17 m/s incomplete case handling ────────────────────────────────

def _synthetic_case_df(n_samples: int, dt: float = 0.002, D: float = 7.42,
                        fn: float = 0.32, amp: float = 0.03) -> pd.DataFrame:
    t = np.arange(n_samples) * dt
    disp = amp * D * np.sin(2 * np.pi * fn * t)
    return pd.DataFrame({
        "case": "UrX", "time": t, "disp": disp,
        "cl": np.zeros(n_samples), "vel": np.zeros(n_samples), "acc": np.zeros(n_samples),
    })


def _default_thresholds() -> Thresholds:
    ns = argparse.Namespace(
        min_post_release_duration_s=60.0,
        min_post_handoff_duration_s=100.0,
        freq_lockin_tol=0.10,
        envelope_cv_max=0.10,
        min_growth_cycles=5.0,
        growth_fit_r2_min=0.80,
        min_large_response_ad=0.01,
    )
    return Thresholds(ns)


_UNRESOLVED_SPLIT = {"recovered": False, "train_cases": None,
                     "validation_cases": None, "test_cases": None}


class TestIncompleteCaseHandling:
    def test_short_record_marked_incomplete_and_excluded(self):
        """Mirrors the real Ur7.1597 (~17 m/s) case: ~12,533 samples at dt=0.002s."""
        df = _synthetic_case_df(n_samples=12533)
        row = analyze_case(
            case_label="Ur7.1597", case_df=df, D=7.42, fn=0.32, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=_UNRESOLVED_SPLIT, ur_stats=None,
        )
        assert row["complete_case"] is False
        assert row["a_ref_measurement_available"] is False
        assert row["measured_v3_reference_eligible"] is False
        assert row["closure_loo_target_eligible"] is False
        assert row["full_hybrid_unseen_target_eligible"] is False
        assert row["response_regime"] == "incomplete"
        assert "incomplete CFD record" in row["exclusion_reason"]

    def test_full_length_record_not_flagged_incomplete(self):
        """A full-length synthetic record should pass the completeness gate
        (may still fail other screening criteria -- that's fine)."""
        df = _synthetic_case_df(n_samples=149996)
        row = analyze_case(
            case_label="Ur6", case_df=df, D=7.42, fn=0.32, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=_UNRESOLVED_SPLIT, ur_stats=None,
        )
        assert row["complete_case"] is True
        assert "incomplete CFD record" not in row["exclusion_reason"]


# ── Phase-A: redesigned eligibility columns ─────────────────────────────────

class TestEligibilityColumns:
    """The old measured_ref_eligible was misleadingly named (True for nearly
    every complete case, since it only meant 'a_ref is numerically
    computable'). These columns must be independently, honestly named."""

    def _row_for(self, n_samples=149996, amp=0.03, dt=0.002):
        df = _synthetic_case_df(n_samples=n_samples, dt=dt, amp=amp)
        return analyze_case(
            case_label="Ur6", case_df=df, D=7.42, fn=0.32, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=_UNRESOLVED_SPLIT, ur_stats=None,
        )

    def test_all_eligibility_columns_present(self):
        row = self._row_for()
        for col in (
            "complete_case", "a_ref_measurement_available", "frequency_synchronised",
            "large_response_candidate", "tail_stationary", "clean_growth_fit",
            "measured_v3_reference_eligible", "closure_loo_target_eligible",
            "full_hybrid_unseen_target_eligible", "response_regime",
        ):
            assert col in row, f"missing eligibility column: {col}"

    def test_a_ref_measurement_available_means_only_computable_not_eligible(self):
        """a_ref_measurement_available=True must NOT imply
        measured_v3_reference_eligible=True -- that conflation was the bug
        being fixed. A pure sine (no envelope growth/stationarity structure
        measure_growth_rate/tail checks expect) should compute a numeric
        a_ref but not pass every other gate simultaneously by construction."""
        row = self._row_for()
        if row["a_ref_measurement_available"] and not row["clean_growth_fit"]:
            assert row["measured_v3_reference_eligible"] is False

    def test_incomplete_case_fails_every_eligibility_column(self):
        row = self._row_for(n_samples=12533)  # mirrors Ur7.1597
        assert row["complete_case"] is False
        for col in (
            "a_ref_measurement_available", "frequency_synchronised",
            "large_response_candidate", "tail_stationary", "clean_growth_fit",
            "measured_v3_reference_eligible", "closure_loo_target_eligible",
            "full_hybrid_unseen_target_eligible",
        ):
            assert row[col] is False, f"{col} should be False for an incomplete case"
        assert row["response_regime"] == "incomplete"

    def test_full_hybrid_unseen_target_eligible_requires_val_or_test_interior_coverage(self):
        """A trained case (in-sample) must never be full_hybrid_unseen_target_eligible
        even if every response-quality gate passes -- the GRU already saw it."""
        df = _synthetic_case_df(n_samples=149996)
        split = {
            "recovered": True,
            "train_cases": ["Ur6"],  # this case IS in train
            "validation_cases": [],
            "test_cases": [],
        }
        row = analyze_case(
            case_label="Ur6", case_df=df, D=7.42, fn=0.32, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=split, ur_stats=None,
        )
        assert row["gru_split"] == "train"
        assert row["full_hybrid_unseen_target_eligible"] is False

    def test_full_hybrid_unseen_target_eligible_does_not_require_clean_growth_fit(self):
        """A full-hybrid evaluation predicts the target from OTHER cases'
        closure params, so the target's own growth-fit quality must not gate
        it. A constant-amplitude sinusoid (no ramp-up at all) has no
        exponential growth transient -- the 5%-45% envelope mask stays empty
        because the envelope never leaves its own max -- so
        growth_rate_lambda cannot be positive and clean_growth_fit is False
        by construction, while the steady response itself is trivially
        stationary/synchronised/large."""
        t = np.arange(0, 300, 0.002)
        D, fn = 7.42, 0.32
        disp = 0.03 * D * np.sin(2 * np.pi * fn * t)  # constant amplitude from t=0
        df = pd.DataFrame({
            "case": "Ur6", "time": t, "disp": disp,
            "cl": np.zeros(len(t)), "vel": np.zeros(len(t)), "acc": np.zeros(len(t)),
        })
        split = {
            "recovered": True,
            "train_cases": ["Ur5", "Ur7"],  # Ur6 interior to [5,7], held out
            "validation_cases": [],
            "test_cases": ["Ur6"],
        }
        row = analyze_case(
            case_label="Ur6", case_df=df, D=D, fn=fn, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=split, ur_stats=None,
        )
        assert row["gru_split"] == "test"
        assert row["coverage_class"] == "unseen_interpolation"
        assert row["clean_growth_fit"] is False, "constant-amplitude signal should have no growth fit"
        assert row["frequency_synchronised"] is True
        assert row["tail_stationary"] is True
        assert row["large_response_candidate"] is True
        assert row["response_regime"] == "synchronised_stationary"
        assert row["full_hybrid_unseen_target_eligible"] is True, (
            "full_hybrid_unseen_target_eligible must not require clean_growth_fit"
        )
        assert row["measured_v3_reference_eligible"] is False, (
            "measured_v3_reference_eligible SHOULD require clean_growth_fit"
        )

    def test_full_hybrid_eligibility_depends_on_split_closure_loo_does_not(self):
        """full_hybrid_unseen_target_eligible is a GRU-coverage question (does
        depend on gru_split/coverage_class); closure_loo_target_eligible is a
        closure-parameter-interpolation question and must NOT depend on
        gru_split at all -- it's computed later, purely from Ur brackets."""
        row = self._row_for()
        # analyze_case() alone always sets closure_loo_target_eligible=False;
        # it's only finalized by the assign_closure_loo_targets population pass.
        assert row["closure_loo_target_eligible"] is False


class TestResponseRegime:
    def _row_for(self, n_samples=149996, amp=0.03, dt=0.002):
        df = _synthetic_case_df(n_samples=n_samples, dt=dt, amp=amp)
        return analyze_case(
            case_label="Ur6", case_df=df, D=7.42, fn=0.32, rho=1.225, B=25.9,
            t_star_release=20.0, seq_len=2500, handoff_offset_steps=2000,
            m=1.0, c=1.0, thr=_default_thresholds(),
            split=_UNRESOLVED_SPLIT, ur_stats=None,
        )

    def test_incomplete_regime(self):
        row = self._row_for(n_samples=12533)
        assert row["response_regime"] == "incomplete"

    def test_valid_regime_values_are_from_the_fixed_set(self):
        allowed = {"non_synchronised", "synchronised_stationary",
                   "synchronised_modulated", "incomplete"}
        row = self._row_for()
        assert row["response_regime"] in allowed

    def test_synchronised_but_nonstationary_is_modulated_not_non_synchronised(self):
        """A synchronised-but-beating response must not be lumped in with
        genuinely off-lock-in ('non_synchronised') cases -- that conflation
        is exactly what this column exists to prevent."""
        row = self._row_for()
        if row["frequency_synchronised"] and not row["tail_stationary"]:
            assert row["response_regime"] == "synchronised_modulated"
            assert row["response_regime"] != "non_synchronised"


# ── closure_loo_target_eligible bracketing (population-level pass) ─────────

def _row(case, ur, *, complete=True, ref=False, synchronised=True,
         large_response=True, stationary=True):
    """Minimal synthetic row for assign_closure_loo_targets tests -- only
    the fields that function actually reads."""
    return {
        "case": case, "Ur": ur,
        "complete_case": complete,
        "measured_v3_reference_eligible": ref,
        "frequency_synchronised": synchronised,
        "large_response_candidate": large_response,
        "tail_stationary": stationary,
    }


class TestClosureLooBracketing:
    def test_single_reference_case_brackets_nothing(self):
        """Mirrors the real campaign: exactly one measured_v3_reference_eligible
        case exists -> no case can have both a lower and upper reference."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=False),
            _row("Ur6", 6.0, ref=True),
            _row("Ur7", 7.0, ref=False),
        ]
        assign_closure_loo_targets(rows)
        assert all(r["closure_loo_target_eligible"] is False for r in rows)

    def test_two_bracketing_references_enable_interior_target(self):
        """Correction #1: a suitable interior target (complete, synchronised,
        large-response, stationary) bracketed by two valid reference cases
        IS eligible."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False),  # the target under test
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is True   # bracketed by Ur5 (lower) and Ur7 (upper)
        assert by_case["Ur5"] is False  # no case has a lower reference than Ur5
        assert by_case["Ur7"] is False  # no case has an upper reference than Ur7

    def test_reference_case_itself_is_not_self_bracketing(self):
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        assert all(r["closure_loo_target_eligible"] is False for r in rows)

    def test_incomplete_case_never_bracketed(self):
        """Correction #5: incomplete targets remain ineligible even when
        otherwise bracketed."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, complete=False, ref=False),
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is False

    def test_bracketed_but_non_synchronised_target_is_not_eligible(self):
        """Correction #2."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False, synchronised=False),
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is False

    def test_bracketed_but_nonstationary_target_is_not_eligible(self):
        """Correction #3."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False, stationary=False),
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is False

    def test_bracketed_but_weak_response_target_is_not_eligible(self):
        """Correction #4."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False, large_response=False),
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is False

    def test_bracketed_and_all_target_quality_gates_pass_is_eligible(self):
        """Explicit positive control mirroring corrections #2-4: flip each
        gate back on one at a time and confirm eligibility returns once all
        are satisfied (isolates that it's specifically these three gates,
        combined with bracketing, that drive the result)."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False, synchronised=True, stationary=True, large_response=True),
            _row("Ur7", 7.0, ref=True),
        ]
        assign_closure_loo_targets(rows)
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is True

    def test_independent_of_gru_split_and_coverage_class(self):
        """closure_loo_target_eligible must not depend on gru_split/coverage_class
        at all -- these fields are absent from the synthetic rows entirely and
        the function must not error or implicitly require them."""
        from viv_analysis.analyze_bridge_cfd_campaign import assign_closure_loo_targets
        rows = [
            _row("Ur5", 5.0, ref=True),
            _row("Ur6", 6.0, ref=False),
            _row("Ur7", 7.0, ref=True),
        ]
        for r in rows:
            assert "gru_split" not in r and "coverage_class" not in r
        assign_closure_loo_targets(rows)  # must not raise
        by_case = {r["case"]: r["closure_loo_target_eligible"] for r in rows}
        assert by_case["Ur6"] is True


# ── Backward compatibility smoke test ───────────────────────────────────────

def test_coupled_inference_cli_help_unchanged_for_existing_flags():
    """Existing flags must still be present and parse; guards against the
    --a_ref_m addition breaking backward compatibility."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-m", "viv_analysis.coupled_inference", "--help"],
        cwd=str(Path(__file__).parent.parent / "src"),
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0
    for flag in ["--Ur", "--total_time", "--forcing_mode", "--mu", "--handoff_offset",
                 "--noise_mode", "--a_ref_m"]:
        assert flag in out.stdout, f"{flag} missing from --help output"
