import numpy as np
import pytest

from viv_analysis.open_loop_energy_diagnostics import (
    phase_difference_rad, blockwise_energy_and_phase, detect_disagreement_onset,
)

RHO, U, B, C = 1.225, 16.0, 25.9, 989.39


def _sinusoid_result(cl_fn, f=0.3, T=60.0, dt=0.002):
    t = np.arange(0, T, dt)
    h = 0.1 * np.sin(2 * np.pi * f * t)
    h_dot = 0.1 * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
    cl = cl_fn(t, f)
    return dict(t=t, h=h, h_dot=h_dot, cl_cfd=cl, cl_pred=cl, hidden_norm=np.ones_like(t))


def test_in_phase_cl_hdot_gives_large_positive_work_and_zero_phase():
    result = _sinusoid_result(lambda t, f: 0.2 * np.cos(2 * np.pi * f * t))  # in phase with hdot
    df = blockwise_energy_and_phase(result, RHO, U, B, C, block_duration_s=20.0)
    assert (df["W_f_cfd"] > df["W_d"]).all()
    assert np.abs(df["phase_cl_hdot_cfd_rad"]).max() < 1e-6


def test_quadrature_cl_hdot_gives_near_zero_work_and_90deg_phase():
    result = _sinusoid_result(lambda t, f: 0.2 * np.sin(2 * np.pi * f * t))  # in phase with h, not hdot
    df = blockwise_energy_and_phase(result, RHO, U, B, C, block_duration_s=20.0)
    assert (df["W_f_cfd"].abs() < 1e-3 * df["W_d"]).all()
    assert df["phase_cl_hdot_cfd_rad"].apply(lambda p: abs(abs(p) - np.pi / 2) < 1e-6).all()


def test_anti_phase_cl_hdot_gives_large_negative_work():
    result = _sinusoid_result(lambda t, f: -0.2 * np.cos(2 * np.pi * f * t))  # anti-phase with hdot
    df = blockwise_energy_and_phase(result, RHO, U, B, C, block_duration_s=20.0)
    assert (df["W_f_cfd"] < -df["W_d"]).all()


def test_blockwise_splits_into_expected_number_of_blocks():
    result = _sinusoid_result(lambda t, f: 0.2 * np.cos(2 * np.pi * f * t), T=61.0)
    df = blockwise_energy_and_phase(result, RHO, U, B, C, block_duration_s=20.0)
    assert len(df) == 4  # 0-20, 20-40, 40-60, 60-61 (partial)


def test_disagreement_onset_detects_sustained_divergence():
    t = np.arange(0, 60, 0.002)
    cl_true = 0.2 * np.cos(2 * np.pi * 0.3 * t)
    cl_pred = cl_true.copy()
    onset_sample = int(50 / 0.002)
    cl_pred[onset_sample:] += 5.0
    onset = detect_disagreement_onset(t, cl_pred, cl_true, threshold_std_mult=2.0, sustain_window_s=5.0)
    assert onset is not None
    assert abs(onset - 50.0) < 0.1


def test_disagreement_onset_none_when_signals_match():
    t = np.arange(0, 60, 0.002)
    cl_true = 0.2 * np.cos(2 * np.pi * 0.3 * t)
    assert detect_disagreement_onset(t, cl_true, cl_true) is None


def test_phase_difference_rad_wraps_to_pi_range():
    t = np.arange(0, 60, 0.002)
    f = 0.3
    x1 = np.cos(2 * np.pi * f * t)
    x2 = np.cos(2 * np.pi * f * t + np.pi * 0.9)
    phase = phase_difference_rad(t, x1, x2, f)
    assert -np.pi <= phase <= np.pi
