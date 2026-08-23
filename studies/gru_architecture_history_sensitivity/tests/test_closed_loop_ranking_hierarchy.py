"""
Locks down the predeclared closed-loop ranking hierarchy in
select_configuration.py's _closed_loop_rank_cols/build_pareto_table,
frozen BEFORE the validation arrays were run:

  1. number of stable validation responses (maximize)
  2. median absolute amplitude error (minimize)
  3. worst-case amplitude error (minimize)
  4. frequency error (minimize)

The whole point of this ordering is "do not choose one fortunate seed or
the lowest single validation error" -- these tests build configurations
that are deliberately best on every lower-priority metric while losing on
a higher-priority one, and assert they still rank below a competitor that
wins the higher-priority tier.
"""
import pandas as pd

from select_configuration import build_pareto_table

_CYLINDER_BASE = {
    "dataset": "cylinder200", "stage": 1, "seq_len": 1000, "history_label": None,
    "n_seeds": 3, "seeds": [123, 456, 789], "all_valid": True,
    "open_loop_val_r2_median": 0.99, "open_loop_val_r2_iqr": 0.0,
    "trainable_parameter_count_median": 10000.0,
    "closed_loop_mean_inference_time_s_median": 5.0,
}


def _cyl_row(H, L, stable, median_amp, worst_amp, freq):
    return {
        **_CYLINDER_BASE, "hidden_size": H, "num_layers": L,
        "closed_loop_validation_stable_count_median": stable,
        "closed_loop_validation_median_abs_rel_amp_error_median": median_amp,
        "closed_loop_validation_worst_abs_rel_amp_error_median": worst_amp,
        "closed_loop_validation_median_abs_f_osc_rel_error_median": freq,
    }


def test_hierarchy_orders_by_stable_count_then_median_then_worst_then_freq():
    # A (64,2) is also the required baseline for the retention-delta check
    # -- all rows share open_loop_val_r2_median so the retention gate is a
    # no-op tie here, isolating the closed-loop hierarchy itself.
    agg = pd.DataFrame([
        _cyl_row(64, 2, stable=4, median_amp=0.02, worst_amp=0.05, freq=0.01),   # A
        _cyl_row(32, 1, stable=2, median_amp=0.001, worst_amp=0.001, freq=0.001),  # B: wins every metric except stability
        _cyl_row(32, 2, stable=4, median_amp=0.05, worst_amp=0.05, freq=0.01),   # C: ties A on stability, loses on median amp
        _cyl_row(128, 1, stable=4, median_amp=0.02, worst_amp=0.10, freq=0.01),  # D: ties A through median amp, loses on worst
        _cyl_row(128, 2, stable=4, median_amp=0.02, worst_amp=0.05, freq=0.05),  # E: ties A through worst, loses only on freq
    ])
    ranked = build_pareto_table("cylinder200", agg)
    order = list(zip(ranked["hidden_size"].tolist(), ranked["num_layers"].tolist()))
    assert order == [(64, 2), (128, 2), (128, 1), (32, 2), (32, 1)], (
        "expected A > E > D > C > B: B must rank LAST despite having the "
        f"single lowest amplitude/frequency error of any row, got {order}")


def test_stable_count_alone_beats_a_much_lower_amplitude_error():
    """Isolates tier 1 vs tiers 2-4 collapsed into one dominant competitor,
    matching the task's own framing most directly: a configuration with
    fewer stable validation responses must lose even if its amplitude
    error is an order of magnitude better."""
    agg = pd.DataFrame([
        _cyl_row(64, 2, stable=4, median_amp=0.15, worst_amp=0.15, freq=0.15),
        _cyl_row(32, 1, stable=1, median_amp=0.001, worst_amp=0.001, freq=0.001),
    ])
    ranked = build_pareto_table("cylinder200", agg)
    top = ranked.iloc[0]
    assert (int(top["hidden_size"]), int(top["num_layers"])) == (64, 2)


_BRIDGE_BASE = {
    "dataset": "bridge", "stage": 1, "seq_len": 2500, "history_label": None,
    "n_seeds": 3, "seeds": [123, 456, 789], "all_valid": True,
    "open_loop_val_r2_median": 0.99, "open_loop_val_r2_iqr": 0.0,
    "trainable_parameter_count_median": 10000.0,
    "closed_loop_mean_inference_time_s_median": 5.0,
}


def _bridge_row(H, L, stable, median_amp, worst_amp, freq):
    return {
        **_BRIDGE_BASE, "hidden_size": H, "num_layers": L,
        "closed_loop_validation_stable_count_among_settled_lco_median": stable,
        "closed_loop_validation_median_abs_rel_amp_error_among_settled_lco_median": median_amp,
        "closed_loop_validation_worst_abs_rel_amp_error_among_settled_lco_median": worst_amp,
        "closed_loop_validation_median_abs_f_osc_rel_error_median": freq,
    }


def test_bridge_ranking_reads_the_settled_lco_column_variants():
    """Bridge's amplitude-gate framework only scores settled_lco cases, so
    the ranking must read the _among_settled_lco columns. A silent typo/
    rename mismatch would make _closed_loop_rank_cols return an empty
    list (filtered out by "if c in agg.columns"), degrading to no
    closed-loop ranking at all rather than raising -- this test is placed
    to catch exactly that by requiring an actual reordering, not merely
    an input order that happens to already match.
    """
    agg = pd.DataFrame([
        # Listed less-stable-but-better-errors FIRST: if the settled_lco
        # columns were not found, sort_values would have nothing to
        # reorder on and this row would stay on top -- the assertion
        # below only passes if a real reorder happened.
        _bridge_row(32, 1, stable=2, median_amp=0.01, worst_amp=0.01, freq=0.01),
        _bridge_row(64, 2, stable=5, median_amp=0.10, worst_amp=0.20, freq=0.05),
    ])
    ranked = build_pareto_table("bridge", agg)
    order = list(zip(ranked["hidden_size"].tolist(), ranked["num_layers"].tolist()))
    assert order == [(64, 2), (32, 1)], (
        f"bridge ranking must prioritize settled-LCO stable count over "
        f"amplitude/frequency error, same as cylinder200, got {order}")
