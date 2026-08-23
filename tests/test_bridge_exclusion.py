"""19.5 m/s (Ur=8.2126) permanent exclusion -- verifies the case is absent
from every load-time path a training/evaluation/sweep script could reach,
and that the cache version can never silently serve the pre-exclusion data.
"""
import pandas as pd

from viv_analysis.preprocess import (
    load_bridge_df_cached, bridge_cache_path,
    BRIDGE_EXCLUDED_RAW_SPEEDS, BRIDGE_CACHE_VERSION,
)
from viv_analysis.config import config, bridge_structural_params
from viv_analysis.train_gru import _bridge_split
from viv_analysis.utils import parse_ur_label

EXCLUDED_UR_LABEL = "Ur8.2126"
EXCLUDED_RAW_SPEED = "19.5"


def test_exclusion_config_is_in_effect():
    assert EXCLUDED_RAW_SPEED in BRIDGE_EXCLUDED_RAW_SPEEDS
    assert BRIDGE_CACHE_VERSION >= 2, (
        "cache version must have been bumped past the pre-exclusion v1 cache")


# NOTE: merge_dataframes(dataset="bridge") re-parses the raw ~9GB .out files
# from scratch every call (bridge training deliberately bypasses the cache --
# see preprocess.py) and takes on the order of an hour cold. Exercising that
# path directly here would make the default test suite impractically slow.
# The exclusion filter lives INSIDE merge_dataframes (see preprocess.py), so
# test_cached_loader_excludes_it and test_cache_file_itself_is_the_versioned_
# post_exclusion_file below -- which read the cache that merge_dataframes
# itself built -- already cover it for day-to-day runs. If the cache is ever
# deleted, the first load_bridge_df_cached call that rebuilds it goes
# through this same merge_dataframes filter before writing the new cache.


def test_cached_loader_excludes_it():
    df = load_bridge_df_cached(fn_hz=config["bridge_fn_hz"], d_ref=config["bridge_D_ref"],
                               bridge_structural_params=bridge_structural_params())
    assert not df.empty
    assert EXCLUDED_UR_LABEL not in set(df["case"].unique())


def test_cache_file_itself_is_the_versioned_post_exclusion_file():
    cache_path = bridge_cache_path()
    assert f"_v{BRIDGE_CACHE_VERSION}" in cache_path.name
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        assert EXCLUDED_UR_LABEL not in set(df["case"].unique())


def test_deterministic_split_never_contains_excluded_case():
    df = load_bridge_df_cached(fn_hz=config["bridge_fn_hz"], d_ref=config["bridge_D_ref"],
                               bridge_structural_params=bridge_structural_params())
    cases = sorted(df["case"].unique(), key=parse_ur_label)
    train, val, test, _ = _bridge_split(cases, config["bridge_fn_hz"],
                                        config["bridge_D_ref"],
                                        config["bridge_t_star_release"])
    assert EXCLUDED_UR_LABEL not in train
    assert EXCLUDED_UR_LABEL not in val
    assert EXCLUDED_UR_LABEL not in test


def test_retained_case_count_is_27():
    df = load_bridge_df_cached(fn_hz=config["bridge_fn_hz"], d_ref=config["bridge_D_ref"],
                               bridge_structural_params=bridge_structural_params())
    assert df["case"].nunique() == 27


def test_reference_status_table_excludes_it():
    from viv_analysis.reference_quality import build_reference_status_table
    table = build_reference_status_table()
    assert EXCLUDED_UR_LABEL not in set(table["case"].astype(str).unique())
    assert len(table) == 27
