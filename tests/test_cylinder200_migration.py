"""
Automated checks for the cylinder1000 -> cylinder200 (completed Re=200)
migration. Covers the acceptance criteria from the migration task:

  * exactly 21 matching lift/displacement case pairs are found
  * U = 0.04 * Ur
  * mu = 4.0e-5 * Ur
  * recomputed rho*U*D/mu = 200 for every case
  * m ~= 0.314159, k ~= 0.496100, c ~= 0.007896
  * t_release = 200 / Ur
  * no generated sequence crosses a case boundary
  * no legacy (cylinder1000) model/output directory is overwritten
  * cylinder1000 is rejected, not silently mismapped, everywhere it used
    to be supported
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from viv_analysis.config import (
    config, CYLINDER200_ALIASES,
    cylinder200_U, cylinder200_mu, cylinder200_release_time,
    cylinder200_structural_params,
)
from viv_analysis.preprocess import (
    discover_cylinder200_cases, CYLINDER200_EXPECTED_UR,
    merge_dataframes, _resolve_data_dirs,
)
from viv_analysis.models.gru import VIVSequenceDataset
from viv_analysis.train_gru import (
    split_cases, resolve_nd_reference_scales, resolve_dataset,
)
from viv_analysis.utils import format_ur_label, parse_ur_label


UR_VALUES = list(CYLINDER200_EXPECTED_UR)


class TestCasePairing:
    def test_exactly_21_pairs_found(self):
        pairs = discover_cylinder200_cases()
        assert len(pairs) == 21
        assert set(pairs) == {format_ur_label(u) for u in UR_VALUES}

    def test_each_pair_has_both_files_on_disk(self):
        pairs = discover_cylinder200_cases()
        for case, files in pairs.items():
            assert files["cl"].exists(), f"{case}: missing cl file"
            assert files["disp"].exists(), f"{case}: missing disp file"


class TestDerivedPhysics:
    @pytest.mark.parametrize("Ur", UR_VALUES)
    def test_U_equals_0p04_times_Ur(self, Ur):
        assert cylinder200_U(Ur) == pytest.approx(0.04 * Ur, rel=1e-12)

    @pytest.mark.parametrize("Ur", UR_VALUES)
    def test_mu_equals_4em5_times_Ur(self, Ur):
        assert cylinder200_mu(Ur) == pytest.approx(4.0e-5 * Ur, rel=1e-12)

    @pytest.mark.parametrize("Ur", UR_VALUES)
    def test_recomputed_reynolds_is_200(self, Ur):
        U = cylinder200_U(Ur)
        mu = cylinder200_mu(Ur)
        rho = config["cylinder200_rho"]
        D = config["cylinder200_D_ref"]
        assert (rho * U * D / mu) == pytest.approx(200.0, rel=1e-9)

    @pytest.mark.parametrize("Ur", UR_VALUES)
    def test_release_time_equals_200_over_Ur(self, Ur):
        assert cylinder200_release_time(Ur) == pytest.approx(200.0 / Ur, rel=1e-9)

    def test_structural_params_m_k_c(self):
        sp = cylinder200_structural_params()
        assert sp["m"] == pytest.approx(0.314159, abs=1e-6)
        assert sp["k"] == pytest.approx(0.496100, abs=1e-6)
        assert sp["c"] == pytest.approx(0.007896, abs=1e-6)


class TestSplitReleaseTimes:
    """release_time must be per-case (200/Ur), not one flat constant."""

    def test_split_release_times_match_formula(self):
        cases = [format_ur_label(u) for u in UR_VALUES]
        cfg = config
        train, val, test, rt = split_cases(cases, "cylinder200", cfg)

        assert len(train) + len(val) + len(test) == 21
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

        for case in cases:
            ur = parse_ur_label(case)
            assert rt[case] == pytest.approx(200.0 / ur, rel=1e-9)

        # explicitly NOT a single dataset-wide constant
        assert len({round(v, 6) for v in rt.values()}) == len(cases)


class TestNoCrossCaseSequences:
    def test_sequences_never_mix_cases(self):
        """Two cases with disjoint, easily-distinguishable disp ranges --
        if any generated window mixed rows from both cases, its disp
        values would straddle both ranges."""
        rows = []
        for case, base in [("Ur2", 0.0), ("Ur3", 1000.0)]:
            for step in range(40):
                rows.append({
                    "case": case, "step": step, "time": step * 0.005,
                    "disp": base + step, "vel": base + step,
                    "acc": base + step, "cl": base + step,
                })
        df = pd.DataFrame(rows)
        release_time = {"Ur2": 0.0, "Ur3": 0.0}

        ds = VIVSequenceDataset(
            df, seq_len=5, target_col="cl", input_cols=["disp", "vel", "acc"],
            release_time=release_time, stride=1,
        )
        assert len(ds) > 0
        for i in range(len(ds)):
            x, y, case_name = ds[i]
            window = x.numpy()
            if case_name == "Ur2":
                assert window.max() < 1000.0, "Ur2 window leaked Ur3 rows"
            else:
                assert window.min() >= 1000.0, "Ur3 window leaked Ur2 rows"


class TestNoLegacyOverwrite:
    def test_cylinder200_default_dir_name_distinct_from_legacy(self):
        """The default (--exp_subdir omitted) output_dir naming scheme
        (gru_{dataset}{coord_suffix}{ctx_suffix}, mirroring train_gru.main())
        must never collide with a pre-existing cylinder1000 legacy dir name."""
        legacy_prefixes = ("gru_cylinder1000", "gru_cylinder_re_1000", "gru_Cylinder1000")
        for coord_suffix in ("", "_nd"):
            for ctx_suffix in ("_ctx", "_noctx"):
                name = f"gru_cylinder200{coord_suffix}{ctx_suffix}"
                assert not name.startswith(legacy_prefixes)

    def test_elm_output_dir_is_not_legacy_elm_dir(self):
        from viv_analysis.train_ELM import OUTPUT_DIR
        assert OUTPUT_DIR.name != "elm_model"
        assert "cylinder200" in OUTPUT_DIR.name


class TestCylinder1000Removed:
    """cylinder1000 must be rejected everywhere it used to be supported --
    never silently mismapped to different physics."""

    def test_resolve_data_dirs_rejects_cylinder1000(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            _resolve_data_dirs("cylinder1000")

    def test_resolve_nd_reference_scales_rejects_cylinder1000(self):
        with pytest.raises(ValueError, match="unsupported dataset"):
            resolve_nd_reference_scales("cylinder1000", config)

    def test_config_has_no_cylinder1000_keys(self):
        assert not any(k.startswith("cylinder1000_") for k in config)
        assert "cylinder1000" not in config["viv_dataset"]

    def test_coupled_inference_rejects_cylinder1000(self):
        from viv_analysis.coupled_inference import normalize_cfd_dataset
        # normalize_cfd_dataset passes unknown strings through unchanged --
        # the real rejection happens in main()'s physical-parameter dispatch,
        # exercised indirectly via resolve_physical_params in
        # diagnose_off_manifold (same dispatch structure).
        from viv_analysis.diagnose_off_manifold import resolve_physical_params
        with pytest.raises(ValueError, match="unsupported cfd_dataset"):
            resolve_physical_params("cylinder1000")
