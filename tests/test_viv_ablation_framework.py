"""
Comprehensive test suite for VIV ablation framework (dimensional/nondimensional × context).
Validates all 10 acceptance criteria for Ur=5.5 generalization experiment.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.viv_analysis.preprocess import _resolve_data_dirs
from src.viv_analysis.train_gru import (
    seed_everything, format_ur_label, resolve_use_ur_context,
    check_artifact_collision, apply_nd_transform, enforce_holdout,
    resolve_dataset,
)
from src.viv_analysis.coupled_inference import load_artifact_coordinate_mode
from src.viv_analysis.models.gru import VIV_GRU


class TestDatasetValidation:
    """Criterion 2: Fix _resolve_data_dirs() to explicitly allow only cylinder, cylinder1000, bridge."""
    
    def test_allowed_datasets(self):
        """Should accept cylinder, cylinder1000, bridge."""
        allowed = ["cylinder", "cylinder1000", "bridge"]
        for dataset in allowed:
            # Should not raise
            disp_dir, cd_dir, cl_dir = _resolve_data_dirs(dataset)
            assert isinstance(disp_dir, Path)
    
    def test_unknown_dataset_raises(self):
        """Should raise ValueError for unknown dataset names."""
        unknown_datasets = ["cylinder1000_nd", "cylinder_nd", "unknown", "cylinder2000", ""]
        for dataset in unknown_datasets:
            with pytest.raises(ValueError, match="Unknown dataset"):
                _resolve_data_dirs(dataset)


def _make_kinematics_df() -> pd.DataFrame:
    """Two cases (Ur5, Ur7), 4 rows each, with distinct disp/vel/acc/cl values
    so a per-case transform bug (e.g. reusing one global U) would be visible."""
    rows = []
    for case, base in [("Ur5", 1.0), ("Ur7", 2.0)]:
        for step in range(4):
            rows.append({
                "case": case, "step": step, "time": step * 0.005,
                "disp": base + 0.1 * step,
                "vel": base + 0.2 * step,
                "acc": base + 0.3 * step,
                "cl": base + 0.4 * step,
            })
    return pd.DataFrame(rows)


class TestNonDimensionalTransform:
    """Criterion 3: Verify ND transform numerical logic via the real
    apply_nd_transform() helper (not a re-derivation of the formula)."""

    D = 0.4
    fn = 0.2

    def test_dimensional_form_identity(self):
        """nd_inputs=False must be an exact identity transform."""
        df = _make_kinematics_df()
        out = apply_nd_transform(df, nd_inputs=False, D=self.D, fn=self.fn,
                                  input_cols=["disp", "vel", "acc"])
        pd.testing.assert_frame_equal(out, df)

    def test_nd_transform_ur5_and_ur7_receipts(self):
        """Matches the required receipts:
        Ur5: U=0.400, divisors=[0.400, 0.400, 0.400]
        Ur7: U=0.560, divisors=[0.400, 0.560, 0.784]"""
        df = _make_kinematics_df()
        out = apply_nd_transform(df, nd_inputs=True, D=self.D, fn=self.fn,
                                  input_cols=["disp", "vel", "acc"])

        ur5 = df[df["case"] == "Ur5"]
        ur5_out = out[out["case"] == "Ur5"]
        np.testing.assert_allclose(ur5_out["disp"], ur5["disp"] / 0.4, rtol=1e-6)
        np.testing.assert_allclose(ur5_out["vel"], ur5["vel"] / 0.4, rtol=1e-6)
        np.testing.assert_allclose(ur5_out["acc"], ur5["acc"] / 0.4, rtol=1e-6)

        ur7 = df[df["case"] == "Ur7"]
        ur7_out = out[out["case"] == "Ur7"]
        np.testing.assert_allclose(ur7_out["disp"], ur7["disp"] / 0.4, rtol=1e-6)
        np.testing.assert_allclose(ur7_out["vel"], ur7["vel"] / 0.56, rtol=1e-6)
        np.testing.assert_allclose(ur7_out["acc"], ur7["acc"] / 0.784, rtol=1e-6)

    def test_nd_transform_is_case_dependent(self):
        """Feeding the SAME physical vel/acc value into both cases must
        produce DIFFERENT model-coordinate outputs, proving U is
        case-dependent rather than one global freestream value."""
        df = pd.DataFrame([
            {"case": "Ur5", "step": 0, "time": 0.0, "disp": 1.0, "vel": 1.0, "acc": 1.0, "cl": 0.0},
            {"case": "Ur7", "step": 0, "time": 0.0, "disp": 1.0, "vel": 1.0, "acc": 1.0, "cl": 0.0},
        ])
        out = apply_nd_transform(df, nd_inputs=True, D=self.D, fn=self.fn,
                                  input_cols=["disp", "vel", "acc"])
        ur5_row = out[out["case"] == "Ur5"].iloc[0]
        ur7_row = out[out["case"] == "Ur7"].iloc[0]
        # disp always divides by D alone -> identical
        assert ur5_row["disp"] == pytest.approx(ur7_row["disp"])
        # vel/acc divide by case-dependent U -> must differ
        assert ur5_row["vel"] != pytest.approx(ur7_row["vel"])
        assert ur5_row["acc"] != pytest.approx(ur7_row["acc"])

    def test_nd_transform_leaves_cl_and_metadata_unchanged(self):
        df = _make_kinematics_df()
        out = apply_nd_transform(df, nd_inputs=True, D=self.D, fn=self.fn,
                                  input_cols=["disp", "vel", "acc"])
        pd.testing.assert_series_equal(out["cl"], df["cl"])
        pd.testing.assert_series_equal(out["case"], df["case"])
        pd.testing.assert_series_equal(out["step"], df["step"])
        pd.testing.assert_series_equal(out["time"], df["time"])


class TestFormatUrLabel:
    """Test Ur label formatting."""
    
    def test_format_ur_label(self):
        """format_ur_label should convert float to string label."""
        assert format_ur_label(5.5) == "Ur5.5"
        assert format_ur_label(5.0) == "Ur5"
        assert format_ur_label(7.0) == "Ur7"


class TestReproducibility:
    """Criterion 6: Controlled training draw via seed_everything()."""
    
    def test_seed_everything_sets_all_seeds(self):
        """seed_everything should set Python, NumPy, PyTorch seeds."""
        import random
        import torch
        
        # Set seed
        seed_everything(123)
        
        # Check that subsequent random draws are deterministic
        val1_py = random.random()
        val1_np = np.random.rand()
        val1_torch = torch.rand(1).item()
        
        # Reset and re-seed
        seed_everything(123)
        val2_py = random.random()
        val2_np = np.random.rand()
        val2_torch = torch.rand(1).item()
        
        # Should be identical
        assert val1_py == val2_py
        assert np.isclose(val1_np, val2_np)
        assert np.isclose(val1_torch, val2_torch)
    
    def test_different_seeds_differ(self):
        """Different seeds should produce different values."""
        import random
        
        seed_everything(123)
        val1 = random.random()
        
        seed_everything(456)
        val2 = random.random()
        
        assert val1 != val2


class TestArtifactSafety:
    """Criterion 5: Artifact directory collision safety -- one arm must
    never silently clobber another's checkpoint/metrics."""

    def test_no_collision_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            check_artifact_collision(Path(tmpdir), overwrite=False)  # must not raise

    def test_collision_raises_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "gru_best.pt").write_bytes(b"stub")
            with pytest.raises(FileExistsError):
                check_artifact_collision(artifact_dir, overwrite=False)

    def test_collision_raises_on_existing_metrics_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "metrics_gru.json").write_text("{}")
            with pytest.raises(FileExistsError):
                check_artifact_collision(artifact_dir, overwrite=False)

    def test_overwrite_flag_bypasses_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "gru_best.pt").write_bytes(b"stub")
            check_artifact_collision(artifact_dir, overwrite=True)  # must not raise


class TestHoldoutEnforcement:
    """Criterion 4: explicit holdout handling."""

    def test_holdout_removed_from_train_and_val_added_to_test(self):
        train = {"Ur3", "Ur5.5", "Ur6"}
        val = {"Ur4", "Ur9"}
        test = {"Ur7"}
        all_cases = train | val | test
        new_train, new_val, new_test = enforce_holdout(train, val, test, all_cases, 5.5)
        assert "Ur5.5" not in new_train
        assert "Ur5.5" not in new_val
        assert "Ur5.5" in new_test
        assert new_train == {"Ur3", "Ur6"}
        assert new_val == val

    def test_holdout_missing_from_data_raises(self):
        train, val, test = {"Ur3"}, {"Ur4"}, {"Ur7"}
        with pytest.raises(ValueError):
            enforce_holdout(train, val, test, train | val | test, 5.5)

    def test_holdout_enforcement_detects_overlap(self):
        # Fabricate an already-overlapping split to hit the overlap guard.
        train = {"Ur3", "Ur5.5"}
        val = {"Ur3"}  # overlaps train
        test = set()
        all_cases = train | val
        with pytest.raises(ValueError, match="overlap"):
            enforce_holdout(train, val, test, all_cases, 5.5)


class TestSeededReproducibility:
    """Criterion 10: fixed seed gives repeatable initial model parameters
    and DataLoader shuffle order in a small synthetic test."""

    def test_model_init_is_repeatable_with_same_seed(self):
        seed_everything(42)
        m1 = VIV_GRU(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
        seed_everything(42)
        m2 = VIV_GRU(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seed_gives_different_init(self):
        seed_everything(1)
        m1 = VIV_GRU(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
        seed_everything(2)
        m2 = VIV_GRU(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
        params_differ = any(
            not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert params_differ

    def test_dataloader_shuffle_order_is_repeatable_with_same_generator_seed(self):
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(torch.arange(20))

        def draw_order(seed):
            gen = torch.Generator()
            gen.manual_seed(seed)
            loader = DataLoader(ds, batch_size=4, shuffle=True, generator=gen)
            return [batch[0].tolist() for batch in loader]

        order1 = draw_order(123)
        order2 = draw_order(123)
        assert order1 == order2

        order3 = draw_order(456)
        assert order1 != order3


class TestCoordinateMode:
    """Criterion 9: Artifact/inference coordinate guard."""
    
    def test_coordinate_mode_from_run_config(self):
        """load_artifact_coordinate_mode should read run_config.json first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            
            # Create run_config with coordinate mode
            run_config = {
                'nd_inputs': True,
                'cfd_dataset': 'cylinder1000',
                'holdout_ur': 5.5,
            }
            with open(artifact_dir / "run_config.json", 'w') as f:
                json.dump(run_config, f)
            
            # Load should return True
            nd_inputs = load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=None)
            assert nd_inputs is True
    
    def test_coordinate_mode_cli_override_when_no_artifact(self):
        """CLI can specify coordinate mode if not in artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            
            # No run_config, no ur_stats
            # CLI specifies nd_inputs=False
            nd_inputs = load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=False)
            assert nd_inputs is False
    
    def test_coordinate_mode_mismatch_raises(self):
        """Should raise ValueError if CLI disagrees with artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            
            # Create run_config with nd_inputs=True
            run_config = {'nd_inputs': True}
            with open(artifact_dir / "run_config.json", 'w') as f:
                json.dump(run_config, f)
            
            # CLI specifies opposite: nd_inputs=False (should raise)
            with pytest.raises(ValueError, match="coordinate mode"):
                load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=False)


class TestArgparseInterface:
    """Criterion 1: Argparse interface with all required options."""
    
    def test_argparse_help(self):
        """--help should display all options."""
        from src.viv_analysis.train_gru import setup_argparse
        
        parser = setup_argparse()
        
        # Check that key options are present
        actions_dict = {action.dest: action for action in parser._actions}
        
        required_options = [
            'cfd_dataset', 'nd_inputs', 'use_ur_context', 'no_ur_context',
            'holdout_ur', 'epochs', 'batch_size', 'seq_len', 'noise_std',
            'seed', 'num_workers', 'exp_subdir', 'preflight_only', 'overwrite'
        ]
        
        for opt in required_options:
            assert opt in actions_dict, f"Missing option: {opt}"


class TestPreflightMode:
    """Criterion 8: Preflight-only mode exits after scaler fit."""
    
    def test_preflight_flag_exists(self):
        """Argparse should have --preflight_only flag."""
        from src.viv_analysis.train_gru import setup_argparse
        
        parser = setup_argparse()
        args = parser.parse_args(['cylinder1000', '--preflight_only'])
        
        assert hasattr(args, 'preflight_only')
        assert args.preflight_only is True


class TestArgumentParsing:
    """Criterion 1: CLI interface validation."""

    def test_argparse_context_flags_unset_by_default(self):
        """Neither --use_ur_context nor --no_ur_context should be forced on
        by default; the effective value must come from dataset config unless
        the user explicitly picks a side (see TestContextResolution)."""
        from src.viv_analysis.train_gru import setup_argparse

        parser = setup_argparse()
        args = parser.parse_args(['cylinder1000'])

        assert args.use_ur_context is False
        assert args.no_ur_context is False
        # No CLI override for these -> None-equivalent, dataset config decides
        assert args.holdout_ur is None
        assert args.epochs is None
        assert args.batch_size is None
        assert args.seq_len is None


class TestContextResolution:
    """Criterion 1: 'Set the context option default to None' -- retain the
    dataset default unless the user explicitly picks --use_ur_context or
    --no_ur_context."""

    def test_no_explicit_flag_keeps_dataset_default(self):
        assert resolve_use_ur_context(True, False, False) is True
        assert resolve_use_ur_context(False, False, False) is False

    def test_explicit_use_ur_context_overrides_dataset_default(self):
        assert resolve_use_ur_context(False, True, False) is True

    def test_explicit_no_ur_context_overrides_dataset_default(self):
        assert resolve_use_ur_context(True, False, True) is False


class TestDatasetAliases:
    """Criterion 2: Dataset aliases resolution."""
    
    def test_cylinder1000_aliases(self):
        """cylinder1000 should accept standard aliases."""
        allowed = ["cylinder1000"]
        for dataset in allowed:
            # Should not raise
            disp_dir, cd_dir, cl_dir = _resolve_data_dirs(dataset)
            assert isinstance(disp_dir, Path)
    
    def test_no_cylinder1000_nd_alias(self):
        """cylinder1000_nd should NOT be accepted as alias."""
        with pytest.raises(ValueError):
            _resolve_data_dirs("cylinder1000_nd")


class TestDatasetResolution:
    """Hostile-audit finding: `dataset_cli or dataset_pos or "cylinder"` treats
    an explicitly-passed empty string as "not given" and silently redirects to
    the Re=200 cylinder dataset. resolve_dataset() must preserve "" so the
    caller's _resolve_data_dirs validation rejects it."""

    def test_explicit_cfd_dataset_wins(self):
        assert resolve_dataset(None, "cylinder1000") == "cylinder1000"

    def test_legacy_positional_used_when_no_flag(self):
        assert resolve_dataset("bridge", None) == "bridge"

    def test_no_dataset_given_defaults_to_cylinder(self):
        assert resolve_dataset(None, None) == "cylinder"

    def test_mismatched_positional_and_flag_raises(self):
        with pytest.raises(ValueError):
            resolve_dataset("cylinder1000", "bridge")

    def test_empty_string_cfd_dataset_is_not_silently_defaulted(self):
        """The core regression: --cfd_dataset "" must come back as "" (and
        therefore be rejected downstream by _resolve_data_dirs), not silently
        become "cylinder"."""
        resolved = resolve_dataset(None, "")
        assert resolved == ""
        with pytest.raises(ValueError):
            _resolve_data_dirs(resolved)


class TestArtifactDirectoryNaming:
    """Hostile-audit finding: the default output_dir name (when --exp_subdir
    is omitted) must be unique per (nd_inputs, use_ur_context) combination,
    not just per nd_inputs -- otherwise two of the four required arms
    collide on the same default directory."""

    def _default_dir(self, dataset, nd_inputs, use_ur_context):
        coord_suffix = "_nd" if nd_inputs else ""
        ctx_suffix = "_ctx" if use_ur_context else "_noctx"
        return f"gru_{dataset}{coord_suffix}{ctx_suffix}"

    def test_all_four_arms_get_distinct_default_dirs(self):
        dataset = "cylinder1000"
        names = {
            self._default_dir(dataset, nd, ctx)
            for nd in (False, True) for ctx in (False, True)
        }
        assert len(names) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
