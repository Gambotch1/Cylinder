from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from viv_analysis.config import config, prepare_gru_config
from viv_analysis.coupled_inference import Newmark_beta
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.train_gru import train_one_epoch, train_rollout_epoch, split_cases
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label


BASELINE_DIR = PROJECT_ROOT / "results" / "gru_cylinder_re_1000"


def make_model(input_size: int, hidden_size: int = 8) -> VIV_GRU:
    torch.manual_seed(1234)
    model = VIV_GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
    )
    return model


def make_baseline_loader(batch_size: int = 4, seq_len: int = 6, n_feat: int = 3):
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, n_feat)
    y = torch.randn(batch_size)
    cases = [f"case_{i}" for i in range(batch_size)]
    return [(x, y, cases)]


def make_rollout_loader(
    batch_size: int = 4,
    seq_len: int = 6,
    n_feat: int = 3,
    rollout_k: int = 1,
    u_value: float = 0.32,
):
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, n_feat)
    y = torch.randn(batch_size, rollout_k)
    cases = [f"case_{i}" for i in range(batch_size)]
    U = torch.full((batch_size,), u_value)
    return [(x, y, cases, U)]


def test_baseline_artifacts_match_curriculum_assumptions():
    metrics_path = BASELINE_DIR / "metrics_gru.json"
    ur_stats_path = BASELINE_DIR / "ur_stats.pkl"

    assert metrics_path.exists(), "baseline metrics_gru.json is missing"
    assert ur_stats_path.exists(), "baseline ur_stats.pkl is missing"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    with open(ur_stats_path, "rb") as f:
        ur_stats = pickle.load(f)

    assert metrics["gru_config"]["input_cols"] == ["disp", "vel", "acc"]
    assert metrics["gru_config"]["seq_len"] > 0
    assert metrics["gru_config"]["hidden_size"] > 0
    assert isinstance(ur_stats["use_ur_context"], bool)

    # If the baseline used Ur context, the curriculum runner must match it.
    if ur_stats["use_ur_context"]:
        assert metrics["gru_config"]["use_ur_context"] is True
        assert metrics["gru_config"]["input_cols"] == ["disp", "vel", "acc"]


def test_rollout_k1_matches_train_one_epoch():
    device = "cpu"
    x, y, cases = make_baseline_loader()[0]
    # Use the exact same 3-tuple loader for both paths so y shape, data, and
    # random state are identical — the only difference is which function is called.
    loader = [(x, y, cases)]

    model_a = make_model(input_size=3)
    model_b = make_model(input_size=3)
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))

    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    base_loss = train_one_epoch(
        model=model_a,
        loader=loader,
        optimizer=opt_a,
        criterion=criterion,
        device=device,
    )

    rollout_loss = train_rollout_epoch(
        model=model_b,
        loader=loader,
        optimizer=opt_b,
        criterion=criterion,
        device=device,
        rollout_k=1,
        fn=2.0,
        dt=0.005,
        m=0.2513,
        c=0.004421,
        k_phys=0.3969,
        idx_h=0,
        idx_hdot=1,
        idx_hddot=2,
        expected_features=3,
        x_mean=[0.] * 3,
        x_scale=[1.] * 3,
        y_mean=0.0,
        y_scale=1.0,
        rho=1.0,
        D=1.0,
        max_grad_norm=1.0,
    )

    assert rollout_loss == pytest.approx(base_loss, abs=1e-7, rel=1e-7)


def test_rollout_k2_runs_and_returns_finite_loss():
    device = "cpu"
    torch.manual_seed(0)
    batch_size, seq_len, n_feat = 4, 6, 3
    x = torch.randn(batch_size, seq_len, n_feat)
    y = torch.randn(batch_size, 2)                  # (batch, rollout_k)
    cases = ["Ur4"] * batch_size                    # valid Ur labels for parse_ur_label
    loader = [(x, y, cases)]                        # 3-tuple, matches loader contract

    model = make_model(input_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    loss = train_rollout_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        rollout_k=2,
        dt=0.005,
        m=0.2513,
        c=0.004421,
        k_phys=0.3969,
        fn=0.2,
        idx_h=0,
        idx_hdot=1,
        idx_hddot=2,
        expected_features=3,
        x_mean=[0.] * 3,
        x_scale=[1.] * 3,
        y_mean=0.0,
        y_scale=1.0,
        rho=1.0,
        D=0.4,
        max_grad_norm=1.0,
    )

    assert isinstance(loss, float)
    assert loss > 0.0
    import math
    assert math.isfinite(loss)


def test_newmark_beta_accepts_gradients():
    F = torch.tensor([0.001], dtype=torch.float32, requires_grad=True)
    h = torch.tensor([0.05], dtype=torch.float32)
    h_dot = torch.tensor([0.001], dtype=torch.float32)
    h_ddot = torch.tensor([-0.01], dtype=torch.float32)

    h_new, v_new, a_new = Newmark_beta(F, h, h_dot, h_ddot, 0.005, 0.2513, 0.004421, 0.3969)
    loss = h_new.sum()
    loss.backward()

    assert F.grad is not None
    assert torch.isfinite(F.grad).all()
    assert h_new.shape == F.shape
    assert v_new.shape == F.shape
    assert a_new.shape == F.shape


def test_release_time_covers_training_and_validation_cases():
    cfg = prepare_gru_config("cylinder1000", config)
    # Must include every case that _cylinder1000_split references in its
    # hardcoded test/val sets ("Ur4.75", "Ur5.5", "Ur7" were missing before).
    raw_cases = [
        "Ur2", "Ur3", "Ur4", "Ur4.25", "Ur4.75",
        "Ur5", "Ur5.25", "Ur5.5", "Ur6.5", "Ur7",
        "Ur8", "Ur9", "Ur10", "Ur11", "Ur12",
    ]
    train_cases, val_cases, test_cases, release_time = split_cases(raw_cases, "cylinder1000", cfg)

    assert set(train_cases).isdisjoint(val_cases)
    assert set(train_cases).isdisjoint(test_cases)
    assert set(val_cases).isdisjoint(test_cases)

    for case in train_cases | val_cases | test_cases:
        assert case in release_time


def test_ur_value_computation_matches_case_labels():
    assert parse_ur_label("Ur4") == pytest.approx(4.0)
    assert parse_ur_label("Ur5.5") == pytest.approx(5.5)
    assert parse_ur_label("Ur11") == pytest.approx(11.0)