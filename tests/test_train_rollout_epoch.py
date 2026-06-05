import copy
import pytest
import torch

from viv_analysis.models.gru import VIV_GRU
# Make sure to import your training functions correctly
from viv_analysis.train_gru import train_one_epoch, train_rollout_epoch


def make_model(input_size: int = 3) -> VIV_GRU:
    torch.manual_seed(1234)
    return VIV_GRU(
        input_size=input_size,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


# FIX 1: Add rollout_k to the mock data generator
def make_batch(batch_size: int = 4, seq_len: int = 6, n_feat: int = 3, rollout_k: int = 1):
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, n_feat)
    
    # y_seq must have shape [batch_size, rollout_k]
    y_seq = torch.randn(batch_size, rollout_k) 
    
    # train_one_epoch expects a 1D tensor [batch_size] for y. 
    # To make the k=1 test mathematically equivalent, y must be exactly the first column of y_seq.
    y = y_seq[:, 0] 
    
    cases = [f"case_{i}" for i in range(batch_size)]
    U = torch.full((batch_size,), 2.5)
    return x, y, y_seq, cases, U


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# Goal 1: Critical test ensuring rollout_k=1 exactly matches baseline train_one_epoch
def test_train_rollout_epoch_rollout1_matches_baseline():
    device = "cpu"
    # Pass rollout_k=1
    x, y, y_seq, cases, U = make_batch(rollout_k=1)

    model_base = make_model()
    model_roll = make_model()
    # Ensure identical initial weights
    model_roll.load_state_dict(copy.deepcopy(model_base.state_dict()))

    optimizer_base = make_optimizer(model_base)
    optimizer_roll = make_optimizer(model_roll)

    # Reduction must be 'mean' (the default) to scale correctly
    criterion = torch.nn.MSELoss() 

    baseline_loader = [(x, y, cases)]
    rollout_loader = [(x, y_seq, cases, U)]

    x_mean = torch.zeros(3)
    x_scale = torch.ones(3)
    y_mean = torch.zeros(1)
    y_scale = torch.ones(1)

    baseline_loss = train_one_epoch(
        model=model_base,
        loader=baseline_loader,
        optimizer=optimizer_base,
        criterion=criterion,
        device=device,
    )

    rollout_loss = train_rollout_epoch(
        model=model_roll,
        loader=rollout_loader,
        optimizer=optimizer_roll,
        criterion=criterion,
        device=device,
        rollout_k=1,
        dt=0.005,
        m=0.2513,
        c=0.004421,
        k_phys=0.3969,
        idx_h=0,
        idx_hdot=1,
        idx_hddot=2,
        expected_features=3,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        rho=1.0,
        D=1.0,
    )

    # This asserts that turning off the physics loop reduces EXACTLY to standard Teacher Forcing
    assert rollout_loss == pytest.approx(baseline_loss, abs=1e-7)


# Goal 2: Ensure shapes align correctly for multi-step physics integration
def test_train_rollout_epoch_prints_debug_shapes(capfd):
    device = "cpu"
    
    # FIX 2: Explicitly pass rollout_k=2 to generate correct mock data shapes
    rollout_k = 2 
    x, y, y_seq, cases, U = make_batch(rollout_k=rollout_k)

    model = make_model()
    optimizer = make_optimizer(model)
    criterion = torch.nn.MSELoss()

    rollout_loader = [(x, y_seq, cases, U)]

    x_mean = torch.zeros(3)
    x_scale = torch.ones(3)
    y_mean = torch.zeros(1)
    y_scale = torch.ones(1)

    _ = train_rollout_epoch(
        model=model,
        loader=rollout_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        rollout_k=rollout_k,
        dt=0.005,
        m=0.2513,
        c=0.004421,
        k_phys=0.3969,
        idx_h=0,
        idx_hdot=1,
        idx_hddot=2,
        expected_features=3,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        rho=1.0,
        D=1.0,
    )

    # capfd will now successfully read the output because the function won't crash on the assertion
    out = capfd.readouterr().out
    assert "--- SHAPE DEBUG ---" in out
    assert "pred_cl_norm:" in out
    assert "h_phys:" in out
    assert "U_b:" in out
    assert "F_phys:" in out