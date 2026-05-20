# src/config.py

import numpy as np

config = {
    # ── Dataset reference parameters ──────────────────────────────────────
    "bridge_D_ref":       7.42,   # deck depth [m]
    "bridge_fn_hz":       0.32,   # natural frequency [Hz]
    "bridge_t_star_release": 20.0,
    "bridge_downsample":  10,     # keep every 10th step (dt_eff = 0.002s)
    "bridge_seq_len":     2000,    # after downsampling: ~1.8s of history
    "bridge_stride_train": 5,

    "cylinder_D_ref":     1.0,    # Re=200 cylinder diameter [m]
    "cylinder_dt":        0.02,   # Re=200 timestep [s]
    "cylinder_t_release": 60.0,   # physical release time [s]

    "cylinder1000_D_ref":     0.2,   # Re=1000 cylinder diameter [m]
    "cylinder1000_dt":        0.005, # Re=1000 timestep [s]
    "cylinder1000_t_release": 16.0,  # physical release time [s]
    "cylinder1000_U_inf":     1.0,   # freestream velocity [m/s]
    "cylinder1000_zeta":      0.007,  # structural damping ratio (assumed)

    # ── GRU architecture ───────────────────────────────────────────────────
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.1,

    # ── Training ───────────────────────────────────────────────────────────
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "n_epochs":      100,
    "batch_size":    256,
    "patience":      15,

    # ── Seq len and stride (set per dataset at runtime) ────────────────────
    "seq_len":       None,   # filled by prepare_gru_config()
    "stride_train":  None,
    "stride_val":    1,
    "use_ur_context": False,

    # ── ELM hyperparameter search ──────────────────────────────────────────
    "hl_range":      [50, 400],
    "hl_step":       10,
    "lam_range":     [1e-8, 1e-3],
    "n_trials":      50,
    "n_jobs":        -1,
    "n_models":      15,
    "n_hidden_nodes": 300,
    "alpha_reg":     1.0,
    "n_ensemble":    10,

    # ── Shared ─────────────────────────────────────────────────────────────
    "seed":          123,
    "target_col":    "cl",
    "input_cols":    ["disp", "vel", "acc"],
    "motion_type":   "heave",

    # ── Supported datasets ─────────────────────────────────────────────────
    "viv_dataset":   ["cylinder", "cylinder1000", "bridge"],
}

np.random.seed(config["seed"])


def prepare_gru_config(dataset: str, cfg: dict) -> dict:
    """
    Return a copy of cfg with seq_len and stride_train
    set appropriately for the requested dataset.

    The seq_len target is 2 oscillation cycles at the highest Ur.
    For cylinder (Re=200):  max Ur=9, T=9s,  2T=18s, dt=0.02s  → 900 steps
    For cylinder1000:       max Ur=12, T=2.4s, 2T=4.8s, dt=0.005s → 960 steps
    For bridge (downsampled x10): max Ur≈10.6, T=33s, dt_eff=0.002s → capped 900
    """
    out = cfg.copy()
    ds  = dataset.strip().lower()

    if ds == "cylinder":
        out["seq_len"]      = 900
        out["stride_train"] = 3
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = False

    elif ds in {"cylinder1000", "cylinder_re_1000", "re1000"}:
        out["seq_len"]      = 960
        out["stride_train"] = 8
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = True

    elif ds == "bridge":
        # After 10x downsampling dt_eff=0.002s
        # 2 cycles at Ur≈10.6: T=Ur*D/U≈10.6*7.42/17=4.6s, 2T=9.2s, /0.002=4600
        # Cap at 2000 for memory; covers ~1.8s which is ~0.4 cycles at Ur=10
        out["seq_len"]      = 2000
        out["stride_train"] = 5
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = False

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return out