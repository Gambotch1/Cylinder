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

    "cylinder1000_D_ref":     0.4,   # Re=1000 cylinder diameter [m]
    "cylinder1000_dt":        0.005, # Re=1000 timestep [s]
    "cylinder1000_t_release": 16.0,  # physical release time [s]
    "cylinder1000_U_inf":     1.0,   # freestream velocity [m/s]
    "cylinder1000_zeta":      0.007,  # structural damping ratio (assumed)
    "cylinder1000_rho" :      1.0,
    "cylinder1000_fn":        0.2,
    "cylinder1000_M_star":    2.0,

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
    "hl_range":      [500, 2500],
    "hl_step":       100,
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

    out = cfg.copy()
    ds  = dataset.strip().lower()

    if ds == "cylinder":
        out["seq_len"]      = 900
        out["stride_train"] = 3
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = False

    elif ds in {"cylinder1000", "cylinder_re_1000", "re1000","re1000_disp","re1000_vel","re1000_acc"}:
        out["seq_len"]      = 1000
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

def structural_params() -> dict:
    rho = config['cylinder1000_rho']
    D   = config['cylinder1000_D_ref']
    fn  = config['cylinder1000_fn']

    M_star = config['cylinder1000_M_star']
    zeta   = config['cylinder1000_zeta']
    m = M_star * rho * (np.pi * D**2 / 4.0)
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta


    params_Re1000 = {
            "m": m,                 # kg/m (legacy key)
            "c": c,              # N*s/m (legacy key)
            "k": k,                    # N/m (legacy key)
            "cylinder_mass": m,    # kg/m
            "c_struct": c,       # N*s/m
            "k_struct": k,             # N/m
        }
    
    return params_Re1000