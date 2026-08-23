# src/config.py

import numpy as np

config = {
    # ── Dataset reference parameters ──────────────────────────────────────
    "bridge_D_ref":       7.42,   # deck depth [m]
    "bridge_B_ref":       25.9,   # deck width [m]
    "bridge_fn_hz":       0.32,   # natural frequency [Hz]
    "bridge_t_star_release": 20.0,
    "bridge_downsample":  20,     # keep every 10th step (dt_eff = 0.002s)
    "bridge_seq_len":     2500,    # after downsampling: ~1.8s of history
    "bridge_stride_train": 5,
    "bridge_zeta":          0.01,  # structural damping ratio (assumed)
    "bridge_rho":           1.225,  # air density [kg/m^3]
    "bridge_mass":          24604.0,    # mass per unit span [kg/m]

    "cylinder_D_ref":     1.0,    # Re=200 cylinder diameter [m]
    "cylinder_dt":        0.02,   # Re=200 timestep [s]
    "cylinder_t_release": 60.0,   # physical release time [s]

    # Re=200 cylinder (completed dataset) -- the only primary parameters;
    # everything else (U, mu, release_time, m, k, c) is DERIVED from these,
    # never duplicated as separate numeric constants elsewhere.
    "cylinder200_D_ref":           0.2,    # diameter [m]
    "cylinder200_rho":             1.0,    # fluid density [kg/m^3]
    "cylinder200_Re":              200.0,  # Reynolds number
    "cylinder200_M_star":          10.0,   # mass ratio
    "cylinder200_zeta":            0.01,   # structural damping ratio
    "cylinder200_fn":              0.2,    # natural frequency [Hz]
    "cylinder200_dt":              0.005,  # timestep [s]
    "cylinder200_t_star_release":  40.0,   # nondimensional release time T*
    "cylinder200_ref_area":        0.2,    # reference area [m^2 per unit span] (D * 1m span)

    # ── GRU architecture ───────────────────────────────────────────────────
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.1,

    # ── Training ───────────────────────────────────────────────────────────
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "n_epochs":      100,
    "batch_size":    512,
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
    "viv_dataset":   ["cylinder", "cylinder200", "bridge"],
}

# Canonical primary key + accepted spellings for the completed Re=200
# dataset. Matching is done against dataset.strip().lower().
CYLINDER200_ALIASES = frozenset({
    "cylinder200", "cylinder_re200", "cylinder_re_200", "re200",
    "cylinder-re-200",
})

np.random.seed(config["seed"])


def prepare_gru_config(dataset: str, cfg: dict) -> dict:

    out = cfg.copy()
    ds  = dataset.strip().lower()

    if ds == "cylinder":
        out["seq_len"]      = 900
        out["stride_train"] = 3
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = False

    elif ds in CYLINDER200_ALIASES:
        # Same dt (0.005s) and multi-Ur/context structure as cylinder1000 --
        # nothing about the input/output schema changed, only the physics,
        # so seq_len/stride/architecture are deliberately kept identical.
        out["seq_len"]      = 1000
        out["stride_train"] = 8
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = True

    elif ds == "bridge":
        out["seq_len"]      = 2500
        out["stride_train"] = 5
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return out

def cylinder200_U(Ur: float) -> float:
    """U(Ur) = Ur * fn * D."""
    return float(Ur) * config["cylinder200_fn"] * config["cylinder200_D_ref"]


def cylinder200_mu(Ur: float) -> float:
    """Dynamic viscosity implied by Re=200 at this Ur: mu = rho*U*D/Re."""
    U = cylinder200_U(Ur)
    return config["cylinder200_rho"] * U * config["cylinder200_D_ref"] / config["cylinder200_Re"]


def cylinder200_release_time(Ur: float) -> float:
    """release_time(Ur) = T_star_release * D / U(Ur)."""
    U = cylinder200_U(Ur)
    return config["cylinder200_t_star_release"] * config["cylinder200_D_ref"] / U


def cylinder200_structural_params() -> dict:
    rho = config["cylinder200_rho"]
    D   = config["cylinder200_D_ref"]
    fn  = config["cylinder200_fn"]

    M_star = config["cylinder200_M_star"]
    zeta   = config["cylinder200_zeta"]
    m = M_star * rho * (np.pi * D**2 / 4.0)
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta

    return {
        "m": m,                 # kg/m (legacy key)
        "c": c,                 # N*s/m (legacy key)
        "k": k,                 # N/m (legacy key)
        "cylinder_mass": m,     # kg/m
        "c_struct": c,          # N*s/m
        "k_struct": k,          # N/m
    }


def bridge_structural_params() -> dict:
    rho = config['bridge_rho']
    D   = config['bridge_D_ref']
    fn  = config['bridge_fn_hz']
    m   = config['bridge_mass']

    zeta   = config['bridge_zeta']
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta

    params_bridge = {
            "c": c,              # N*s/m (legacy key)
            "k": k,                    # N/m (legacy key)
            "m": m,                    # N/m (per unit span)
        }
    
    return params_bridge