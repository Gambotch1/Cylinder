import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from viv_analysis.models.gru import VIV_GRU
import pickle
import matplotlib.pyplot as plt

from viv_analysis.preprocess import compute_kinematics, merge_dataframes, correct_cl_for_reference_velocity
from viv_analysis.utils import PROJECT_ROOT
from viv_analysis.utils import format_ur_label


def Newmark_beta( F, h, h_dot, h_ddot, dt, m, c, k, beta=0.25, gamma=0.5):
    """
    Newmark-beta
    """
    # coefficients
    a1 = 1/(beta * dt**2) * m + gamma / (beta * dt) * c
    a2 = 1/(beta * dt) * m + (gamma / beta - 1) * c
    a3 = (1/(2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c
    kbar = k + a1
    
    # Next time step
    pbar = F + a1 * h + a2 * h_dot + a3 * h_ddot
    h1 = pbar / kbar
    h_dot_1 = gamma / (beta * dt) * (h1 - h) + (1 - gamma / beta) * h_dot + dt * (1 - gamma / (2 * beta)) * h_ddot
    h_ddot_1 = 1 / (beta * dt**2) * (h1 - h) - 1 / (beta * dt) * h_dot - (1 / (2 * beta) - 1) * h_ddot

    return h1, h_dot_1, h_ddot_1

def warmup_history(cfd_case_df, release_t, seq_len, input_cols,
                   x_scaler, use_ur_context, ur_value, ur_stats):
    
    ordered = cfd_case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float32)
    release_idx = int(np.searchsorted(times, release_t))

    handoff_idx = release_idx + seq_len

    if handoff_idx >= len(ordered):
        raise ValueError(
            f"CFD trajectory too short: need {seq_len} steps AFTER release "
            f"at index {release_idx}, but trajectory has only "
            f"{len(ordered) - release_idx} post-release steps."
        )

    
    win = ordered.iloc[release_idx : handoff_idx]
    kinematics = win[input_cols].to_numpy(dtype=np.float32)
    kinematics_scaled = x_scaler.transform(kinematics)

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe 
        ur_column = np.full((seq_len, 1), ur_scaled, dtype=np.float32)
        history = np.hstack([kinematics_scaled, ur_column])
    else:
        history = kinematics_scaled

    initial_state = {
        "h": float(ordered["disp"].iloc[handoff_idx]),
        "h_dot": float(ordered["vel"].iloc[handoff_idx]),
        "h_ddot": float(ordered["acc"].iloc[handoff_idx]),
    }

    handoff_time = float(times[handoff_idx])
    return history.astype(np.float32), initial_state, handoff_time, handoff_idx


def run_coupled_viv(
    model:        VIV_GRU,
    x_scaler,
    y_scaler,
    seq_len:      int,
    input_cols:   list[str],
    initial_history: np.ndarray,
    initial_state: dict,
    # Structural parameters
    m:            float,   # mass per unit span [kg/m]
    c:            float,   # damping coefficient [N·s/m]
    k:            float,   # stiffness [N/m]
    # Flow parameters
    rho:          float,   # fluid density [kg/m³]
    U:            float,   # freestream velocity [m/s]
    D:            float,   # reference depth [m]
    dt:           float,   # timestep [s]
    n_steps:      int,     # total timesteps to simulate
    use_ur_context: bool = False,
    ur_value:     float = 0.0,
    ur_stats:     tuple   = (0.0, 1.0),
    device:       str = "cpu",
) -> dict:
    """
    Fully coupled GRU-structural VIV simulation.
    
    No CFD required. The GRU predicts CL from kinematics,
    which drives the structural equation of motion,
    which produces new kinematics for the next GRU prediction.
    
    This is the 2-way FSI loop described in the thesis.
    """
    model.eval()

    # State vectors
    h      = np.zeros(n_steps + 1, dtype=np.float32)  # displacement
    h_dot  = np.zeros(n_steps + 1, dtype=np.float32)  # velocity
    h_ddot = np.zeros(n_steps + 1, dtype=np.float32) # acceleration
    CL     = np.zeros(n_steps + 1, dtype=np.float32)  # lift coefficient


    h[0]      = initial_state["h"]
    h_dot[0]  = initial_state["h_dot"]
    h_ddot[0] = initial_state["h_ddot"]
    history   = initial_history.copy()  

    # Build initial kinematic history (all zeros — stationary phase)
    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe

    with torch.no_grad():
        for i in range(n_steps):  

            x = torch.from_numpy(history).unsqueeze(0).to(device)
            cl_scaled, _ = model(x)
            cl = float(y_scaler.inverse_transform(
                np.array([[cl_scaled.item()]], dtype=np.float32)
            )[0,0])
            CL[i] = cl

            # ── Step 2: compute aerodynamic force ─────────────────────
            F_aero = 0.5 * rho * U**2 * D * cl

            h[i+1], h_dot[i+1], h_ddot[i+1] = Newmark_beta(
                F=F_aero,
                h=h[i],
                h_dot=h_dot[i],
                h_ddot=h_ddot[i],
                dt=dt,
                m=m,
                c=c,
                k=k,
            )
            # ── Step 4: update kinematic history window ────────────────────
            # acc_i = (h_dot[i] - h_dot[max(0, i-1)]) / dt if i > 0 else 0.0

            new_kinematics_raw = np.array([h[i+1], h_dot[i+1], h_ddot[i+1]], dtype=np.float32)
            new_kinematics_scaled = x_scaler.transform(new_kinematics_raw.reshape(1, -1))[0]

            if use_ur_context:
                new_row = np.append(new_kinematics_scaled, ur_scaled).astype(np.float32)
            else:
                new_row = new_kinematics_scaled.astype(np.float32)

            history = np.roll(history, -1, axis=0)
            history[-1] = new_row


    t = np.arange(n_steps + 1) * dt

    return {
        "time": t[:n_steps],
        "displacement": h[:n_steps],
        "velocity": h_dot[:n_steps],
        "acceleration": h_ddot[:n_steps],
        "CL": CL[:n_steps],
    }




def main(
    Ur: float = 6.0, # reduced velocity to simulate
    dataset: str = "Cylinder1000", # dataset name for loading scalers and stats
    total_time: float = 700.0, # total simulation time in seconds
    artifact_dir: Optional[Path] = None,
    ):

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if artifact_dir is None:
        artifact_dir = PROJECT_ROOT / "results" / f"gru_{dataset}"

    # ── Load model + scalers + ur_stats ───────────────────────────────────
    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(artifact_dir / "ur_stats.pkl", "rb") as f:
        ur_info = pickle.load(f)



    # ── Ur statistics from training cases ─────────────────────────────────
    use_ur_context = bool(ur_info["use_ur_context"])
    ur_mean   = float(ur_info["mean"])
    ur_std    = float(ur_info["std"]) + 1e-8
    print(f"Loaded scalers and UR stats: use_ur_context={use_ur_context}"  
          f"mean={ur_mean:.4f}, std={ur_std:.4f}")
    print(f"y_scaler: mean={float(y_scaler.mean_[0]):.6f}  "
          f"scale={float(y_scaler.scale_[0]):.6f}")


    # ── Physical parameters — must match UDF exactly ───────────────────────
    rho = 1.0
    D   = 0.2
    fn  = 0.2
    U   = Ur * fn * D
    M_star = 2.0
    zeta   = 0.007
    m = M_star * rho * (np.pi * D**2 / 4.0)
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta
    print(f"Ur={Ur}  U={U:.4f} m/s  fn={fn:.4f} Hz")
    print(f"m={m:.6f} kg/m  k={k:.6f}  c={c:.6f}")

    print(f"Expected steady-state period: {1/fn:.3f} s")

    params_Re1000 = {
            "m": m,                 # kg/m (legacy key)
            "c": c,              # N*s/m (legacy key)
            "k": k,                    # N/m (legacy key)
            "cylinder_mass": m,    # kg/m
            "c_struct": c,       # N*s/m
            "k_struct": k,             # N/m
        }

    # ── Simulation setup ───────────────────────────────────────────────────
    t_star_release = 80.0
    t_release = t_star_release * D / U
    dt = 0.005
    print(f"t_release={t_release:.4f}s")


    # ── Load CFD trajectory at this Ur ────────────────────────────────────
    print(f"\nLoading CFD trajectory at Ur={Ur} for warm-start...")
    raw_df = merge_dataframes(dataset=dataset)
    if raw_df.empty:
        raise RuntimeError("Could not load CFD data for warmup.")
    # Correct CL normalization if Fluent used a different reference velocity
    raw_df = correct_cl_for_reference_velocity(raw_df, fn=fn, d_ref=D)
    raw_df = compute_kinematics(raw_df,dataset= dataset, structural_params=params_Re1000)

    case_label = format_ur_label(Ur)
    case_df = raw_df[raw_df["case"] == case_label].copy()
    if case_df.empty:
        available_cases = sorted(raw_df["case"].unique().tolist())
        raise ValueError(f"No CFD data found for case '{case_label}'."
                        f"Available cases: {available_cases}")
    print(f"  CFD trajectory has {len(case_df)} steps")

    seq_len = 960  
    input_cols = ["disp", "vel", "acc"]
    initial_history, initial_state, t_handoff, handoff_idx = warmup_history(
        cfd_case_df=case_df,
        release_t=t_release,
        seq_len=seq_len,
        input_cols=input_cols,
        x_scaler=x_scaler,
        use_ur_context=use_ur_context,
        ur_value=Ur,
        ur_stats=(ur_mean, ur_std),
    )
    print(f"  Warm-start window: CFD steps "
          f"{handoff_idx}]  (post-release)")
    print(f"  Handoff at t={t_handoff:.4f}s   "
          f"({seq_len*dt:.2f}s after release)")
    print(f"  Initial state at handoff: "
          f"h={initial_state['h']:.6f}  h_dot={initial_state['h_dot']:.6f}  "
          f"h_ddot={initial_state['h_ddot']:.6f}")

        # Sanity check: warm-start kinematics should NOT all be zero
    raw_kin = case_df.iloc[handoff_idx - seq_len : handoff_idx][input_cols].to_numpy()
    print(f"  Warm-start raw stats:")
    print(f"    disp range: [{raw_kin[:,0].min():.5f}, {raw_kin[:,0].max():.5f}]")
    print(f"    vel  range: [{raw_kin[:,1].min():.5f}, {raw_kin[:,1].max():.5f}]")
    print(f"    acc  range: [{raw_kin[:,2].min():.5f}, {raw_kin[:,2].max():.5f}]")


    # ── Build model (read hidden_size from saved metrics) ─────────────────
    input_size = len(input_cols) + (1 if use_ur_context else 0)
    hidden_size, num_layers = 128, 2
    metrics_path = artifact_dir / "metrics_gru.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)
        hidden_size = saved_metrics["gru_config"].get("hidden_size", hidden_size)
        num_layers  = saved_metrics["gru_config"].get("num_layers", num_layers)


    model = VIV_GRU(input_size=input_size, hidden_size=hidden_size, 
                    num_layers=num_layers, dropout=0.1).to(device)
    model.load_state_dict(torch.load(artifact_dir / "gru_best.pt", map_location=device))

    print(f"Model loaded: input_size={input_size}  hidden_size={hidden_size}")


    # ── Run coupled inference ─────────────────────────────────────────────
    n_steps = int((total_time - t_handoff) / dt)
    print(f"\nRunning coupled inference for {total_time}s ({n_steps} steps)...")
    print(f"{n_steps} steps) ...")
    result = run_coupled_viv(
        model        = model,
        x_scaler     = x_scaler,
        y_scaler     = y_scaler,
        initial_history = initial_history,
        initial_state = initial_state,
        seq_len      = 960,
        input_cols   = ["disp", "vel", "acc"],
        m            = m,
        c            = c,
        k            = k,
        rho          = rho,
        U            = U,
        D            = D,
        dt           = dt,
        n_steps      = n_steps,
        device       = device,
        use_ur_context = use_ur_context,
        ur_value=Ur,
        ur_stats     = (ur_mean, ur_std),
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    t    = result["time"] + t_handoff
    h    = result["displacement"]
    CL   = result["CL"]
    FL   = 0.5 * rho * U**2 * D * CL

    # ── Plot: GRU result alongside CFD ground truth for comparison ────────
    CFD_t = case_df["time"].values
    CFD_h = case_df["disp"].values
    CFD_cl = case_df["cl"].values


    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)

    axes[0].plot(CFD_t, CFD_h / D, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[0].plot(t, h / D, lw=1, color="tab:blue", alpha=0.9, label="GRU coupled")
    axes[0].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7,
                    label=f"handoff t={t_handoff:.1f}s")
    axes[0].axhline(0, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel(r"$h/D$")
    axes[0].set_title(f"Coupled GRU-Structural VIV  —  $U_r={Ur}$  (post-release warm-start)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(CFD_t, CFD_cl, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[1].plot(t, CL, lw=1, color="tab:orange", alpha=0.9, label="GRU coupled")
    axes[1].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7)
    axes[1].axhline(0, color="gray", lw=0.5, ls=":")
    axes[1].set_ylabel("$C_L$")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
 
    axes[2].plot(t, FL, lw=1, color="tab:red")
    axes[2].axhline(0, color="gray", lw=0.5, ls=":")
    axes[2].set_ylabel("Lift force [N/m]")
    axes[2].grid(True, alpha=0.3)
 
    axes[3].plot(t, result["acceleration"], lw=1, color="tab:green")
    axes[3].set_ylabel("Acc [m/s²]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)


    out_png = PROJECT_ROOT / "results" / f"coupled_viv_Ur_{Ur}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved coupled VIV plot to {out_png}")

    # ── Steady-state amplitude ─────────────────────────────────────────────
    ss_start = int(0.7 * len(h))
    amp      = (h[ss_start:].max() - h[ss_start:].min()) / (2 * D)
    print(f"\nSteady-state A/D = {amp:.4f}")
    print(f"Displacement range: {h.min():.4f} to {h.max():.4f} m")
    print(f"CL range: {CL.min():.4f} to {CL.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ur",type=float, default=6.0)
    parser.add_argument("--dataset", type=str, default="cylinder1000")
    parser.add_argument("--total_time", type=float, default=700.0)
    args = parser.parse_args()
    main(
        Ur=args.Ur,
        dataset=args.dataset,
        total_time=args.total_time,
    )
    
