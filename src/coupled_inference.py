from unittest import result

import numpy as np
import torch
from models.gru import VIV_GRU
import pickle
import matplotlib.pyplot as plt

def run_coupled_viv(
    model:        VIV_GRU,
    x_scaler,
    case_stats:   dict,
    case_name:    str,
    seq_len:      int,
    input_cols:   list[str],
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
    release_step: int,     # hold stationary until this step
    device:       str = "cpu",
    use_ur_context: bool = False,
    ur_stats:     tuple   = (0.0, 1.0),
) -> dict:
    """
    Fully coupled GRU-structural VIV simulation.
    
    No CFD required. The GRU predicts CL from kinematics,
    which drives the structural equation of motion,
    which produces new kinematics for the next GRU prediction.
    
    This is the 2-way FSI loop described in the thesis.
    """
    model.eval()
    _, sigma = case_stats[case_name]

    mu = 0.0
    # State vectors
    h     = np.zeros(n_steps + 1, dtype=np.float32)  # displacement
    h_dot = np.zeros(n_steps + 1, dtype=np.float32)  # velocity
    CL    = np.zeros(n_steps + 1,     dtype=np.float32)  # lift coefficient
    t     = np.arange(n_steps + 1) * dt

    # Build initial kinematic history (all zeros — stationary phase)
    # Shape: (seq_len, n_features)
    n_feat  = len(input_cols) + (1 if use_ur_context else 0)
    history = np.zeros((seq_len, n_feat), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_steps):

            if i >= release_step:
                # ── Step 1: predict CL from current kinematic history ──────
                history_scaled = history.copy()
                # Scale kinematics (input_cols only, not Ur column)
                history_scaled[:, :len(input_cols)] = x_scaler.transform(
                    history[:, :len(input_cols)])

                x = torch.from_numpy(history_scaled).unsqueeze(0).to(device)
                cl_scaled, _ = model(x)
                cl = float(cl_scaled.item()) * sigma + mu
                CL[i] = cl

                # ── Step 2: compute aerodynamic force ─────────────────────
                F_aero = 0.5 * rho * U**2 * D * cl

                # ── Step 3: solve structural equation (Newmark-β, β=0.25) ─
                # Simple explicit Euler for clarity; use Newmark for accuracy
                h_ddot = (F_aero - c * h_dot[i] - k * h[i]) / m
                h_dot[i+1] = h_dot[i] + dt * h_ddot
                h[i+1]     = h[i]     + dt * h_dot[i+1]

            # ── Step 4: update kinematic history window ────────────────────
            acc_i = (h_dot[i] - h_dot[max(0, i-1)]) / dt if i > 0 else 0.0
            new_row = np.array([h[i], h_dot[i], acc_i], dtype=np.float32)

            if use_ur_context:
                ur_val    = float(case_name[2:])
                ur_mean, ur_std = ur_stats
                ur_scaled = (ur_val - ur_mean) / (ur_std + 1e-8)
                new_row   = np.append(new_row, ur_scaled)

            history = np.roll(history, -1, axis=0)
            history[-1] = new_row

    return {
        "time":         t[release_step:],
        "displacement": h[release_step:],
        "velocity":     h_dot[release_step:],
        "CL":           CL[release_step:],
    }




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model and artifacts ───────────────────────────────────────────
    model = VIV_GRU(
        input_size=4, hidden_size=128, num_layers=2, dropout=0.1
    ).to(device)
    model.load_state_dict(
        torch.load("results/gru_cylinder1000/gru_best.pt", map_location=device)
    )

    with open("results/gru_cylinder1000/x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open("results/gru_cylinder1000/case_stats.pkl", "rb") as f:
        case_stats = pickle.load(f)
    
    print(f"Ur4 stats: mu={case_stats['Ur4'][0]:.6f}  sigma={case_stats['Ur4'][1]:.6f}")

    # ── Ur statistics from training cases ─────────────────────────────────
    ur_values = [float(c[2:]) for c in case_stats if c.startswith("Ur")]
    ur_mean   = float(np.mean(ur_values))
    ur_std    = float(np.std(ur_values)) + 1e-8
    print(f"Ur stats: mean={ur_mean:.3f}  std={ur_std:.3f}")

    # ── Physical parameters — must match UDF exactly ───────────────────────
    rho    = 1.0       # model fluid density [kg/m³]
    U      = 1.0       # freestream velocity [m/s]  ← matches training
    D      = 0.2       # cylinder diameter [m]
    Ur     = 4.0       # reduced velocity to simulate

    fn     = U / (Ur * D)              # natural frequency [Hz]
    omega_n = 2.0 * np.pi * fn
    M_star = 2.0                        # mass ratio — matches UDF
    zeta   = 0.007                      # damping ratio — matches UDF
    m      = M_star * rho * (np.pi * D**2 / 4.0)
    c      = 2.0 * m * omega_n * zeta
    k      = m * omega_n**2

    print(f"Ur={Ur}  fn={fn:.4f} Hz  m={m:.4f}  k={k:.4f}  c={c:.6f}")
    print(f"Expected steady-state period: {1/fn:.3f} s")

    # ── Simulation setup ───────────────────────────────────────────────────
    dt           = 0.005          # match CFD timestep [s]
    t_release    = 16.0           # physical release time [s] — matches UDF
    total_time   = 200.0          # simulate 200s total
    n_steps      = int(total_time / dt)
    release_step = int(t_release / dt)

    print(f"Steps: {n_steps}  release at step {release_step} (t={t_release}s)")

    result = run_coupled_viv(
        model        = model,
        x_scaler     = x_scaler,
        case_stats   = case_stats,
        case_name    = f"Ur{int(Ur) if Ur == int(Ur) else Ur}",
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
        release_step = release_step,
        device       = device,
        use_ur_context = True,
        ur_stats     = (ur_mean, ur_std),
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    t    = result["time"]
    h    = result["displacement"]
    hdot = result["velocity"]
    CL   = result["CL"]
    FL   = 0.5 * rho * U**2 * D * CL

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    axes[0].plot(t, h / D, lw=1.2, color="tab:blue")
    axes[0].axhline(0, color="gray", lw=0.6, ls="--")
    axes[0].set_ylabel(r"$h/D$  (dimensionless displacement)")
    axes[0].set_title(f"Coupled GRU-Structural VIV Simulation  —  $U_r = {Ur}$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, CL, lw=1.0, color="tab:orange")
    axes[1].axhline(0, color="gray", lw=0.6, ls="--")
    axes[1].set_ylabel("$C_L$  (lift coefficient)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, FL, lw=1.0, color="tab:red")
    axes[2].axhline(0, color="gray", lw=0.6, ls="--")
    axes[2].set_ylabel("Lift force [N/m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.savefig("results/coupled_viv_response.png", dpi=150)
    plt.close(fig)

    # ── Steady-state amplitude ─────────────────────────────────────────────
    ss_start = int(0.7 * len(h))
    amp      = (h[ss_start:].max() - h[ss_start:].min()) / (2 * D)
    print(f"\nSteady-state A/D = {amp:.4f}")
    print(f"Displacement range: {h.min():.4f} to {h.max():.4f} m")
    print(f"CL range: {CL.min():.4f} to {CL.max():.4f}")


if __name__ == "__main__":
    main()