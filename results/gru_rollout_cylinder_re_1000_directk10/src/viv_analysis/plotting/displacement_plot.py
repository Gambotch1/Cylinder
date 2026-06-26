import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

from viv_analysis.plotting.plot import main
from viv_analysis.config import config

# D = 1  # Diameter in meters

# --- 1. CONFIGURATION FOR LATEX-STYLE PLOTS ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,            
    'mathtext.fontset': 'stix',     
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})
DIR = Path(__file__).resolve().parents[3]

Dataset = ["Cylinder200", "Cylinder1000"]

# cdDir = DIR / 'data' / 'cd'
# clDir = DIR / 'data' / 'cl'
# dispDir = DIR / 'data' / 'disp'

# Cylinder_Re1000_CD_DIR = DIR / 'data' / 'Cylinder' / 'cd'
# Cylinder_Re1000_CL_DIR = DIR / 'data' / 'Cylinder' / 'cl'
# Cylinder_Re1000_DISP_DIR = DIR / 'data' / 'Cylinder' / 'disp'

# print(dispDir)


ROOT = DIR


def parse_fluent_out(filepath):
    """
    Parses Ansys Fluent report files.
    Skips header lines and assumes structure: [TimeStep, Value, FlowTime]
    """
    try:
        
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        start_idx = 0
        for i, line in enumerate(lines):
            # Find the start of numeric data (starts with a digit)
            if len(line.strip()) > 0 and line.strip()[0].isdigit():
                start_idx = i
                break
                
        # Read the data portion
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # columns: step, value, flow-time
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
                    
        df = pd.DataFrame(data, columns=["step", "val", "time"])
        return df.sort_values(by="time")
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
    
def get_disp_dir(dataset: str) -> Path:
    if dataset == "Cylinder200":
        return ROOT / "data" / "disp"
    if dataset == "Cylinder1000":
        return ROOT / "data" / "cylinder_Re_1000" / "disp"
    raise ValueError(f"Unknown dataset: {dataset}")

def collect_files(dataset: str):
    disp_dir = get_disp_dir(dataset)
    files = {}
    for f in os.listdir(disp_dir):
        if f.endswith(".out"):
            ur = f.split("Ur")[-1].split(".out")[0]
            if ur.startswith("_"):
                ur = ur[1:]
            files[ur] = disp_dir / f
    return dict(sorted(files.items(), key=lambda kv: float(kv[0])))

# --- 2. GENERATE PLOTS ---
# Store average amplitudes (cm) for each velocity
def run(dataset: str):
    files = collect_files(dataset)
    if not files:
        print(f"No .out files found for {dataset}")
        return

    avg_results = []
    save_dir = ROOT / "plots_disp" / dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    D  = 1.0 if dataset == "Cylinder200" else config['cylinder1000_D_ref']
    fn = config['cylinder1000_fn']  # fixed natural frequency [Hz]

    for u_str, fpath in files.items():
        df = parse_fluent_out(fpath)
        if df is None or df.empty:
            print(f"No data found for Ur={u_str}")
            continue

        U_red = float(u_str)
        U     = U_red * fn * D   # freestream velocity [m/s]

        t = df['time'].to_numpy()
        h = df['val'].to_numpy()

        # ── Steady-state amplitude: half peak-to-peak of last 40% ──────────
        ss_start      = int(0.6 * len(df))
        h_ss          = h[ss_start:]
        amplitude_m   = (h_ss.max() - h_ss.min()) / 2.0
        amplitude_cm  = amplitude_m * 100
        A_star        = amplitude_m / D   # dimensionless A/D

        avg_results.append({
            'Ur':          U_red,
            'U_m_s':       U,
            'amplitude_cm': amplitude_cm,
            'A_star':      A_star,
        })
        print(f"Ur={U_red:.2f}  U={U:.4f} m/s  "
              f"A={amplitude_cm:.3f} cm  A/D={A_star:.4f}")

        # ── Dimensionless time series plot ─────────────────────────────────
        t_star = (U * t) / D
        h_star = h / D

        plt.figure(figsize=(8, 5))
        plt.plot(t_star, h_star, color='black', linewidth=0.8)
        plt.axhline(0, color='gray', lw=0.5, alpha=0.5)

        # Mark steady-state region
        t_star_ss = t_star[ss_start:]
        plt.axvspan(t_star_ss[0], t_star_ss[-1],
                    alpha=0.08, color='tab:blue',
                    label=f'Steady-state  $A^*={A_star:.3f}$')

        plt.xlabel(r'Dimensionless time $t^* = Ut/D$', fontsize=13)
        plt.ylabel(r'Dimensionless displacement $h/D$', fontsize=13)
        plt.title(f'VIV response — $U_r = {U_red:.2f}$  '
                  f'($U = {U:.4f}$ m/s,  Re=1000)')
        plt.xlim(t_star.min(), t_star.max())
        plt.legend(fontsize=11)
        plt.tight_layout()

        save_name = save_dir / f'disp_history_{u_str.replace(".", "p")}.png'
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"  Plot saved: {save_name}")

    # ── Save amplitude summary ─────────────────────────────────────────────
    if avg_results:
        avg_df = pd.DataFrame(avg_results).sort_values('Ur')
        out_csv = save_dir / "avg_amplitudes.csv"
        avg_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(avg_df[['Ur','U_m_s','amplitude_cm','A_star']].to_string(index=False))


def compute_dominant_frequency(t, h):
    """Return dominant oscillation frequency [Hz] via FFT (DC excluded)."""
    dt = np.median(np.diff(t))
    h_centered = h - h.mean()
    freqs = np.fft.rfftfreq(len(h_centered), d=dt)
    power = np.abs(np.fft.rfft(h_centered)) ** 2
    idx = np.argmax(power[1:]) + 1   # skip DC bin
    return freqs[idx]


def plot_frequency_curve(dataset: str, output_path=None):
    """Compute dominant displacement frequency per Ur and plot f*/fn vs Ur."""
    files = collect_files(dataset)
    if not files:
        print(f"No .out files found for {dataset}")
        return

    D  = 1.0 if dataset == "Cylinder200" else config['cylinder1000_D_ref']
    fn = config['cylinder1000_fn']   # 0.2 Hz
    St = 0.2                          # Strouhal number

    results = []
    for u_str, fpath in files.items():
        df = parse_fluent_out(fpath)
        if df is None or df.empty:
            continue

        U_red = float(u_str)
        t = df['time'].to_numpy()
        h = df['val'].to_numpy()

        ss_start = int(0.6 * len(df))
        t_ss = t[ss_start:]
        h_ss = h[ss_start:]

        # Skip cases with negligible oscillation
        amp = (h_ss.max() - h_ss.min()) / 2.0
        if amp / D < 1e-4:
            print(f"Ur={U_red:.2f}  skipped (A/D < 1e-4)")
            continue

        f_dom = compute_dominant_frequency(t_ss, h_ss)
        f_norm = f_dom / fn

        results.append({'Ur': U_red, 'f_star_Hz': f_dom, 'f_over_fn': f_norm})
        print(f"Ur={U_red:.2f}  f*={f_dom:.4f} Hz  f*/fn={f_norm:.4f}")

    if not results:
        print("No frequency data computed.")
        return

    res_df = pd.DataFrame(results).sort_values('Ur')

    ur_line = np.linspace(res_df['Ur'].min(), res_df['Ur'].max(), 300)
    strouhal_line = St * ur_line   # f_St/fn = St * Ur

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(res_df['Ur'], res_df['f_over_fn'],
            'o-', color='tab:blue', lw=1.5, ms=6, label=r'CFD: $f^*/f_n$')
    ax.plot(ur_line, strouhal_line,
            '--', color='tab:orange', lw=1.5, label=fr'Strouhal ($St={St}$)')
    ax.axhline(1.0, color='tab:red', lw=1.2, ls=':', label=r'Lock-in ($f^*/f_n=1$)')

    ax.set_xlabel(r'Reduced velocity $U_r = U/(f_n D)$', fontsize=13)
    ax.set_ylabel(r'Normalised frequency $f^*/f_n$', fontsize=13)
    ax.set_title(f'VIV frequency response — {dataset}', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    save_dir = ROOT / "plots_disp" / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = save_dir / "frequency_curve.png"

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)

    csv_path = save_dir / "frequency_curve.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_all_amplitudes(dataset_filter=None, amplitudes_dir=None, output_path=None):
    if amplitudes_dir is None:
        amplitudes_dir = ROOT / "plots_disp"

    csv_files = list(amplitudes_dir.rglob("avg_amplitudes.csv"))
    all_data  = []

    for csv_file in csv_files:
        case_label = csv_file.parent.name
        if dataset_filter and case_label != dataset_filter:
            continue
        df = pd.read_csv(csv_file)
        df["case"] = case_label
        all_data.append(df)

    if not all_data:
        print("No amplitude data found.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for case in combined["case"].unique():
        sub = combined[combined["case"] == case].sort_values("Ur")
        # Use A* = A/D if available, otherwise fall back to cm
        y_col = "A_star" if "A_star" in sub.columns else "amplitude_cm"
        y_label = r"Normalised amplitude $A^* = A/D$" \
                  if y_col == "A_star" else "Average amplitude [cm]"
        ax.plot(sub["Ur"], sub[y_col],
                marker='o', label=case, lw=1.5, ms=6)

    ax.set_xlabel(r"Reduced velocity $U_r = U / (f_n D)$", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title("VIV amplitude response curve", fontsize=14)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)

    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot displacement histories from Fluent .out files")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Cylinder200",
        choices=["Cylinder200", "Cylinder1000"],
        help="Dataset to process",
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    run(args.dataset)

    amplitudes_output = ROOT / "plots_disp" / f"{args.dataset}_amplitudes.png"
    plot_all_amplitudes(dataset_filter=args.dataset, output_path=amplitudes_output)

    plot_frequency_curve(args.dataset)