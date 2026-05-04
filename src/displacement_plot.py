import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

from plot import main

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
DIR = Path(os.path.relpath(__file__).strip("src/displacement_plot.py"))

# Dataset = ["Cylinder200", "Cylinder1000"]

# cdDir = DIR / 'data' / 'cd'
# clDir = DIR / 'data' / 'cl'
# dispDir = DIR / 'data' / 'disp'

# Cylinder_Re1000_CD_DIR = DIR / 'data' / 'Cylinder' / 'cd'
# Cylinder_Re1000_CL_DIR = DIR / 'data' / 'Cylinder' / 'cl'
# Cylinder_Re1000_DISP_DIR = DIR / 'data' / 'Cylinder' / 'disp'

# print(dispDir)


ROOT = Path(__file__).resolve().parent.parent


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

    if dataset == "Cylinder200":
        D = 1.0  # Diameter in meters for Cylinder200
    elif dataset == "Cylinder1000":
        D = 0.2  # Diameter in meters for Cylinder1000

    for u_str, fpath in files.items():  
        df = parse_fluent_out(fpath)
        
        if df is not None and not df.empty:
            U_red = float(u_str)
            U = U_red * D  * 0.2 # Convert Ur to actual velocity (U = Ur * f_n * D) assuming f_n=1 Hz for simplicity

            print(U)
            t = df['time'].to_numpy()

            # Average amplitude in centimeters (mean of absolute displacement)
            avg_amplitude_cm = df['val'].abs().mean() * 100
            avg_results.append({'Ur': U_red, 'avg_h_cm': avg_amplitude_cm})
            print(f"Average amplitude for Ur={u_str}: {avg_amplitude_cm:.3f} cm")
            
            # Dimensionless quantities
            t_star = (U * t) / D  # Dimensionless time
            h_star = df['val'].to_numpy() / D  # Dimensionless displacement

            plt.figure(figsize=(8, 5))
            
            # Plot Data
            plt.plot(t_star, h_star, color='black', linewidth=1.0)
            
            # Labels and Title
            plt.xlabel(r'Dimensionless Time $t^* = Ut/D$')
            plt.ylabel(r'Dimensionless Displacement $h^* = h/D$')
            # Optional: Add Velocity to title or as text box
            plt.title(f'Vertical Response Time History ($U = {u_str}$ m/s)')
            
            # Axis Limits (Tight)
            plt.xlim(t_star.min(), t_star.max())
            
            # Optional: Add zero line for reference
            plt.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.ylim(-1, 1) 
            

            save_name = f'{save_dir}/disp_history_{u_str.replace(".", "p")}.png'
            plt.savefig(save_name, dpi=300)
            print(f"Plot saved: {save_name}")
            plt.show() 
            plt.close()
        else:
            print(f"No data found for {u_str} m/s")

    # --- 3. SAVE AVERAGE AMPLITUDES ---
    if avg_results:
        avg_df = pd.DataFrame(avg_results).sort_values(by="Ur")
        out_csv = save_dir / "avg_amplitudes.csv"
        avg_df.to_csv(out_csv, index=False)
        print(f"Saved average amplitudes: {out_csv}")

def plot_all_amplitudes(dataset_filter: str = None, amplitudes_dir: Path = None, output_path: Path = None):
    """
    Plots all average heights (avg_h_cm) against their respective velocities (U_m_s)
    in a single plot, combining data from all CSV files in the amplitudes directory.
    
    """
    if amplitudes_dir is None:
        amplitudes_dir = ROOT / "plots_disp"
    
    if not amplitudes_dir.exists():
        print(f"Error: Could not find {amplitudes_dir}")
        return
    
    # Collect all data from CSV files
    all_data = []
    csv_files = list(amplitudes_dir.rglob("avg_amplitudes.csv"))
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        case_label = csv_file.parent.name
        
        # Filter by dataset if specified
        if dataset_filter and case_label != dataset_filter:
            continue
            
        print(f"  - {csv_file}")
        df = pd.read_csv(csv_file)
        df["case"] = case_label
        all_data.append(df)
    
    if not all_data:
        print("No CSV files found with amplitude data.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Plot each case with a different marker/color
    for case in combined_df["case"].unique():
        case_data = combined_df[combined_df["case"] == case].sort_values("Ur")
        ax.plot(case_data["Ur"], case_data["avg_h_cm"], 
                marker='o', label=case, linewidth=1.5, markersize=6)
    
    # Formatting
    ax.set_xlabel("Reduced Velocity (Ur)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Height (cm)", fontsize=12, fontweight="bold")
    ax.set_title("Amplitude Response vs. Velocity", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=11)
    
    # Save or show
    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
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
    
    # Plot amplitudes for the processed dataset
    amplitudes_output = ROOT / "plots_disp" / f"{args.dataset}_amplitudes.png"
    plot_all_amplitudes(dataset_filter=args.dataset, output_path=amplitudes_output)