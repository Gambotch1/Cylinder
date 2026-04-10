import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob

D = 1  # Diameter in meters

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

cdDir = DIR / 'data' / 'cd'
clDir = DIR / 'data' / 'cl'
dispDir = DIR / 'data' / 'disp'

print(dispDir)

# Dictionary of your specific files (Velocity -> Filename)
files = dict()
for f in os.listdir(dispDir):
    if f.endswith(".out"):
        files[f.split("Ur")[-1].split(".out")[0]] = os.path.relpath(os.path.join(dispDir, f), DIR)
files = dict(sorted(files.items()))



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

# --- 2. GENERATE PLOTS ---
# Store average amplitudes (cm) for each velocity
avg_results = []
for u_str, fpath in files.items():  
    df = parse_fluent_out(fpath)
    
    if df is not None and not df.empty:
        U_red = float(u_str)
        U = U_red * D  * 0.2 # Convert Ur to actual velocity (U = Ur * f_n * D) assuming f_n=1 Hz for simplicity

        print(U)
        t = df['time'].to_numpy()

        # Average amplitude in centimeters (mean of absolute displacement)
        avg_amplitude_cm = df['val'].abs().mean() * 100
        avg_results.append({'U_m_s': U, 'avg_h_cm': avg_amplitude_cm})
        print(f"Average amplitude for U={u_str} m/s: {avg_amplitude_cm:.3f} cm")
        
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
        
        # Save
        disp_plots = os.makedirs("plots_disp", exist_ok=True)
        save_name = f'plots_disp/disp_history_{u_str.replace(".", "p")}.png'
        plt.savefig(save_name, dpi=300)
        print(f"Plot saved: {save_name}")
        plt.show() 
        plt.close()
    else:
        print(f"No data found for {u_str} m/s")

# --- 3. SAVE AVERAGE AMPLITUDES ---
if avg_results:
    os.makedirs('plots_disp', exist_ok=True)
    avg_df = pd.DataFrame(avg_results)
    avg_df = avg_df.sort_values(by='U_m_s')
    csv_path = 'plots_disp/avg_amplitudes.csv'
    avg_df.to_csv(csv_path, index=False)
    print(f"Saved average amplitudes: {csv_path}")
