import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

D = 1.0  # UPDATED: Diameter in meters

# --- 1. CONFIGURATION FOR LATEX-STYLE PLOTS ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,            
    'mathtext.fontset': 'stix',     
    'axes.labelsize': 14,
    'font.size': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 0.8, # Very thin line to capture that raw textbook look
})

def parse_fluent_out(filepath):
    try:
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        start_idx = 0
        for i, line in enumerate(lines):
            if len(line.strip()) > 0 and line.strip()[0].isdigit():
                start_idx = i
                break
                
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
                    
        df = pd.DataFrame(data, columns=["step", "val", "time"])
        return df.sort_values(by="time")
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def calculate_psd(time, signal):
    dt = np.mean(np.diff(time))
    fs_samp = 1.0 / dt
    
    # Using a large nperseg gives that "spiky/jagged" unsmoothed look from the book
    nperseg = min(len(signal), 4096) 
    frequencies, psd = welch(signal, fs=fs_samp, nperseg=nperseg)
    return frequencies, psd

def plot_exact_textbook_spectra(files_dict, Ur_dict, fn_hz):
    """
    Creates the 3-panel plot matching the Simiu & Yeo visual style.
    Ur_dict maps the state to its Reduced Velocity (Ur) to calculate physics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False) # sharey=False lets each peak scale naturally
    
    states = ['Before Lock-in', 'At Lock-in', 'After Lock-in']
    
    for ax, state in zip(axes, states):
        filepath = files_dict.get(state)
        Ur = Ur_dict.get(state)
        
        if filepath and os.path.exists(filepath):
            df = parse_fluent_out(filepath)
            
            if df is not None and not df.empty:
                # Calculate PSD 
                freqs, psd = calculate_psd(df['time'].values, df['val'].values)
                
                # Plot Spectral Density S(f)
                ax.plot(freqs, psd, color='black')
                
                # --- VISUAL STYLING TO MATCH TEXTBOOK ---
                ax.set_xlim(0, fn_hz * 3)  # Frame the plot around fn (up to 3x fn)
                ax.set_ylim(bottom=0)      # Force y-axis to start at 0
                
                # Inward ticks on all sides, no grid
                ax.tick_params(direction='in', right=True, top=True, length=5)
                ax.grid(False)
                
                # Minimalist x-axis label on the bottom right of each panel
                ax.set_xlabel('f', loc='right', fontstyle='italic')
                
                # Only add y-axis label to the leftmost plot
                if ax == axes[0]:
                    ax.set_ylabel('S(f)', rotation=0, labelpad=20, fontstyle='italic')
                
                # --- ADDING FLOATING LABELS FOR fn AND fs ---
                # Calculate expected Strouhal frequency
                # U = Ur * fn * D
                # fs = St * U / D  (Assuming Strouhal number ~ 0.2 for a cylinder)
                U_physical = Ur * fn_hz * D
                fs_hz = 0.2 * U_physical / D
                
                # Helper function to find the max PSD height near a target frequency
                def get_local_max_psd(target_f, search_window=0.05):
                    mask = (freqs > target_f - search_window) & (freqs < target_f + search_window)
                    if any(mask):
                        return max(psd[mask])
                    return 0
                
                psd_at_fn = get_local_max_psd(fn_hz)
                psd_at_fs = get_local_max_psd(fs_hz)
                
                # Annotate fn and fs
                # Adjust the '+ 0.05' and '+ max(psd)*0.05' to nudge the text around visually
                ax.text(fn_hz + (fn_hz*0.05), psd_at_fn + (max(psd)*0.05), r'$f_n$', fontsize=14)
                
                # Only plot fs if it's distinct from fn (otherwise they overlap during lock-in)
                if abs(fs_hz - fn_hz) > (fn_hz * 0.1): 
                    ax.text(fs_hz - (fn_hz*0.15), psd_at_fs + (max(psd)*0.05), r'$f_s$', fontsize=14)

        else:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    os.makedirs('plots_disp', exist_ok=True)
    plt.savefig('plots_disp/exact_textbook_spectra.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Print the current working directory to help debug
    print(f"Current Working Directory: {os.getcwd()}")
    
    # 1. Map your states to the files (CHECK THESE PATHS)
    files = {
        'Before Lock-in': "disp/disp-Ur3.0.out",
        'At Lock-in':     "disp/disp-Ur5.4.out", # Set to your actual lock-in file
        'After Lock-in':  "disp/disp-Ur9.0.out",

    }
    
    # Diagnostic print loop
    for state, path in files.items():
        if os.path.exists(path):
            print(f"SUCCESS: Found file for {state} at '{path}'")
        else:
            print(f"FAILED: Could NOT find file for {state} at '{path}'")
            
    # 2. Map states to the Reduced Velocity (Ur)
    reduced_velocities = {
        'Before Lock-in': 3.0,
        'At Lock-in':     5.4,
        'After Lock-in':  9.0,
    }
    
    # 3. Define your structure's natural frequency (fn) in Hz
    fn_hz = 0.2 # <--- UPDATE THIS TO YOUR ACTUAL fn
    
    # Run the plot
    plot_exact_textbook_spectra(files, reduced_velocities, fn_hz)