import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from viv_analysis.utils import PROJECT_ROOT


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



# Setup paths
ROOT_DIR = PROJECT_ROOT
RESULTS_DIR = ROOT_DIR / "results" / "elm_model"

def plot_case_with_uncertainty(csv_path: Path, case_name: str, output_path: Path = None):
    """
    Plots the true vs predicted time series for a specific case, 
    including the ensemble's uncertainty bands.
    """
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    # Load the predictions
    df = pd.read_csv(csv_path)
    
    # Filter and sort the specific case
    case_df = df[df["case"] == case_name].sort_values(by=["time", "step"])
    
    if case_df.empty:
        print(f"Case '{case_name}' not found in the dataset.")
        print("Available cases:", df["case"].unique())
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    
    # 1. Plot the True target values
    ax.plot(case_df["time"], case_df["y_true"], 
            label="True cl", color="black", linewidth=1.5)
    
    # 2. Plot the Predicted mean values
    ax.plot(case_df["time"], case_df["y_pred"], 
            label="ELM Ensemble Mean", color="tab:blue", linewidth=1.5, linestyle="--")
    
    # 3. Plot the Uncertainty Band (if available)
    if "y_std" in case_df.columns:
        # 95% confidence interval is approximately ±2 standard deviations
        lower_bound = case_df["y_pred"] - 2 * case_df["y_std"]
        upper_bound = case_df["y_pred"] + 2 * case_df["y_std"]
        
        ax.fill_between(case_df["time"], lower_bound, upper_bound, 
                        color="tab:blue", alpha=0.3, 
                        label="95% Confidence Interval (±2σ)")

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Lift Coefficient (cl)", fontsize=11)
    ax.set_title(f"Aeroelastic cl Prediction - Case: {case_name}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=600)
        print(f"Saved highly-detailed plot to: {output_path}")
    else:
        plt.show()


def main():
    # File to read
    csv_file = RESULTS_DIR / "test_predictions_cl_elm.csv"
    print(f"Looking for CSV file at: {csv_file}")
    print(f"File exists: {csv_file.exists()}")
    
    # Read the CSV just to see what cases we have available
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        available_cases = df["case"].unique()
        print(f"✓ CSV loaded successfully.")
        print(f"Found {len(available_cases)} cases in test set.")
        
        # Plot the first available case as an example
        if len(available_cases) > 0:
            example_case = available_cases[0]
            print(f"Plotting case: {example_case}")
            output_img = RESULTS_DIR / f"detailed_plot_{example_case.replace('.', 'p')}_2.png"
            
            plot_case_with_uncertainty(csv_file, example_case, output_img)
            
            # If you are running this in an IDE (like VSCode or PyCharm) or a notebook,
            # you can comment out the output_img parameter above and just pass None 
            # to make it pop up on your screen interactively:
            # plot_case_with_uncertainty(csv_file, example_case, output_path=None)
    else:
        print(f"✗ CSV file not found at {csv_file}")
        print(f"Did you run src/train.py first to generate predictions?")

if __name__ == "__main__":
    main()