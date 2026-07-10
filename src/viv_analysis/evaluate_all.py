#!/usr/bin/env python3
"""
Sweep over all rollout checkpoints and Ur values.
Compare coupled GRU-struct models vs CFD steady-state amplitudes.
"""
import subprocess
import re
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.utils import PROJECT_ROOT, format_ur_label
from viv_analysis.config import config

# Configuration: (model_subdir, checkpoint_file, label)
CONFIGS = [
    #("gru_cylinder_re_1000",         "gru_best.pt",       "baseline"),
    #("gru_rollout_cylinder_re_1000", "gru_rollout_k1.pt", "k1"),
    #("gru_rollout_cylinder_re_1000", "gru_rollout_k2.pt", "k2"),
    #("gru_rollout_cylinder_re_1000", "gru_rollout_k5.pt", "k5"),
    #("gru_rollout_cylinder_re_1000", "gru_rollout_k10.pt", "k10"),
    ("gru_cylinder_re_1000_noise0.05", "gru_best.pt", "noise0.05"),
]

Ur_LIST = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]

# Physical parameters (must match coupled_inference.py)
D = config['cylinder1000_D_ref']
fn = 0.2
M_star = 2.0
zeta = 0.007
rho = 1.0

m = M_star * rho * (np.pi * D**2 / 4.0)
omega_n = 2.0 * np.pi * fn
k = m * omega_n**2
c = 2.0 * m * omega_n * zeta

STRUCTURAL_PARAMS = {
    "m": m,
    "c": c,
    "k": k,
    "cylinder_mass": m,
    "c_struct": c,
    "k_struct": k,
}


def extract_steady_state_amplitude(stdout: str) -> float | None:
    """Parse 'Steady-state A/D = X.XXXX' from coupled_inference output."""
    match = re.search(r"Steady-state A/D = ([\d.]+)", stdout)
    if match:
        return float(match.group(1))
    return None


def get_cfd_steady_state_amplitude(Ur: float, dataset: str = "cylinder_re_1000") -> float | None:
    """
    Extract CFD steady-state amplitude (A/D) for a given Ur.
    Uses last 30% of CFD trajectory after release.
    """
    raw_df = merge_dataframes(dataset=dataset)
    if raw_df.empty:
        print(f"  WARNING: Could not load CFD data for Ur={Ur}")
        return None
    
    raw_df = compute_kinematics(raw_df, dataset=dataset, structural_params=STRUCTURAL_PARAMS)
    
    case_label = format_ur_label(Ur)
    case_df = raw_df[raw_df["case"] == case_label].copy()
    if case_df.empty:
        print(f"  WARNING: No CFD case found for Ur={Ur}")
        return None
    
    # Get last 30% of trajectory (well into steady-state)
    ss_start = int(0.7 * len(case_df))
    cfd_disp = case_df["disp"].iloc[ss_start:].to_numpy()
    cfd_ad = (cfd_disp.max() - cfd_disp.min()) / (2 * D)
    
    return cfd_ad


def main():
    print("=" * 80)
    print("COUPLED GRU-STRUCTURAL VIV SWEEP")
    print("Evaluating all rollout checkpoints across Ur values")
    print("=" * 80)
    
    # Pre-compute CFD amplitudes
    print("\nPre-computing CFD steady-state amplitudes...")
    cfd_amplitudes = {}
    for Ur in Ur_LIST:
        cfd_amplitudes[Ur] = get_cfd_steady_state_amplitude(Ur)
        if cfd_amplitudes[Ur] is not None:
            print(f"  Ur={Ur}: CFD A/D = {cfd_amplitudes[Ur]:.4f}")
    
    # Run coupled simulations
    results = {}
    total_runs = len(CONFIGS) * len(Ur_LIST)
    run_count = 0
    
    print(f"\nRunning {total_runs} simulations (expect ~1 hours)...")
    print("-" * 80)
    
    for subdir, ckpt, label in CONFIGS:
        for Ur in Ur_LIST:
            run_count += 1
            print(f"[{run_count}/{total_runs}] {label:12s} Ur={Ur} ... ", end="", flush=True)
            
            try:
                out = subprocess.run(
                    [sys.executable, "-m", "src.viv_analysis.coupled_inference",
                     "--model_subdir", subdir,
                     "--checkpoint", ckpt,
                     "--cfd_dataset", "cylinder_re_1000",
                     "--Ur", str(Ur),
                     "--total_time", "200"],  # shorter sweep
                    capture_output=True,
                    text=True,
                    timeout=1200,  # 20 min timeout per run
                )
                
                if out.returncode != 0:
                    print("FAILED (exit code)")
                    err_snip = (out.stderr or "").splitlines()[:40]
                    if err_snip:
                        print("--- STDERR ---")
                        print("\n".join(err_snip))
                        print("--- /STDERR ---")
                    results[(label, Ur)] = None
                else:
                    amp = extract_steady_state_amplitude(out.stdout)
                    if amp is not None:
                        results[(label, Ur)] = amp
                        print(f"A/D={amp:.4f}")
                    else:
                        print("FAILED (no amplitude in output)")
                        out_snip = (out.stdout or "").splitlines()[-80:]
                        if out_snip:
                            print("--- STDOUT (tail) ---")
                            print("\n".join(out_snip))
                            print("--- /STDOUT ---")
                        results[(label, Ur)] = None

            except subprocess.TimeoutExpired as e:
                print("TIMEOUT")
                if getattr(e, "stdout", None):
                    tail = (e.stdout or "").splitlines()[-80:]
                    if tail:
                        print("--- PARTIAL STDOUT (tail) ---")
                        print("\n".join(tail))
                        print("--- /PARTIAL STDOUT ---")
                if getattr(e, "stderr", None):
                    tail = (e.stderr or "").splitlines()[-80:]
                    if tail:
                        print("--- PARTIAL STDERR (tail) ---")
                        print("\n".join(tail))
                        print("--- /PARTIAL STDERR ---")
                results[(label, Ur)] = None
            except Exception as e:
                print(f"ERROR: {e}")
                results[(label, Ur)] = None
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Ur':<6} {'CFD':<10} " + " ".join(f"{lbl:<10}" for _, _, lbl in CONFIGS))
    print("-" * (6 + 10 + 10 * len(CONFIGS)))
    
    for Ur in Ur_LIST:
        cfd_val = cfd_amplitudes.get(Ur)
        cfd_str = f"{cfd_val:.4f}" if cfd_val is not None else "N/A"
        
        row_vals = []
        for _, _, lbl in CONFIGS:
            val = results.get((lbl, Ur))
            row_vals.append(f"{val:.4f}" if val is not None else "FAIL")
        
        print(f"{Ur:<6.1f} {cfd_str:<10} " + " ".join(f"{v:<10}" for v in row_vals))
    
    # Compute error vs CFD for each model
    print("\n" + "=" * 80)
    print("ERROR vs CFD (absolute)")
    print("=" * 80)
    print(f"{'Ur':<6} " + " ".join(f"{lbl:<10}" for _, _, lbl in CONFIGS))
    print("-" * (6 + 10 * len(CONFIGS)))
    
    for Ur in Ur_LIST:
        cfd_val = cfd_amplitudes.get(Ur)
        if cfd_val is None:
            continue
        
        row_vals = []
        for _, _, lbl in CONFIGS:
            val = results.get((lbl, Ur))
            if val is not None:
                error = abs(val - cfd_val)
                row_vals.append(f"{error:.4f}")
            else:
                row_vals.append("FAIL")
        
        print(f"{Ur:<6.1f} " + " ".join(f"{v:<10}" for v in row_vals))
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for _, _, lbl in CONFIGS:
        errors = []
        for Ur in Ur_LIST:
            cfd_val = cfd_amplitudes.get(Ur)
            model_val = results.get((lbl, Ur))
            if cfd_val is not None and model_val is not None:
                errors.append(abs(model_val - cfd_val))
        
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            print(f"{lbl:12s}  Mean error: {mean_error:.4f}  Max error: {max_error:.4f}")
        else:
            print(f"{lbl:12s}  No valid results")
    
    # Save results as CSV
    output_csv = PROJECT_ROOT / "results" / "sweep_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Build dataframe
    rows = []
    for Ur in Ur_LIST:
        row_dict = {"Ur": Ur, "CFD": cfd_amplitudes.get(Ur)}
        for _, _, lbl in CONFIGS:
            row_dict[lbl] = results.get((lbl, Ur))
        rows.append(row_dict)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
