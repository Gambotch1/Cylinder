import os
import glob
import re
import pandas as pd
import numpy as np

def process_fluent_out(file_path, correction_shift=-0.01258768, max_time=700):
    # 1. Extract cd from the filename (e.g., disp_cd_3.50.out -> 3.50)
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)\.?', os.path.basename(file_path))
    if not match:
        print(f"Skipping {file_path}: Could not find 'cd_X.XX' in filename.")
        return
    
    Ur = float(match.group(1).rstrip('.'))
    
    # 2. Calculate exact release time based on the UDF equations
    # release_time = (80.0 * 0.2) / (Ur * 0.2 * 0.2) = 400.0 / Ur
    release_time = 400.0 / Ur
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    headers = lines[:3]
    
    data = []
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) == 3: 
            data.append([int(parts[0]), float(parts[1]), float(parts[2])])

    df = pd.DataFrame(data, columns=['Time_Step', 'disp', 'flow-time'])

    # 3. Apply the offset ONLY after the release time
    # Using np.where(condition, true_value, false_value)
    new_func(correction_shift, release_time, df)

    # 4. Restrict simulation time to max_time (300s)
    df_truncated = df[df['flow-time'] <= max_time]

    # 5. Re-save into a new .out file
    base_name = os.path.splitext(file_path)[0]
    output_filename = f"{base_name}_corrected.out"
    
    output_lines = headers.copy()
    for _, row in df_truncated.iterrows():
        output_lines.append(f"{int(row['Time_Step'])} {row['disp']:.10e} {row['flow-time']:.6f}\n")

    with open(output_filename, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Processed [{file_path}] | Ur = {Ur} | t_release = {release_time:.2f}s -> Saved to {output_filename}")

def new_func(correction_shift, release_time, df):
    df['disp_corrected'] = np.where(
        df['flow-time'] > release_time,
        df['disp'] + correction_shift, # Apply your exact Fluent offset shift
        df['disp']                     # Keep original (0.0) before release
    )

if __name__ == "__main__":
    # Loop over all .out files in the directory
    for file in glob.glob("data/cylinder_Re_1000/cl/raw/cl_Ur_1*.out"):
        # Prevent re-processing already corrected files
        if not file.endswith("_corrected.out"):  
            process_fluent_out(file, correction_shift=-0.01258768, max_time=700)