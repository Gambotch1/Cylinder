#!/bin/bash

# ==========================================================
# CYLINDER VIV SWEEP (Laminar - 2D) - 16 CORES OPTIMIZED
# ==========================================================

# --- 1. CONFIGURATION ---
PROJECT_DIR="/scratch/jote9827/Cylinder"  # <-- UPDATE THIS
CASE_NAME="Cylinder_VIV_Project.cas.h5"
UDF_SOURCE="cylinder_viv.c"

cd "${PROJECT_DIR}"

# --- 2. THE REDUCED VELOCITY (UR) LOOP ---
# Testing with 3.0. Once it works, change this back to your full list!
UR_VALUES=(3.0 4.0 4.2 4.4 4.6 4.8 5.0 5.2 5.4 5.6 5.8 6.0 6.5 7.0 8.0 9.0)

for UR in "${UR_VALUES[@]}"
do
    echo "--------------------------------------------------"
    echo "Preparing VIV Simulation for Ur = ${UR}"

    # Create isolated directories for each run
    RUN_DIR="${PROJECT_DIR}/Ur${UR}_Run"
    RESULTS_DIR="${RUN_DIR}/results"
    JOU="${RUN_DIR}/run_Ur${UR}.jou"
    UDF_RUN="${RUN_DIR}/${UDF_SOURCE}"

    mkdir -p "${RUN_DIR}" "${RESULTS_DIR}"

    # Copy ONLY the base case and the UDF.
    cp "${PROJECT_DIR}/${CASE_NAME}" "${RUN_DIR}/"
    cp "${PROJECT_DIR}/${UDF_SOURCE}" "${UDF_RUN}"

    # Dynamically inject the new UR value into the UDF C-code
    sed -i "s/^#define U_REDUCED.*/#define U_REDUCED ${UR}/" "${UDF_RUN}"

    # =================== JOURNAL FILE GENERATION ======================
    cat > "${JOU}" <<EOF
; ---------------------------------------------------------
; Journal for Ur = ${UR} (Laminar 2D VIV)
; ---------------------------------------------------------

; --- 1. COMPILE UDF FIRST ---
/define/user-defined/compiled-functions compile "libudf" yes "${UDF_SOURCE}" "" ""

; --- 2. LOAD UDF INTO MEMORY ---
; We load it into RAM *before* opening the case!
/define/user-defined/compiled-functions load "libudf"

; --- 3. READ THE CASE ---
; Fluent will now smoothly load the mesh because the native Linux 'libudf' is already in memory.
/file/read-case "${CASE_NAME}"

; --- 4. REDIRECT REPORT FILES TO RESULTS FOLDER ---
/solve/report-files/edit/cl-rfile file-name "${RESULTS_DIR}/cl-Ur${UR}.out" q
/solve/report-files/edit/cd-rfile file-name "${RESULTS_DIR}/cd-Ur${UR}.out" q
/solve/report-files/edit/disp-rfile file-name "${RESULTS_DIR}/disp-Ur${UR}.out" q

; --- 5. INITIALIZATION & TIME STEPPING ---
/solve/initialize/hyb-initialization
/solve/set/time-step 0.02

; Run for 400 physical seconds (20,000 steps) with 20 max iterations
/solve/dual-time-iterate 20000 20

; --- 6. SAVE FINAL DATA ---
/file/write-case-data "${RESULTS_DIR}/cylinder_Ur${UR}.cas.h5"
/exit yes
EOF

    echo "Journal ${JOU} created."

    # ===================== LSF JOB SUBMISSION =========================
    bsub <<EOT
#!/bin/bash
#BSUB -q BatchXL
#BSUB -R "mem > 4000 span[hosts=1]"
#BSUB -oo ${RUN_DIR}/bsub_Ur${UR}.out
#BSUB -eo ${RUN_DIR}/bsub_Ur${UR}.err
#BSUB -J VIV_Ur${UR}_16c
#BSUB -n 16
#BSUB -W 48:00
#BSUB -L /bin/bash

cd ${RUN_DIR}
module purge
module load ansys/v2024r1

echo "Starting Fluent Run for Ur = ${UR} on 16 cores..."
fluent 2ddp -t16 -lsf -g -i run_Ur${UR}.jou > ${RUN_DIR}/fluent_output_Ur${UR}.log
echo "Finished."
EOT

    echo "Job for Ur = ${UR} submitted to HPC queue."
done