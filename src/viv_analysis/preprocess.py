# src/preprocess.py

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from scipy.signal import savgol_filter
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, parse_ur_label
from viv_analysis.config import config


DIR = PROJECT_ROOT

# ── Directory layout ───────────────────────────────────────────────────────────
# Re=200 cylinder
CYLINDER_DISP_DIR = DIR / "data" / "disp"
CYLINDER_CD_DIR   = DIR / "data" / "cd"
CYLINDER_CL_DIR   = DIR / "data" / "cl"

# Re=1000 cylinder
CYLINDER_RE1000_ROOT     = DIR / "data" / "cylinder_Re_1000"
CYLINDER_RE1000_DISP_DIR = CYLINDER_RE1000_ROOT / "disp"
CYLINDER_RE1000_CD_DIR   = CYLINDER_RE1000_ROOT / "cd"
CYLINDER_RE1000_CL_DIR   = CYLINDER_RE1000_ROOT / "cl"
CYLINDER_RE1000_VEL_DIR  = CYLINDER_RE1000_ROOT / "vel"
CYLINDER_RE1000_FY_DIR   = CYLINDER_RE1000_ROOT / "force"


# Bridge
BRIDGE_DISP_DIR = DIR / "data" / "Bridge" / "disp"
BRIDGE_CM_DIR   = DIR / "data" / "Bridge" / "cm"
BRIDGE_CL_DIR   = DIR / "data" / "Bridge" / "cl"
BRIDGE_VEL_DIR  = DIR / "data" / "Bridge" / "vel"
BRIDGE_FY_DIR   = DIR / "data" / "Bridge" / "force"

BASE_DTYPES  = {"step": "int32", "time": "float32"}
VALUE_DTYPE  = "float32"
# Number of initial time-steps to drop from each case to remove impulsive start
# Can be overridden with environment variable VIV_INITIAL_TRIM (e.g. export VIV_INITIAL_TRIM=50)
INITIAL_TRIM_STEPS = int(os.getenv("VIV_INITIAL_TRIM", "100"))


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def read_out_files(filepath: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a Fluent .out file → DataFrame(step, val, time)."""
    try:
        data    = []
        started = False
        with open(filepath, "r") as f:
            for line in f:
                s = line.strip()
                if not started:
                    if s and (s[0].isdigit()
                              or (s[0] == "-" and len(s) > 1 and s[1].isdigit())):
                        started = True
                    else:
                        continue
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        data.append([int(float(parts[0])),
                                     float(parts[1]),
                                     float(parts[2])])
                    except ValueError:
                        continue

        fname = os.path.basename(str(filepath))
        df    = pd.DataFrame(data, columns=["step", "val", "time"])
        if df.empty:
            return df.astype({**BASE_DTYPES, "val": VALUE_DTYPE}), fname
        df = (df.astype({**BASE_DTYPES, "val": VALUE_DTYPE})
                .sort_values(["time", "step"]))
        return df, fname

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return (pd.DataFrame(columns=["step", "val", "time"])
                  .astype({**BASE_DTYPES, "val": VALUE_DTYPE}),
                os.path.basename(str(filepath)))



def extract_case_name(filepath: str | Path) -> str:
    """
    Extract a canonical Ur label from a filename.
    """
    stem = Path(filepath).stem

    ur_match = re.search(r"[Uu][Rr][_\-]?([0-9]+(?:\.[0-9]+)?)", stem)
    if ur_match:
        return format_ur_label(float(ur_match.group(1)))

    if "-" in stem:
        return stem.rsplit("-", 1)[1]

    if "_" in stem:
        return stem.rsplit("_", 1)[1]

    return stem


def _resolve_data_dirs(dataset: str) -> tuple[Path, Path, Path]:
    ds = dataset.strip().lower()
    if ds == "bridge":
        return BRIDGE_DISP_DIR, BRIDGE_CM_DIR, BRIDGE_CL_DIR
    if ds in {"cylinder1000", "cylinder_re_1000", "re1000", "cylinder-re-1000"}:
        return CYLINDER_RE1000_DISP_DIR, CYLINDER_RE1000_CD_DIR, CYLINDER_RE1000_CL_DIR
    return CYLINDER_DISP_DIR, CYLINDER_CD_DIR, CYLINDER_CL_DIR


def _normalize_bridge_cases_to_ur(df: pd.DataFrame, fn_hz: float, d_ref: float) -> pd.DataFrame:

    if fn_hz <= 0 or d_ref <= 0:
        raise ValueError("fn_hz and d_ref must be positive.")
    out   = df.copy()
    speed = pd.to_numeric(out["case"].astype(str), errors="coerce")
    bad   = int(speed.isna().sum())
    if bad:
        raise ValueError(
            f"{bad} bridge case labels could not be parsed as velocities. "
            "Expected filenames like 'disp-16.11.out'."
        )
    out["case"] = (speed / float(fn_hz * d_ref)).map(format_ur_label).astype("string")
    return out


def empty_case_frame(value_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["case", "step", "time", value_name]).astype(
        {"case": "string", **BASE_DTYPES, value_name: VALUE_DTYPE}
    )


def read_out_directory(directory: Path, value_name: str) -> pd.DataFrame:
    files = sorted(directory.glob("*.out"))
    if not files:
        print(f"No .out files found in: {directory}")
        return empty_case_frame(value_name)

    frames = []
    for f in files:
        df, _ = read_out_files(str(f))
        if not df.empty:
            case_df = df[["step", "time", "val"]].copy()
            case_df.insert(0, "case", extract_case_name(f))
            # drop initial transient steps
            if INITIAL_TRIM_STEPS > 0:
                case_df = case_df.iloc[INITIAL_TRIM_STEPS:].reset_index(drop=True)
            if case_df.empty:
                continue
            frames.append(case_df)

    if not frames:
        return empty_case_frame(value_name)

    combined = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["case", "time", "step"])
          .drop_duplicates(subset=["case", "step", "time"], keep="last")
          .reset_index(drop=True)
    )
    return (combined.rename(columns={"val": value_name})
                    .astype({"case": "string", **BASE_DTYPES, value_name: VALUE_DTYPE}))


def downsample(df: pd.DataFrame, every_n: int) -> pd.DataFrame:
    """Keep every nth row per case, preserving case boundaries."""
    if every_n <= 1:
        return df.copy()
    pos = df.groupby("case", sort=False).cumcount()
    return df[pos % every_n == 0].reset_index(drop=True)


# ── Public API ─────────────────────────────────────────────────────────────────

def merge_dataframes(
    dataset: str   | None = None,
    fn_hz:   float | None = None,
    d_ref:   float | None = None,
    convert_bridge_to_ur: bool = True,
) -> pd.DataFrame:
    """
    Load and merge disp / cd / cl .out files for the requested dataset.

    Parameters
    ----------
    dataset : "cylinder" (default), "cylinder1000", or "bridge"
    fn_hz   : bridge natural frequency [Hz]  — required for bridge
    d_ref   : bridge reference depth [m]     — required for bridge
    convert_bridge_to_ur : if True, convert m/s case labels to Ur labels
    """
    ds = (dataset or os.getenv("VIV_DATASET", "cylinder")).strip().lower()
    disp_dir, cd_dir, cl_dir = _resolve_data_dirs(ds)

    disp_df = read_out_directory(disp_dir, "disp")
    cd_df   = read_out_directory(cd_dir,   "cd")
    cl_df   = read_out_directory(cl_dir,   "cl")

    empty = pd.DataFrame(
        columns=["case", "step", "time", "disp", "cd", "cl"]
    ).astype({"case": "string", **BASE_DTYPES,
              "disp": VALUE_DTYPE, "cd": VALUE_DTYPE, "cl": VALUE_DTYPE})

    if disp_df.empty or cd_df.empty or cl_df.empty:
        print(f"WARNING: one or more signal directories empty for dataset='{ds}'")
        return empty

    df = (disp_df
          .merge(cd_df,  on=["case", "step", "time"], how="inner", validate="one_to_one")
          .merge(cl_df,  on=["case", "step", "time"], how="inner", validate="one_to_one")
          .sort_values(["case", "time", "step"])
          .reset_index(drop=True))

    if ds == "bridge" and convert_bridge_to_ur:
        if fn_hz is None:
            env = os.getenv("BRIDGE_FN_HZ")
            fn_hz = float(env) if env else None
        if d_ref is None:
            env = os.getenv("BRIDGE_D_REF")
            d_ref = float(env) if env else None
        if fn_hz is None or d_ref is None:
            raise ValueError(
                "Bridge dataset selected but fn_hz/d_ref are missing.\n"
                "Call merge_dataframes(dataset='bridge', fn_hz=0.32, d_ref=7.42)."
            )
        df = _normalize_bridge_cases_to_ur(df, fn_hz=fn_hz, d_ref=d_ref)

    return df


def _load_force(dataset: str) -> dict[str, pd.DataFrame]:
    ds = dataset.strip().lower()
    if ds not in {"cylinder1000", "cylinder_re_1000", "re1000", "cylinder-re-1000", "bridge"}:
        return {}

    out = {}
    if ds == "bridge":
        # Bridge vel-/force- files are labelled by RAW SPEED ('force-16.out' -> '16').
        # merge_dataframes converted disp/cd/cl to Ur labels, so convert these the SAME
        # way or the merge in compute_kinematics matches nothing and silently Savgols.
        fn_hz = float(config["bridge_fn_hz"]); d_ref = float(config["bridge_D_ref"])
        if BRIDGE_VEL_DIR.exists():
            v = read_out_directory(BRIDGE_VEL_DIR, "vel")
            if not v.empty:
                out["vel"] = _normalize_bridge_cases_to_ur(v, fn_hz=fn_hz, d_ref=d_ref)
        if BRIDGE_FY_DIR.exists():
            fdf = read_out_directory(BRIDGE_FY_DIR, "force")
            if not fdf.empty:
                out["force"] = _normalize_bridge_cases_to_ur(fdf, fn_hz=fn_hz, d_ref=d_ref)
        return out

    if CYLINDER_RE1000_VEL_DIR.exists():
        v = read_out_directory(CYLINDER_RE1000_VEL_DIR, "vel")
        if not v.empty: out["vel"] = v
    if CYLINDER_RE1000_FY_DIR.exists():
        fdf = read_out_directory(CYLINDER_RE1000_FY_DIR, "force")
        if not fdf.empty: out["force"] = fdf
    return out


def compute_kinematics(df: pd.DataFrame, dataset: str | None = None, 
                       structural_params: dict | None = None,
                       bridge_structural_params: dict | None = None) -> pd.DataFrame:
    """
    Append 'vel' and 'acc' columns computed per case via numerical
    differentiation of the smoothed displacement signal.
    """
    print("Computing velocity and acceleration...")
    df = df.sort_values(["case", "time", "step"]).reset_index(drop=True)

    force_signal = _load_force(dataset) if dataset else {}
    ds = (dataset or "").strip().lower()
    
    # pick the structural params for the force-residual path
    sp = None
    if ds == "bridge":
        sp = bridge_structural_params if bridge_structural_params is not None else None
    elif structural_params is not None:
        sp = structural_params
    
    if force_signal and sp is not None:

        m = sp['m']
        c = sp['c']
        k = sp['k']
        print(f"[compute_kinematics] m={sp['m']:.6e}  "
              f"c={sp['c']:.6e}  k={sp['k']:.6e}")

        if "vel" in force_signal:
            df = df.merge(force_signal["vel"], on=["case", "step", "time"], how="left", validate="one_to_one")
        if "force" in force_signal:
            df = df.merge(force_signal["force"], on=["case", "step", "time"], how="left", validate="one_to_one")
            F_fluid = df["force"].astype("float32")
            y       = df["disp"].astype("float32")
            v       = df["vel"].astype("float32")  # from velocity monitor
            df["acc"] = (F_fluid - c * v - k * y) / float(m)
            df = df.drop(columns=["force"])

        needs_fill = (
            "vel" not in df.columns or df["vel"].isna().any() or
            "acc" not in df.columns or df["acc"].isna().any()
        )
            
        if needs_fill:
            print("  WARNING: some cases missing force data, using Savgol fallback.")
            df = _fill_missing_kinematics_with_savgol(df)

        return df

    return _fill_missing_kinematics_with_savgol(df)

def _fill_missing_kinematics_with_savgol(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vel/acc by Savgol-smoothing disp, then numerical differentiation."""
    velocities, accelerations = [], []
    for _, case_df in df.groupby("case", sort=False):
        t = case_df["time"].to_numpy()
        d = case_df["disp"].to_numpy()
        if len(d) < 11:
            v = np.gradient(d, t)
            a = np.gradient(v, t)
        else:
            d_smooth = savgol_filter(d, window_length=11, polyorder=3)
            v = np.gradient(d_smooth, t)
            a = np.gradient(v, t)
        velocities.extend(v)
        accelerations.extend(a)
 
    if "vel" in df.columns:
        df["vel"] = df["vel"].fillna(pd.Series(velocities, index=df.index))
    else:
        df["vel"] = velocities
    if "acc" in df.columns:
        df["acc"] = df["acc"].fillna(pd.Series(accelerations, index=df.index))
    else:
        df["acc"] = accelerations
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}   Cases: {df['case'].nunique()}")
    print(df.groupby("case", sort=True).size().to_string())


def main() -> None:
    df = merge_dataframes()
    if df.empty:
        print("Empty dataframe.")
        return
    print_summary(df)
    print(df.head(10))


if __name__ == "__main__":
    main()