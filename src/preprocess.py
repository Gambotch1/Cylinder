import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.signal import savgol_filter


DIR = Path(__file__).resolve().parents[1]

cdDir = DIR / 'data' / 'cd'
clDir = DIR / 'data' / 'cl'
dispDir = DIR / 'data' / 'disp'

BRIDGE_CD_DIR = DIR / 'data' / 'Bridge' / 'CD'
BRIDGE_CL_DIR = DIR / 'data' / 'Bridge' / 'CL'
BRIDGE_DISP_DIR = DIR / 'data' / 'Bridge' / 'disp'

BASE_DTYPES = {
    "step": "int32",
    "time": "float32",
}
VALUE_DTYPE = "float32"


def read_out_files(filepath: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Read a Fluent .out file into a typed dataframe.
    """
    try:
        data = []
        started = False

        with open(filepath, "r") as f:
            for line in f:
                stripped = line.strip()

                # Skip header text until numeric data starts.
                if not started:
                    if stripped and (
                        stripped[0].isdigit()
                        or (stripped[0] == "-" and len(stripped) > 1 and stripped[1].isdigit())
                    ):
                        started = True
                    else:
                        continue

                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        # columns: step, value, flow-time
                        data.append([int(float(parts[0])), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue

        filename = os.path.basename(str(filepath))
        df = pd.DataFrame(data, columns=["step", "val", "time"])
        if df.empty:
            return df.astype({**BASE_DTYPES, "val": VALUE_DTYPE}), filename

        df = df.astype({**BASE_DTYPES, "val": VALUE_DTYPE}).sort_values(by=["time", "step"])
        return df, filename
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame(columns=["step", "val", "time"]).astype(
            {**BASE_DTYPES, "val": VALUE_DTYPE}
        ), os.path.basename(str(filepath))


def extract_case_name(filepath: str | Path) -> str:
    stem = Path(filepath).stem
    if "-" in stem:
        # Works for both "disp-Ur5.0" and "vertical-displacement-16.11".
        return stem.rsplit("-", 1)[1]
    return stem


def _resolve_data_dirs(dataset: str) -> tuple[Path, Path, Path]:
    ds = dataset.strip().lower()
    if ds == "bridge":
        return BRIDGE_DISP_DIR, BRIDGE_CD_DIR, BRIDGE_CL_DIR
    return dispDir, cdDir, clDir


def _format_ur_label(value: float) -> str:
    txt = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"Ur{txt}"


def _normalize_bridge_cases_to_ur(
    df: pd.DataFrame,
    fn_hz: float,
    d_ref: float,
) -> pd.DataFrame:
    if fn_hz <= 0 or d_ref <= 0:
        raise ValueError(
            "Bridge conversion requires fn_hz > 0 and d_ref > 0 for Ur = U / (fn_hz * d_ref)."
        )

    out = df.copy()
    speed = pd.to_numeric(out["case"].astype(str), errors="coerce")
    bad = int(speed.isna().sum())
    if bad:
        raise ValueError(
            f"Failed to parse {bad} Bridge case labels as velocity values. "
            "Expected names like 16.11 in filenames."
        )

    ur = speed / float(fn_hz * d_ref)
    out["case"] = ur.map(_format_ur_label).astype("string")
    return out


def downsample(df: pd.DataFrame, every_n: int) -> pd.DataFrame:
    """Keep every nth row per case, preserving case boundaries."""
    if every_n <= 1:
        return df.copy()

    return (
        df.groupby("case", group_keys=False)
        .apply(lambda group: group.iloc[::every_n])
        .reset_index(drop=True)
    )


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
            frames.append(case_df)

    if not frames:
        return empty_case_frame(value_name)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(by=["case", "time", "step"])
        .drop_duplicates(subset=["case", "step", "time"], keep="last")
        .reset_index(drop=True)
    )
    combined = combined.rename(columns={"val": value_name}).astype(
        {"case": "string", **BASE_DTYPES, value_name: VALUE_DTYPE}
    )
    return combined

def merge_dataframes(
    dataset: str | None = None,
    fn_hz: float | None = None,
    d_ref: float | None = None,
    convert_bridge_to_ur: bool = True,
) -> pd.DataFrame:
    """
    Merge per-case outputs on matching step and time values.

    dataset: "cylinder" (default) or "bridge"
    For bridge data, set fn_hz and d_ref to convert m/s labels to Ur labels.
    """
    ds = (dataset or os.getenv("VIV_DATASET", "cylinder")).strip().lower()
    disp_dir, cd_dir, cl_dir = _resolve_data_dirs(ds)

    disp_df = read_out_directory(disp_dir, "disp")
    cd_df = read_out_directory(cd_dir, "cd")
    cl_df = read_out_directory(cl_dir, "cl")

    if disp_df.empty or cd_df.empty or cl_df.empty:
        return pd.DataFrame(columns=["case", "step", "time", "disp", "cd", "cl"]).astype(
            {
                "case": "string",
                **BASE_DTYPES,
                "disp": VALUE_DTYPE,
                "cd": VALUE_DTYPE,
                "cl": VALUE_DTYPE,
            }
        )
    
    df = disp_df.merge(cd_df, on=["case", "step", "time"], how="inner", validate="one_to_one")
    df = df.merge(cl_df, on=["case", "step", "time"], how="inner", validate="one_to_one")

    df = df.sort_values(by=["case", "time", "step"]).reset_index(drop=True)

    if ds == "bridge" and convert_bridge_to_ur:
        if fn_hz is None:
            env_fn = os.getenv("BRIDGE_FN_HZ")
            fn_hz = float(env_fn) if env_fn else None
        if d_ref is None:
            env_d = os.getenv("BRIDGE_D_REF")
            d_ref = float(env_d) if env_d else None

        if fn_hz is None or d_ref is None:
            raise ValueError(
                "Bridge data selected but fn_hz/d_ref are missing. "
                "Provide merge_dataframes(dataset='bridge', fn_hz=..., d_ref=...) "
                "or set BRIDGE_FN_HZ and BRIDGE_D_REF."
            )

        df = _normalize_bridge_cases_to_ur(df, fn_hz=float(fn_hz), d_ref=float(d_ref))

    return df


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates velocity and acceleration from displacement and time.
    Calculations are strictly grouped by 'case' to avoid differentiating 
    across the boundaries of different simulations.
    """
    print("Computing velocity and acceleration...")
    df = df.sort_values(by=["case", "time", "step"]).reset_index(drop=True)
    
    velocities = []
    accelerations = []
    
    for case_name, case_df in df.groupby("case", sort=False):
        t = case_df["time"].to_numpy()
        d = case_df["disp"].to_numpy()
        
        # Handle cases with insufficient data points for savgol_filter
        if len(d) < 11:
            # Use simple gradient for small datasets
            v = np.gradient(d, t)
            a = np.gradient(v, t)
        else:
            d_smooth = savgol_filter(d, window_length=11, polyorder=3)
            v = np.gradient(d_smooth, t) # type: ignore
            a = np.gradient(v, t) 
        
        velocities.extend(v)
        accelerations.extend(a)
        
    df["vel"] = velocities
    df["acc"] = accelerations
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"Merged dataframe shape: {df.shape}")
    print(f"Cases: {df['case'].nunique()}")

    counts = df.groupby("case", sort=True).size()
    print("Rows per case:")
    print(counts.to_string())



def main() -> None:
    df = merge_dataframes()

    if df.empty:
        print("Merged dataframe is empty.")
        return

    print_summary(df)
    print(df.head(10))


if __name__ == "__main__":
    main()

