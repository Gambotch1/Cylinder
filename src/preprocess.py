# src/preprocess.py

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from scipy.signal import savgol_filter


DIR = Path(__file__).resolve().parents[1]

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

# Bridge
BRIDGE_DISP_DIR = DIR / "data" / "Bridge" / "disp"
BRIDGE_CM_DIR   = DIR / "data" / "Bridge" / "CM"
BRIDGE_CL_DIR   = DIR / "data" / "Bridge" / "CL"

BASE_DTYPES  = {"step": "int32", "time": "float32"}
VALUE_DTYPE  = "float32"


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


def _format_ur_label(value: float) -> str:
    """Float → canonical 'Ur{value}' label, trailing zeros stripped."""
    txt = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"Ur{txt}"


def extract_case_name(filepath: str | Path) -> str:
    """
    Extract a canonical Ur label from a filename.

    Supported patterns (case-insensitive):
      disp_Ur_4.00.out   → Ur4
      disp-Ur5.0.out     → Ur5
      cl-Ur4.25.out      → Ur4.25
      disp-16.11.out     → '16.11'  (bridge m/s label, converted later)
      disp-Ur5.0.out     → Ur5
    """
    stem = Path(filepath).stem

    # Pattern 1: explicit Ur token (handles Ur_4.00, Ur4.25, ur-5.0 etc.)
    ur_match = re.search(
        r"[Uu][Rr][_\-]?([0-9]+(?:\.[0-9]+)?)", stem
    )
    if ur_match:
        return _format_ur_label(float(ur_match.group(1)))

    # Pattern 2: last token after "-" (e.g. disp-16.11 → '16.11')
    if "-" in stem:
        return stem.rsplit("-", 1)[1]

    # Pattern 3: last token after "_"
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


def _normalize_bridge_cases_to_ur(df: pd.DataFrame,
                                   fn_hz: float,    
                                   d_ref: float) -> pd.DataFrame:
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
    out["case"] = (speed / float(fn_hz * d_ref)).map(_format_ur_label).astype("string")
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
                    .astype({"case": "string",
                             **BASE_DTYPES,
                             value_name: VALUE_DTYPE}))


def downsample(df: pd.DataFrame, every_n: int) -> pd.DataFrame:
    """Keep every nth row per case, preserving case boundaries."""
    if every_n <= 1:
        return df.copy()
    return (df.groupby("case", group_keys=False)
              .apply(lambda g: g.iloc[::every_n])
              .reset_index(drop=True))


# ── Public API ─────────────────────────────────────────────────────────────────

def merge_dataframes(
    dataset: str | None = None,
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
          .merge(cd_df,  on=["case", "step", "time"],
                 how="inner", validate="one_to_one")
          .merge(cl_df,  on=["case", "step", "time"],
                 how="inner", validate="one_to_one")
          .sort_values(["case", "time", "step"])
          .reset_index(drop=True))

    if ds == "bridge" and convert_bridge_to_ur:
        # Resolve fn_hz and d_ref from args → env → error
        if fn_hz is None:
            env = os.getenv("BRIDGE_FN_HZ")
            fn_hz = float(env) if env else None
        if d_ref is None:
            env = os.getenv("BRIDGE_D_REF")
            d_ref = float(env) if env else None
        if fn_hz is None or d_ref is None:
            raise ValueError(
                "Bridge dataset selected but fn_hz/d_ref are missing.\n"
                "Call merge_dataframes(dataset='bridge', fn_hz=0.32, d_ref=7.42) "
                "or set env vars BRIDGE_FN_HZ and BRIDGE_D_REF."
            )
        df = _normalize_bridge_cases_to_ur(df, fn_hz=fn_hz, d_ref=d_ref)

    return df


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 'vel' and 'acc' columns computed per case via numerical
    differentiation of the smoothed displacement signal.
    """
    print("Computing velocity and acceleration...")
    df = df.sort_values(["case", "time", "step"]).reset_index(drop=True)

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

    df["vel"] = velocities
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