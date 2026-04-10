import pandas as pd
from pathlib import Path
import os


DIR = Path(__file__).resolve().parents[1]

cdDir = DIR / 'data' / 'cd'
clDir = DIR / 'data' / 'cl'
dispDir = DIR / 'data' / 'disp'

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
        return stem.split("-", 1)[1]
    return stem


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

def merge_dataframes() -> pd.DataFrame:
    """
    Merge per-case outputs on matching step and time values.
    """
    disp_df = read_out_directory(dispDir, "disp")
    cd_df = read_out_directory(cdDir, "cd")
    cl_df = read_out_directory(clDir, "cl")

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

    return df.sort_values(by=["case", "time", "step"]).reset_index(drop=True)


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

