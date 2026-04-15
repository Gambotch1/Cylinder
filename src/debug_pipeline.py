# debug_pipeline.py — run from the Cylinder/ directory as: python src/debug_pipeline.py

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import merge_dataframes, compute_kinematics
from models.elm import build_lookback_dataset, compute_lookback

# ── Step 1: Check raw data ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Raw data")
print("=" * 60)

df = merge_dataframes()
print(f"Shape: {df.shape}")
print(f"Cases: {sorted(df['case'].unique())}")
print(f"Columns: {list(df.columns)}")
print(f"\nPer-case row counts:")
print(df.groupby("case").size().to_string())

print(f"\nSample of CL values:")
for case, cdf in df.groupby("case"):
    cl = cdf["cl"].values
    print(f"  {case}: mean={cl.mean():.4f}  std={cl.std():.4f}  "
          f"min={cl.min():.4f}  max={cl.max():.4f}")

# ── Step 2: Check kinematics ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Kinematics after smoothing")
print("=" * 60)

df = compute_kinematics(df)
for case, cdf in df.groupby("case"):
    d = cdf["disp"].values
    v = cdf["vel"].values
    a = cdf["acc"].values
    print(f"  {case}:")
    print(f"    disp: mean={d.mean():.4f}  std={d.std():.6f}")
    print(f"    vel:  mean={v.mean():.4f}  std={v.std():.6f}")
    print(f"    acc:  mean={a.mean():.4f}  std={a.std():.6f}")
    # Check for NaN or Inf
    for name, arr in [("disp",d),("vel",v),("acc",a)]:
        if np.any(~np.isfinite(arr)):
            print(f"    WARNING: {name} contains NaN or Inf!")

# ── Step 3: Check release timing ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Release timing")
print("=" * 60)

release_time = {v: 60.0 for v in df["case"].unique()}
for case, cdf in df.groupby("case"):
    t = cdf["time"].values
    release_t = release_time[case]
    release_idx = int(np.searchsorted(t, release_t))
    total = len(cdf)
    after_release = total - release_idx
    print(f"  {case}: total={total}  release_idx={release_idx}  "
          f"after_release={after_release}  "
          f"({100*after_release/total:.0f}% of data)")

# ── Step 4: Check lookback dataset ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Lookback dataset")
print("=" * 60)

ur_values = [float(c[2:]) for c in df["case"].unique() if c.startswith("Ur")]
lookback  = min(compute_lookback(ur_values, dt=0.02, D=1.0, U=1.0), 200)
print(f"Lookback: {lookback} steps")

train_cases = {"Ur3.0","Ur5.6","Ur5.0","Ur6.0","Ur6.5","Ur9.0","Ur4.6","Ur5.4"}
train_df    = df[df["case"].isin(train_cases)]

X, y, _ = build_lookback_dataset(
    train_df, lookback,
    target_col="cl",
    input_cols=["disp","vel","acc"],
    release_time=release_time,
    release_window=True,
    include_meta=False,
    stride=5,
)

print(f"X_train shape: {X.shape}  dtype: {X.dtype}")
print(f"y_train shape: {y.shape}  dtype: {y.dtype}")
print(f"y (CL target) stats: mean={y.mean():.4f}  std={y.std():.4f}  "
      f"min={y.min():.4f}  max={y.max():.4f}")

# Check if features and targets are correlated at all
from scipy.stats import pearsonr
# Quick check: correlation of last disp value in window with CL target
last_disp = X[:, -3]   # last timestep, first feature (disp)
r, p      = pearsonr(last_disp, y)
print(f"\nCorrelation of last disp with CL target: r={r:.4f}  p={p:.4e}")

# Check variance of features — zero variance = constant = useless
feature_std = X.std(axis=0)
zero_var    = (feature_std < 1e-8).sum()
print(f"Zero-variance features: {zero_var} / {X.shape[1]}")
print(f"Feature std range: [{feature_std.min():.6f}, {feature_std.max():.6f}]")

# ── Step 5: Sanity check — trivial model baseline ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Baseline comparison")
print("=" * 60)

from sklearn.metrics import r2_score
y_mean_pred = np.full_like(y, y.mean())
print(f"Predicting mean always: R²={r2_score(y, y_mean_pred):.4f}  "
      f"(should be exactly 0.0)")


# Check correlation at every 10th lag for the first feature
print("Correlation of disp at various lags with CL target:")
for lag_idx in range(0, 200, 20):
    r, _ = pearsonr(X[:, lag_idx], y)
    print(f"  disp at lag -{lookback - lag_idx}: r={r:.4f}")

print("\nCorrelation of vel at various lags with CL target:")
for lag_idx in range(200, 400, 20):
    r, _ = pearsonr(X[:, lag_idx], y)
    print(f"  vel at lag -{lookback - (lag_idx-200)}: r={r:.4f}")

# ── Step 6: Plot a single case to visually check alignment ────────────────────
print("\n" + "=" * 60)
print("STEP 6: Visual check — saving plot of Ur5.0")
print("=" * 60)


# Phase portrait: CL vs displacement for Ur5.0 post-release
case_df = df[df["case"] == "Ur5.0"].sort_values("time").reset_index(drop=True)
post    = case_df[case_df["time"] > 60.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Time series overlay
ax = axes[0]
t  = post["time"].values
ax.plot(t, post["disp"].values / post["disp"].std(), 
        lw=0.8, label="disp (normalised)", color="black")
ax.plot(t, post["cl"].values / post["cl"].std(),   
        lw=0.8, label="CL (normalised)",   color="tab:blue", alpha=0.8)
ax.set_xlabel("Time [s]")
ax.set_title("Ur5.0 — disp vs CL (normalised)")
ax.legend()

# Phase portrait
ax = axes[1]
ax.scatter(post["disp"].values, post["cl"].values, 
           s=1, alpha=0.3, color="tab:blue")
ax.set_xlabel("Displacement")
ax.set_ylabel("CL")
ax.set_title("Phase portrait: CL vs disp")

fig.savefig("debug_phase_Ur5p0.png", dpi=150)
plt.close(fig)
print("Saved: debug_phase_Ur5p0.png")



case_df = df[df["case"] == "Ur5.0"].sort_values("time").reset_index(drop=True)
if not case_df.empty:
    t   = case_df["time"].values
    cl  = case_df["cl"].values
    d   = case_df["disp"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                              constrained_layout=True)
    axes[0].plot(t, d, lw=0.6, color="black")
    axes[0].axvline(60.0, color="orange", ls="--", label="release t=60s")
    axes[0].set_ylabel("Displacement")
    axes[0].legend()

    axes[1].plot(t, cl, lw=0.6, color="tab:blue")
    axes[1].axvline(60.0, color="orange", ls="--")
    axes[1].set_ylabel("CL")
    axes[1].set_xlabel("Time [s]")

    fig.suptitle("Ur5.0 — disp and CL time history")
    fig.savefig("debug_Ur5p0.png", dpi=150)
    plt.close(fig)
    print("Saved: debug_Ur5p0.png")
else:
    print("WARNING: Ur5.0 not found in data")


print("\n── Feature correlation analysis ──")
feature_names = []
for col in ["disp", "vel", "acc"]:
    for lag in range(lookback):
        feature_names.append(f"{col}_t-{lookback-lag}")


print("\nDone. Check the output above for anomalies.")