"""
Closed-loop amplitude sweep over the lock-in range.

For each Ur in [5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 7]:
  1. Run coupled GRU-structural inference from the standard warm-start window.
  2. Record steady-state A/D (peak-to-peak / 2D on last 30 % of trajectory).
  3. Compare against the CFD A/D computed the same way.

Also runs two diagnostic checks specific to the Ur=5.5 failure:
  D1 – CL consistency: compares coupled CL vs teacher-forcing CL for the
       first 100 steps after handoff to distinguish a model bug from a
       basin-of-attraction issue.
  D2 – Deep warm-start: shifts the warm-up window 50 000 steps (250 s)
       into the fully-developed lock-in regime.  If *that* initial
       condition sustains large oscillations, the failure at Ur=5.5 is
       purely a basin issue (the GRU has the right dynamics but a
       different attractor basin shape than CFD).

Usage:
    python src/run_lockin_sweep.py [--dataset Cylinder1000]
                                   [--total_time 700]
                                   [--no_diagnostics]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── resolve src/ so imports work whether called from repo root or src/ ─────────
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viv_analysis.coupled_inference import warmup_history, run_coupled_viv
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.preprocess import (
    compute_kinematics,
    correct_cl_for_reference_velocity,
    merge_dataframes,
)
from viv_analysis.utils import format_ur_label

ROOT_DIR = _SRC.parent

UR_SWEEP = [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 7.0]

# ── Physical constants (must match UDF / coupled_inference.py exactly) ─────────
RHO   = 1.0
D     = 0.2
FN    = 0.2
M_STAR = 2.0
ZETA   = 0.007
SEQ_LEN = 960
DT      = 0.005
T_STAR_RELEASE = 80.0
INPUT_COLS = ["disp", "vel", "acc"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _structural_params(Ur: float) -> dict:
    U = Ur * FN * D
    m = M_STAR * RHO * (np.pi * D ** 2 / 4.0)
    omega_n = 2.0 * np.pi * FN
    return {
        "U": U,
        "m": m,
        "k": m * omega_n ** 2,
        "c": 2.0 * m * omega_n * ZETA,
    }


def _steady_state_ad(h: np.ndarray, ss_frac: float = 0.3) -> float:
    ss = int((1.0 - ss_frac) * len(h))
    return float((h[ss:].max() - h[ss:].min()) / (2.0 * D))


def _load_artifacts(dataset: str, device: str):
    art = ROOT_DIR / "results" / f"gru_{dataset}"
    with open(art / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(art / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(art / "ur_stats.pkl", "rb") as f:
        ur_info = pickle.load(f)
    with open(art / "metrics_gru.json", "r") as f:
        metrics = json.load(f)

    use_ur = bool(ur_info["use_ur_context"])
    ur_mean = float(ur_info["mean"])
    ur_std  = float(ur_info["std"]) + 1e-8

    input_size  = len(INPUT_COLS) + (1 if use_ur else 0)
    hidden_size = metrics["gru_config"].get("hidden_size", 64)
    num_layers  = metrics["gru_config"].get("num_layers", 2)

    model = VIV_GRU(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=0.1).to(device)
    model.load_state_dict(torch.load(art / "gru_best.pt", map_location=device))
    model.eval()
    print(f"[artifacts] hidden={hidden_size}  layers={num_layers}  "
          f"use_ur={use_ur}  ur_mean={ur_mean:.4f}  ur_std={ur_std:.4f}")
    return model, x_scaler, y_scaler, use_ur, ur_mean, ur_std, art


def _load_cfd(dataset: str) -> pd.DataFrame:
    structural_params = _structural_params(6.0)  # dummy Ur for loading
    raw = merge_dataframes(dataset=dataset)
    raw = correct_cl_for_reference_velocity(raw, fn=FN, d_ref=D)
    raw = compute_kinematics(raw, dataset=dataset, structural_params=structural_params)
    return raw


# ── Single-Ur coupled inference ────────────────────────────────────────────────

def run_coupled_at_ur(
    Ur: float,
    model, x_scaler, y_scaler, use_ur: bool,
    ur_mean: float, ur_std: float,
    raw_df: pd.DataFrame,
    device: str,
    total_time: float = 700.0,
    warmup_offset_s: float = 0.0,
) -> tuple[dict, float, pd.DataFrame]:
    """
    Run coupled inference at *Ur*.  warmup_offset_s shifts the warm-up
    window forward in physical seconds (used for the deep-warmup diagnostic).

    Returns (result_dict, t_handoff, case_df).
    """
    p = _structural_params(Ur)
    t_release = (T_STAR_RELEASE * D / p["U"]) + warmup_offset_s

    label   = format_ur_label(Ur)
    case_df = raw_df[raw_df["case"] == label].copy()
    if case_df.empty:
        raise ValueError(f"No CFD data for case '{label}'")

    init_hist, init_state, t_handoff, _ = warmup_history(
        cfd_case_df=case_df,
        release_t=t_release,
        seq_len=SEQ_LEN,
        input_cols=INPUT_COLS,
        x_scaler=x_scaler,
        use_ur_context=use_ur,
        ur_value=Ur,
        ur_stats=(ur_mean, ur_std),
    )

    n_steps = int((total_time - t_handoff) / DT)
    if n_steps <= 0:
        raise ValueError(f"total_time={total_time} is too short for "
                         f"t_handoff={t_handoff:.2f} at Ur={Ur}")

    result = run_coupled_viv(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        initial_history=init_hist, initial_state=init_state,
        seq_len=SEQ_LEN, input_cols=INPUT_COLS,
        m=p["m"], c=p["c"], k=p["k"],
        rho=RHO, U=p["U"], D=D, dt=DT, n_steps=n_steps,
        use_ur_context=use_ur, ur_value=Ur,
        ur_stats=(ur_mean, ur_std), device=device,
    )
    return result, t_handoff, case_df


# ── Main sweep ─────────────────────────────────────────────────────────────────

def sweep(
    dataset: str,
    total_time: float,
    device: str,
) -> pd.DataFrame:
    model, x_scaler, y_scaler, use_ur, ur_mean, ur_std, art = \
        _load_artifacts(dataset, device)
    raw_df = _load_cfd(dataset)

    rows = []
    for Ur in UR_SWEEP:
        print(f"\n── Ur={Ur} ──────────────────────────────────────────")
        try:
            result, t_handoff, case_df = run_coupled_at_ur(
                Ur, model, x_scaler, y_scaler, use_ur,
                ur_mean, ur_std, raw_df, device, total_time=total_time,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"Ur": Ur, "ad_cfd": float("nan"), "ad_gru": float("nan")})
            continue

        h_gru = result["displacement"]
        ad_gru = _steady_state_ad(h_gru)

        h_cfd = case_df.sort_values("time")["disp"].to_numpy()
        ad_cfd = _steady_state_ad(h_cfd)

        print(f"  t_handoff = {t_handoff:.2f}s   n_steps = {len(h_gru)}")
        print(f"  CFD A/D   = {ad_cfd:.4f}")
        print(f"  GRU A/D   = {ad_gru:.4f}   ratio = {ad_gru/ad_cfd:.3f}")
        rows.append({"Ur": Ur, "ad_cfd": ad_cfd, "ad_gru": ad_gru})

    df = pd.DataFrame(rows)
    out_csv = ROOT_DIR / "results" / f"gru_{dataset}" / "coupled_amplitude_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved sweep results to {out_csv}")
    return df


# ── Ur=5.5 diagnostics ─────────────────────────────────────────────────────────

def _ur_scaled(Ur, ur_mean, ur_std):
    return (Ur - float(ur_mean)) / (float(ur_std) if abs(float(ur_std)) > 0 else 1.0)


def diagnostic_ur55(
    Ur: float,
    model, x_scaler, y_scaler, use_ur: bool,
    ur_mean: float, ur_std: float,
    raw_df: pd.DataFrame,
    device: str,
    total_time: float = 700.0,
):
    print(f"\n{'='*65}")
    print(f"DIAGNOSTICS FOR Ur={Ur}")
    print(f"{'='*65}")

    p        = _structural_params(Ur)
    t_release = T_STAR_RELEASE * D / p["U"]
    label     = format_ur_label(Ur)
    case_df   = (raw_df[raw_df["case"] == label]
                 .copy().sort_values("time").reset_index(drop=True))

    times      = case_df["time"].to_numpy()
    release_idx = int(np.searchsorted(times, t_release))
    handoff_idx = release_idx + SEQ_LEN

    print(f"  t_release={t_release:.2f}s  release_idx={release_idx}  "
          f"handoff_idx={handoff_idx}")

    # ── D1: CL consistency ────────────────────────────────────────────────────
    print(f"\n─── D1: CL consistency (first 100 steps) ───────────────────")

    # Run coupled inference for 100 steps
    init_hist, init_state, t_handoff, _ = warmup_history(
        cfd_case_df=case_df, release_t=t_release, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, x_scaler=x_scaler, use_ur_context=use_ur,
        ur_value=Ur, ur_stats=(ur_mean, ur_std),
    )
    result_100 = run_coupled_viv(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        initial_history=init_hist, initial_state=init_state,
        seq_len=SEQ_LEN, input_cols=INPUT_COLS,
        m=p["m"], c=p["c"], k=p["k"],
        rho=RHO, U=p["U"], D=D, dt=DT, n_steps=100,
        use_ur_context=use_ur, ur_value=Ur,
        ur_stats=(ur_mean, ur_std), device=device,
    )
    coupled_CL = result_100["CL"]    # shape (100,)
    coupled_h  = result_100["displacement"]

    # Teacher-forcing: slide window over CFD data at handoff_idx
    ur_s = _ur_scaled(Ur, ur_mean, ur_std)
    tf_CL = []
    model.eval()
    with torch.no_grad():
        for i in range(100):
            win_start = handoff_idx - SEQ_LEN + i
            win_end   = handoff_idx + i
            win = case_df.iloc[win_start:win_end][INPUT_COLS].to_numpy(dtype=np.float32)
            win_scaled = x_scaler.transform(win)
            if use_ur:
                win_scaled = np.hstack([
                    win_scaled,
                    np.full((SEQ_LEN, 1), ur_s, dtype=np.float32),
                ])
            x_t = torch.from_numpy(win_scaled).unsqueeze(0).to(device)
            cl_s, _ = model(x_t)
            cl = float(y_scaler.inverse_transform([[cl_s.item()]])[0, 0])
            tf_CL.append(cl)

    tf_CL  = np.array(tf_CL)
    cfd_CL = case_df["cl"].to_numpy()
    cfd_h  = case_df["disp"].to_numpy()

    print(f"  {'step':>4}  {'TF CL':>9}  {'Coupled CL':>10}  {'CFD CL':>9}  "
          f"{'|TF-Cpl|':>9}  {'Cpl h/D':>8}  {'CFD h/D':>8}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")
    for i in range(20):
        cfd_cl_i = cfd_CL[handoff_idx + i] if (handoff_idx + i) < len(cfd_CL) else float("nan")
        cfd_h_i  = cfd_h[handoff_idx + i]  if (handoff_idx + i) < len(cfd_h)  else float("nan")
        print(f"  {i:4d}  {tf_CL[i]:+9.5f}  {coupled_CL[i]:+10.5f}  "
              f"{cfd_cl_i:+9.5f}  {abs(tf_CL[i]-coupled_CL[i]):9.6f}  "
              f"{coupled_h[i]/D:+8.5f}  {cfd_h_i/D:+8.5f}")

    disp_dev = np.abs(coupled_h - cfd_h[handoff_idx: handoff_idx + 100])
    print(f"\n  Max  |h_GRU - h_CFD| / D over 100 steps = {disp_dev.max()/D:.6f}")
    print(f"  Mean |h_GRU - h_CFD| / D over 100 steps = {disp_dev.mean()/D:.6f}")

    # Interpret: if TF and coupled CL agree → model is self-consistent;
    # difference between coupled h and CFD h explains the divergence.
    max_cl_diff = np.abs(tf_CL - coupled_CL).max()
    if max_cl_diff < 0.05:
        print("\n  >>> CL predictions are CONSISTENT between TF and coupled modes.")
        print("      The model is self-consistent; basin-of-attraction / IC mismatch")
        print("      is the likely explanation for the closed-loop failure.")
    else:
        print(f"\n  >>> CL predictions DIVERGE (max |TF-Cpl| = {max_cl_diff:.4f}).")
        print("      This points to a feature-engineering or scaling bug in the")
        print("      inference path — investigate x_scaler / ur_context handling.")

    # ── D2: deep warm-start ───────────────────────────────────────────────────
    print(f"\n─── D2: Deep warm-start (offset +50 000 steps = +250 s) ────")
    DEEP_OFFSET_STEPS = 50_000
    deep_offset_s = DEEP_OFFSET_STEPS * DT
    print(f"  Shifting warm-up by {deep_offset_s:.1f}s into developed lock-in regime")

    try:
        result_deep, t_handoff_deep, _ = run_coupled_at_ur(
            Ur, model, x_scaler, y_scaler, use_ur,
            ur_mean, ur_std, raw_df, device,
            total_time=total_time, warmup_offset_s=deep_offset_s,
        )
        h_deep = result_deep["displacement"]
        ad_deep = _steady_state_ad(h_deep)
        ad_cfd  = _steady_state_ad(case_df["disp"].to_numpy())
        print(f"  t_handoff (deep) = {t_handoff_deep:.2f}s")
        print(f"  Deep warm-start A/D = {ad_deep:.4f}  "
              f"(CFD A/D = {ad_cfd:.4f}, ratio = {ad_deep/ad_cfd:.3f})")

        if ad_deep > 0.5 * ad_cfd:
            print("\n  >>> Deep warm-start SUSTAINS large oscillations.")
            print("      Conclusion: Ur=5.5 failure is a BASIN-OF-ATTRACTION issue.")
            print("      The GRU has correct dynamics but a narrower/different basin")
            print("      than CFD — the standard warm-start IC lies outside the GRU's")
            print("      lock-in basin, while the deep IC (already on the limit cycle)")
            print("      is inside it.")
        else:
            print(f"\n  >>> Deep warm-start also fails (A/D={ad_deep:.4f}).")
            print("      Conclusion: the GRU has genuinely degraded response at Ur=5.5.")
            print("      This may reflect limited training data near this operating point.")

    except ValueError as exc:
        print(f"  Deep warm-start could not run: {exc}")


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_sweep(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    ax.plot(df["Ur"], df["ad_cfd"],
            color="#1f2937", lw=2.2, marker="o", ms=6,
            label="CFD (ground truth)", zorder=3)
    ax.plot(df["Ur"], df["ad_gru"],
            color="#d97706", lw=2.2, marker="s", ms=5,
            label="GRU closed-loop", zorder=3)

    # error fill
    ax.fill_between(df["Ur"], df["ad_cfd"], df["ad_gru"],
                    alpha=0.12, color="#d97706")

    # label each GRU point with its ratio
    for _, row in df.dropna().iterrows():
        ratio = row["ad_gru"] / row["ad_cfd"] if row["ad_cfd"] > 0 else float("nan")
        if not np.isnan(ratio):
            ax.annotate(
                f"{ratio:.2f}",
                (row["Ur"], row["ad_gru"]),
                xytext=(0, 7), textcoords="offset points",
                fontsize=8.5, ha="center", color="#92400e",
            )

    ax.set_xlim(4.85, 7.15)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Reduced velocity $U_r$", fontsize=13)
    ax.set_ylabel(r"Steady-state $A/D$", fontsize=13)
    ax.set_title(
        f"{dataset} — Lock-in amplitude response (closed-loop GRU vs CFD)",
        fontsize=14, pad=10,
    )
    ax.legend(frameon=False, fontsize=11)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="Cylinder1000")
    parser.add_argument("--total_time",   type=float, default=700.0)
    parser.add_argument("--no_diagnostics", action="store_true",
                        help="Skip Ur=5.5 diagnostics")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Run sweep ──────────────────────────────────────────────────────────────
    df = sweep(args.dataset, args.total_time, device)

    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  {'Ur':>5}  {'CFD A/D':>8}  {'GRU A/D':>8}  {'ratio':>6}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}")
    for _, row in df.iterrows():
        ratio = (row["ad_gru"] / row["ad_cfd"]
                 if not np.isnan(row["ad_gru"]) and row["ad_cfd"] > 0
                 else float("nan"))
        print(f"  {row['Ur']:5.2f}  {row['ad_cfd']:8.4f}  {row['ad_gru']:8.4f}  "
              f"{ratio:6.3f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    out_dir  = ROOT_DIR / "results" / "plots_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png  = out_dir / f"{args.dataset}_coupled_amplitude_response.png"
    plot_sweep(df, args.dataset, out_png)

    # ── Ur=5.5 diagnostics ────────────────────────────────────────────────────
    if not args.no_diagnostics:
        model, x_scaler, y_scaler, use_ur, ur_mean, ur_std, _ = \
            _load_artifacts(args.dataset, device)
        raw_df = _load_cfd(args.dataset)
        diagnostic_ur55(
            Ur=5.5,
            model=model, x_scaler=x_scaler, y_scaler=y_scaler,
            use_ur=use_ur, ur_mean=ur_mean, ur_std=ur_std,
            raw_df=raw_df, device=device,
            total_time=args.total_time,
        )


if __name__ == "__main__":
    main()
