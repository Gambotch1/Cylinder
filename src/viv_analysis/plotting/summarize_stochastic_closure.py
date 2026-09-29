#!/usr/bin/env python3
"""
Aggregate the stochastic-closure probe's coupled-loop runs (sec:bridge_
stochastic_closure) into one summary table: A*, c_exc/c, near-fn energy
fraction, mean displacement, and stability classification, per noise_mode,
with median+spread across seeds for the stochastic modes (surrogate/white).

Discovers runs by reading each npz's OWN receipt.json (noise_mode,
noise_seed) rather than parsing filenames -- robust to the exact _exp_tag
string convention in coupled_inference.py.

Reuses, not reimplements:
  - closed_loop_metrics.compute_case_metrics for A*/f_osc/stability
    (the SAME function that produced every other closed_loop_metrics.csv
    in this project, so these numbers are directly comparable).
  - plot_bridge_aerodynamic_work.compute_aerodynamic_work for c_exc/c.
  - the structural band convention (0.5*fn-1.5*fn) for the
    near-fn energy fraction, applied here to h (displacement) rather than
    C_L, since the question this probe answers is whether the INJECTED
    forcing restores near-resonant STRUCTURAL response, not just lift
    spectral content.

Usage:
    python -m viv_analysis.plotting.summarize_stochastic_closure \
        --stochastic_dir gru_bridge_nd_context_noacc_stochastic_closure_coupled_eval \
        --baseline_npz results/gru_bridge_nd_context_noacc_final22_coupled_eval/coupled_bridge_Ur6.7385_gru_bridge_nd_context_noacc_forc-v1_additive_noise-none_nd_scale1_s1_seed0_handoff_2000_muNone.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from viv_analysis.closed_loop_metrics import compute_case_metrics, classify_stability
from viv_analysis.config import config
from viv_analysis.plotting.plot_bridge_aerodynamic_work import compute_aerodynamic_work
from viv_analysis.utils import PROJECT_ROOT

WINDOW_FRAC = 0.5  # matches compute_case_metrics's own default, for consistency


def near_fn_energy_fraction(h: np.ndarray, dt: float, fn: float, window_frac: float = WINDOW_FRAC) -> float:
    """Fraction of h's (displacement) PSD variance lying in the structural/
    lock-in band [0.5*fn, 1.5*fn] -- same band convention as plot_bridge_aerodynamic_work.py,
    applied to displacement (not C_L) over the same tail window compute_
    case_metrics uses."""
    n = len(h)
    h_w = h[int((1.0 - window_frac) * n):]
    fs = 1.0 / dt
    nper = min(max(256, 1 << int(np.floor(np.log2(len(h_w) // 4 + 1)))), len(h_w))
    f, P = welch(h_w, fs=fs, nperseg=nper, detrend="constant", window="hann")
    band = (f >= 0.5 * fn) & (f < 1.5 * fn)
    tot = np.trapezoid(P, f)
    return float(np.trapezoid(P[band], f[band]) / (tot + 1e-30)) if tot > 0 else float("nan")


def one_run_metrics(npz_path: Path) -> dict:
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())
    d = np.load(npz_path, allow_pickle=True)
    t, h = d["t"], d["h"]
    dt = float(np.median(np.diff(t)))
    fn = config["bridge_fn_hz"]

    cm = compute_case_metrics(npz_path, window_frac=WINDOW_FRAC)
    aero = compute_aerodynamic_work(npz_path)
    n = len(h)
    h_w = h[int((1.0 - WINDOW_FRAC) * n):]

    return dict(
        noise_mode=receipt["noise_mode"],
        noise_seed=receipt.get("noise_seed"),
        npz=npz_path.name,
        A_star=cm.get("surrogate_A_star"),
        cfd_A_star=cm.get("cfd_A_star"),
        A_star_rel_error=cm.get("A_star_rel_error"),
        f_osc=cm.get("surrogate_f_osc"),
        stability_label=cm.get("surrogate_label"),
        c_exc_over_c=aero["c_exc_over_c"],
        near_fn_energy_frac=near_fn_energy_fraction(h, dt, fn),
        mean_displacement_m=float(np.mean(h_w)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stochastic_dir", required=True,
                   help="results/-relative dir containing the surrogate/white/replay coupled npz files")
    p.add_argument("--baseline_npz", required=True,
                   help="Path to the existing noise_mode=none coupled npz (not regenerated)")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    stoch_dir = PROJECT_ROOT / "results" / args.stochastic_dir
    npz_paths = sorted(stoch_dir.glob("coupled_bridge_Ur*.npz"))
    if not npz_paths:
        raise SystemExit(f"No coupled_bridge_Ur*.npz found in {stoch_dir}")

    rows = [one_run_metrics(Path(args.baseline_npz))]
    for npz_path in npz_paths:
        print(f"  {npz_path.name}")
        rows.append(one_run_metrics(npz_path))

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv) if args.out_csv else stoch_dir / "stochastic_closure_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n")

    metric_cols = ["A_star", "A_star_rel_error", "f_osc", "c_exc_over_c",
                   "near_fn_energy_frac", "mean_displacement_m"]
    print(f"{'mode':12s} {'n':>2s} " + " ".join(f"{c:>18s}" for c in metric_cols))
    for mode, g in df.groupby("noise_mode"):
        if len(g) == 1:
            vals = " ".join(f"{g[c].iloc[0]:18.6g}" for c in metric_cols)
            print(f"{mode:12s} {len(g):2d} {vals}   (single deterministic run)")
        else:
            med = " ".join(f"{g[c].median():18.6g}" for c in metric_cols)
            print(f"{mode:12s} {len(g):2d} {med}   (median of {len(g)})")
            lo = " ".join(f"{g[c].min():18.6g}" for c in metric_cols)
            hi = " ".join(f"{g[c].max():18.6g}" for c in metric_cols)
            print(f"{'  range':12s} {'':2s} " + " ".join(f"[{l.strip()},{h.strip()}]".rjust(18)
                  for l, h in zip(lo.split(), hi.split())))
    stab = df.groupby("noise_mode")["stability_label"].value_counts()
    print("\nstability_label counts:")
    print(stab)


if __name__ == "__main__":
    main()
