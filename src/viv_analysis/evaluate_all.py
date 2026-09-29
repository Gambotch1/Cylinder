#!/usr/bin/env python3
"""
Sweep over model checkpoints and Ur values.
Compare coupled GRU-struct models vs CFD steady-state amplitudes.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.closed_loop_metrics import compute_case_metrics
from viv_analysis.config import (
    bridge_structural_params, config,
    CYLINDER200_ALIASES, cylinder200_structural_params,
)
from viv_analysis.preprocess import compute_kinematics, merge_dataframes
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, parse_ur_label

CLOSURE_LABELS = {
    "v1_additive": "Raw causal GRU (primary baseline)",
    "v2_multiplicative": "Exploratory amplitude-gained residual forcing",
    "v3_coherent": "Exploratory CFD-calibrated amplitude regulator",
}

CYLINDER200_D = config['cylinder200_D_ref']
CYLINDER200_STRUCTURAL_PARAMS = cylinder200_structural_params()

BRIDGE_D = config['bridge_D_ref']
BRIDGE_STRUCTURAL_PARAMS = bridge_structural_params()


def extract_steady_state_amplitude(stdout: str) -> float | None:
    """Parse 'Steady-state A/D = X.XXXX' from coupled_inference output."""
    match = re.search(r"Steady-state A/D = ([\d.]+)", stdout)
    if match:
        return float(match.group(1))
    return None


def extract_npz_path(stdout: str) -> str | None:
    """Parse the saved trajectory path from coupled_inference's
    'Saved coupled trajectory -> <path>' line."""
    match = re.search(r"Saved coupled trajectory -> (\S+)", stdout)
    if match:
        return match.group(1)
    return None


def _compute_gate(npz_path: str, window_frac: float, pass_amp_rel_error_threshold: float) -> dict:
    """Stability + CFD-amplitude gate for one coupled_*.npz, via
    closed_loop_metrics.compute_case_metrics. A flat/stationary envelope
    (classify_stability's 'stationary_lco') is necessary but NOT sufficient
    for 'Pass' -- it can still be a flat, bounded limit cycle at the wrong
    amplitude, so A_star_rel_error vs CFD must also be within threshold.

    Returns {"stability_label", "A_star_rel_error", "pass"}; on any failure
    (e.g. no cl_cfd in the npz, too few cycles to classify) returns
    "pass": False rather than raising, so one bad case doesn't kill the
    sweep.
    """
    try:
        row = compute_case_metrics(npz_path, window_frac=window_frac)
    except Exception as e:
        return {"stability_label": f"error: {e}", "A_star_rel_error": float("nan"),
                "pass": False}

    stability_label = row.get("surrogate_label", "unknown")
    amp_rel_error = row.get("A_star_rel_error", float("nan"))
    is_pass = (
        stability_label == "stationary_lco"
        and np.isfinite(amp_rel_error)
        and abs(amp_rel_error) <= pass_amp_rel_error_threshold
    )
    return {"stability_label": stability_label, "A_star_rel_error": amp_rel_error,
            "pass": bool(is_pass)}


def load_full_cfd_df(dataset: str) -> pd.DataFrame:
    """
    Load + preprocess (merge, kinematics) the CFD dataset ONCE, for reuse
    across every Ur in the sweep. For bridge this goes through the on-disk
    parquet cache (built once, ~1h; instant thereafter) instead of
    re-parsing the raw ~9GB of .out files per call.
    """
    if dataset == "bridge":
        from viv_analysis.preprocess import load_bridge_df_cached
        return load_bridge_df_cached(
            fn_hz=config["bridge_fn_hz"],
            d_ref=config["bridge_D_ref"],
            bridge_structural_params=BRIDGE_STRUCTURAL_PARAMS,
        )
    if dataset.strip().lower() not in CYLINDER200_ALIASES:
        raise ValueError(f"load_full_cfd_df: unsupported dataset '{dataset}'. "
                         f"Only 'bridge' and cylinder200 (aliases: {sorted(CYLINDER200_ALIASES)}) "
                         f"are supported.")
    raw_df = merge_dataframes(dataset=dataset)
    if raw_df.empty:
        return raw_df
    return compute_kinematics(raw_df, dataset=dataset,
                               structural_params=CYLINDER200_STRUCTURAL_PARAMS)


def cfd_steady_state_amplitude(full_cfd_df: pd.DataFrame, Ur: float, D: float) -> float | None:
    """
    Extract CFD steady-state amplitude (A/D) for a given Ur from an
    already-loaded CFD dataframe (see load_full_cfd_df). Uses last 30% of
    CFD trajectory after release.
    """
    case_label = format_ur_label(Ur)
    case_df = full_cfd_df[full_cfd_df["case"] == case_label]
    if case_df.empty:
        print(f"  WARNING: No CFD case found for Ur={Ur}")
        return None

    # Get last 30% of trajectory (well into steady-state)
    ss_start = int(0.7 * len(case_df))
    cfd_disp = case_df["disp"].iloc[ss_start:].to_numpy()
    cfd_ad = (cfd_disp.max() - cfd_disp.min()) / (2 * D)

    return cfd_ad


def bridge_ur_list_from_model(model_subdir: str) -> list[float]:
    """All Ur cases the bridge model's train+val+test split was drawn from.

    Prefers run_config.json; falls back to metrics_gru.json's case_split
    for legacy artifacts (e.g. gru_bridge_noise0.05) that predate
    run_config.json being written at all."""
    model_dir = PROJECT_ROOT / "results" / model_subdir
    run_config_path = model_dir / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path) as f:
            rc = json.load(f)
        cases = rc["train_cases"] + rc["val_cases"] + rc["test_cases"]
        ur_list = sorted({parse_ur_label(c) for c in cases})
    else:
        metrics_path = model_dir / "metrics_gru.json"
        with open(metrics_path) as f:
            m = json.load(f)
        cs = m["case_split"]
        cases = cs["train"] + cs["val"] + cs["test"]
        ur_list = sorted({parse_ur_label(c) for c in cases})

    live_df = merge_dataframes(dataset="bridge", fn_hz=config["bridge_fn_hz"],
                               d_ref=config["bridge_D_ref"])
    live_cases = {parse_ur_label(str(c)) for c in live_df["case"].drop_duplicates()}
    stale = [ur for ur in ur_list if ur not in live_cases]
    if stale:
        print(f"[bridge_ur_list_from_model] Dropping {len(stale)} case(s) from "
              f"'{model_subdir}'s run_config.json that are no longer in the "
              f"current bridge cache (excluded upstream, e.g. 19.5 m/s / "
              f"Ur=8.2126): {stale}")
    return [ur for ur in ur_list if ur in live_cases]


def cylinder200_ur_list_from_model(model_subdir: str) -> list[float]:
    """All Ur cases the cylinder200 model's train+val+test split was drawn
    from (normally all 21, since the split has no holdout)."""
    run_config_path = PROJECT_ROOT / "results" / model_subdir / "run_config.json"
    with open(run_config_path) as f:
        rc = json.load(f)
    cases = rc["train_cases"] + rc["val_cases"] + rc["test_cases"]
    return sorted({parse_ur_label(c) for c in cases})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cylinder200", "bridge"], default="cylinder200")
    parser.add_argument("--model_subdir", default=None,
                    help="Model artifact subdir under results/. Required for both "
                         "--dataset bridge and --dataset cylinder200.")
    parser.add_argument("--label", default=None,
                    help="Column/legend label for this model (default: model_subdir)")
    parser.add_argument("--forcing_mode", default="v1_additive",
                    choices=["v1_additive", "v2_multiplicative", "v3_coherent"])
    parser.add_argument("--mu", type=float, default=None,
                    help="v3_coherent negative-damping strength. Omit to measure it from the "
                         "target CFD growth transient (matches coupled_inference.py's own "
                         "default when --mu isn't passed).")
    parser.add_argument("--total_time", type=float, default=None,
                    help="Sim end time per Ur, seconds (default: 200s cylinder, 500s bridge)")
    parser.add_argument("--t_star_end", type=float, default=None,
                    help="Sim end time as nondimensional t*=tU/D, applied per-Ur (U varies "
                         "with Ur so the equivalent physical seconds differ per case). "
                         "Overrides --total_time when given.")
    parser.add_argument("--window_frac", type=float, default=0.5,
                    help="Fraction of each trajectory's tail used for closed_loop_metrics's "
                         "stability/amplitude/frequency/energy analysis (default 0.5, matching "
                         "closed_loop_metrics.py's own default).")
    parser.add_argument("--pass_amp_rel_error_threshold", type=float, default=0.20,
                    help="A case is only a 'Pass' if it's a stationary_lco AND its cycle-based "
                         "A_star relative error vs CFD is within this fraction (default 0.20, "
                         "i.e. 20%%). A flat-but-wrong-amplitude limit cycle is a Fail.")
    parser.add_argument("--handoff_offset", type=int, default=2000)
    parser.add_argument("--ur_list", default=None,
                    help="Comma-separated Ur override. Default: every Ur in the model's "
                         "recorded train+val+test split (both cylinder200 and bridge).")
    parser.add_argument("--output_dir", default=None,
                    help="Subdir under results/ (or an absolute path) for ALL outputs of this "
                         "sweep: each Ur's npz/png/receipt from coupled_inference.py, plus this "
                         "script's own sweep_results.csv/.png. Default: results/ directly.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    if dataset == "bridge":
        if args.model_subdir is None:
            raise SystemExit("--model_subdir is required for --dataset bridge")
        cfd_dataset = "bridge"
        D = BRIDGE_D
        # Every bridge CFD case tops out at t=300s (several earlier), so a
        # longer default just runs the surrogate past the last point any
        # CFD reference exists for.
        default_total_time = 300.0
        fn_hz = config["bridge_fn_hz"]
        configs = [(args.model_subdir, "gru_best.pt", args.label or args.model_subdir)]
        Ur_list = ([float(x) for x in args.ur_list.split(",")] if args.ur_list
                   else bridge_ur_list_from_model(args.model_subdir))
    else:
        if args.model_subdir is None:
            raise SystemExit("--model_subdir is required for --dataset cylinder200")
        cfd_dataset = "cylinder200"
        D = CYLINDER200_D
        default_total_time = 700.0
        fn_hz = config["cylinder200_fn"]
        configs = [(args.model_subdir, "gru_best.pt", args.label or args.model_subdir)]
        Ur_list = ([float(x) for x in args.ur_list.split(",")] if args.ur_list
                   else cylinder200_ur_list_from_model(args.model_subdir))

    total_time = args.total_time if args.total_time is not None else default_total_time
    # t* = t*U/D with U(Ur) = Ur*fn*D  =>  t[s] = t_star_end / (Ur * fn_hz).
    # Computed per-Ur below (inside the sweep loop) when --t_star_end is set,
    # since U -- and therefore the physical seconds equivalent to a fixed
    # t* -- varies with Ur.
    per_ur_total_time = (
        {Ur: args.t_star_end / (Ur * fn_hz) for Ur in Ur_list}
        if args.t_star_end is not None else None
    )

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    elif Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    closure_label = CLOSURE_LABELS.get(args.forcing_mode, args.forcing_mode)
    print("=" * 80)
    print("COUPLED GRU-STRUCTURAL VIV SWEEP")
    print(f"dataset={dataset}  forcing_mode={args.forcing_mode}  ({closure_label})  "
          f"mu={'measured from CFD' if args.mu is None else args.mu}")
    print(f"{len(Ur_list)} Ur cases: {Ur_list}")
    print(f"output_dir={output_dir}")
    if args.forcing_mode != "v1_additive":
        print(f"NOTE: {closure_label} is not an independently predictive closure -- "
              f"its parameters (mu, a_ref) are measured from each target case's own "
              f"CFD trajectory. Treat as exploratory, not the primary evaluation.")
    print("=" * 80)

    # Pre-compute CFD amplitudes -- load the CFD dataset ONCE and slice it
    # per Ur, instead of re-parsing/re-differentiating the whole dataset
    # (all cases) on every iteration.
    print("\nLoading CFD dataset for the precompute step...")
    full_cfd_df = load_full_cfd_df(cfd_dataset)
    if full_cfd_df.empty:
        print("  WARNING: Could not load CFD data; all CFD amplitudes will be None")

    print("\nPre-computing CFD steady-state amplitudes...")
    cfd_amplitudes = {}
    for Ur in Ur_list:
        cfd_amplitudes[Ur] = (
            cfd_steady_state_amplitude(full_cfd_df, Ur, D) if not full_cfd_df.empty else None
        )
        if cfd_amplitudes[Ur] is not None:
            print(f"  Ur={Ur}: CFD A/D = {cfd_amplitudes[Ur]:.4f}")

    # Run coupled simulations
    results = {}
    gate_results = {}  # (label, Ur) -> {"stability_label", "A_star_rel_error", "pass"} or None
    total_runs = len(configs) * len(Ur_list)
    run_count = 0

    print(f"\nRunning {total_runs} simulations...")
    print("-" * 80)

    for subdir, ckpt, label in configs:
        for Ur in Ur_list:
            run_count += 1
            ur_total_time = per_ur_total_time[Ur] if per_ur_total_time is not None else total_time
            print(f"[{run_count}/{total_runs}] {label:12s} Ur={Ur} "
                  f"(total_time={ur_total_time:.1f}s) ... ", end="", flush=True)

            cmd = [
                sys.executable, "-m", "src.viv_analysis.coupled_inference",
                "--model_subdir", subdir,
                "--checkpoint", ckpt,
                "--cfd_dataset", cfd_dataset,
                "--Ur", str(Ur),
                "--total_time", str(ur_total_time),
                "--handoff_offset", str(args.handoff_offset),
                "--forcing_mode", args.forcing_mode,
                "--output_dir", str(output_dir),
            ]
            if args.mu is not None:
                cmd += ["--mu", str(args.mu)]

            try:
                out = subprocess.run(
                    cmd,
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
                        npz_path = extract_npz_path(out.stdout)
                        gate_results[(label, Ur)] = (
                            _compute_gate(npz_path, args.window_frac,
                                          args.pass_amp_rel_error_threshold)
                            if npz_path is not None else None
                        )
                        gate = gate_results[(label, Ur)]
                        gate_str = (
                            f"  [{gate['stability_label']}, "
                            f"A*_rel_err={gate['A_star_rel_error']:.1%}, "
                            f"{'PASS' if gate['pass'] else 'FAIL'}]"
                            if gate is not None else "  [gate: N/A]"
                        )
                        print(f"A/D={amp:.4f}{gate_str}")
                    else:
                        print("FAILED (no amplitude in output)")
                        out_snip = (out.stdout or "").splitlines()[-80:]
                        if out_snip:
                            print("--- STDOUT (tail) ---")
                            print("\n".join(out_snip))
                            print("--- /STDOUT ---")
                        results[(label, Ur)] = None
                        gate_results[(label, Ur)] = None

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
    print(f"\n{'Ur':<10} {'CFD':<10} " + " ".join(f"{lbl:<10}" for _, _, lbl in configs))
    print("-" * (10 + 10 + 10 * len(configs)))

    for Ur in Ur_list:
        cfd_val = cfd_amplitudes.get(Ur)
        cfd_str = f"{cfd_val:.4f}" if cfd_val is not None else "N/A"

        row_vals = []
        for _, _, lbl in configs:
            val = results.get((lbl, Ur))
            row_vals.append(f"{val:.4f}" if val is not None else "FAIL")

        print(f"{Ur:<10.4f} {cfd_str:<10} " + " ".join(f"{v:<10}" for v in row_vals))

    # Compute error vs CFD for each model
    print("\n" + "=" * 80)
    print("ERROR vs CFD (absolute)")
    print("=" * 80)
    print(f"{'Ur':<10} " + " ".join(f"{lbl:<10}" for _, _, lbl in configs))
    print("-" * (10 + 10 * len(configs)))

    for Ur in Ur_list:
        cfd_val = cfd_amplitudes.get(Ur)
        if cfd_val is None:
            continue

        row_vals = []
        for _, _, lbl in configs:
            val = results.get((lbl, Ur))
            if val is not None:
                error = abs(val - cfd_val)
                row_vals.append(f"{error:.4f}")
            else:
                row_vals.append("FAIL")

        print(f"{Ur:<10.4f} " + " ".join(f"{v:<10}" for v in row_vals))

    # Overall statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for _, _, lbl in configs:
        errors = []
        for Ur in Ur_list:
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

    # Stability + amplitude gate summary -- a flat/stationary envelope alone
    # is NOT "converged": it can be a stable limit cycle at the wrong
    # amplitude. Pass requires stationary_lco AND A_star_rel_error within
    # --pass_amp_rel_error_threshold.
    print("\n" + "=" * 80)
    print(f"STABILITY + AMPLITUDE GATE (threshold={args.pass_amp_rel_error_threshold:.0%})")
    print("=" * 80)

    for _, _, lbl in configs:
        n_pass = n_fail = n_missing = 0
        for Ur in Ur_list:
            gate = gate_results.get((lbl, Ur))
            if gate is None:
                n_missing += 1
            elif gate["pass"]:
                n_pass += 1
            else:
                n_fail += 1
        n_total = n_pass + n_fail + n_missing
        print(f"{lbl:12s}  Pass: {n_pass}/{n_total}  Fail: {n_fail}/{n_total}  "
              f"Missing/error: {n_missing}/{n_total}")

    # Save results as CSV
    output_csv = output_dir / "sweep_results.csv"

    # Build dataframe
    rows = []
    for Ur in Ur_list:
        row_dict = {"Ur": Ur, "CFD": cfd_amplitudes.get(Ur), "closure_mode": closure_label}
        for _, _, lbl in configs:
            row_dict[lbl] = results.get((lbl, Ur))
            gate = gate_results.get((lbl, Ur))
            row_dict[f"{lbl}_stability_label"] = gate["stability_label"] if gate else None
            row_dict[f"{lbl}_A_star_rel_error"] = gate["A_star_rel_error"] if gate else None
            row_dict[f"{lbl}_pass"] = gate["pass"] if gate else False
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Summary plot: closed-loop A/D vs Ur, model(s) vs CFD
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["Ur"], df["CFD"], "o-", color="black", label="CFD")
    for _, _, lbl in configs:
        ax.plot(df["Ur"], df[lbl], "s--", label=lbl)
    ax.set_xlabel("$U_r$")
    ax.set_ylabel("Steady-state $A/D$")
    ax.set_title(f"Coupled GRU-Structural VIV: closed-loop A/D vs $U_r$ ({dataset}, {args.forcing_mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_png = output_dir / "sweep_results.png"
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Summary plot saved to {output_png}")


if __name__ == "__main__":
    main()
