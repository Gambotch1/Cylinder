#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viv_analysis.self_excitation import amplitude_envelope, dominant_freq


ROOT = Path("results")

UR = 5.5
FN = 0.2
DT = 0.005

CFD_AD = 0.350372
CFD_F = 0.216406

AMP_TOL = 0.10
STATIONARITY_TOL = 0.10
FREQ_BIN_TOL = 1.0

# "Same endpoint" criterion across all five starts for a given arm.
MAX_ENDPOINT_SPREAD_FRAC = 0.05

ARMS = [
    "dim_context",
    "nd_nocontext",
    "nd_context",
]

SEEDS = [
    ("below_040", 0.40, 0.1401488),
    ("below_055", 0.55, 0.1927046),
    ("below_070", 0.70, 0.2452604),
    ("above_130", 1.30, 0.4554836),
    ("above_150", 1.50, 0.5255580),
]


def scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if np.ndim(value) == 0:
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    raise ValueError(f"{key} is not scalar: shape={value.shape}")


def trajectory_metrics(h: np.ndarray, D: float) -> dict:
    h = np.asarray(h, dtype=np.float64)

    if h.ndim != 1 or len(h) < 100:
        raise ValueError(f"Invalid trajectory shape: {h.shape}")

    _, envelope = amplitude_envelope(
        h=h,
        D=D,
        dt=DT,
        fn=FN,
    )

    if len(envelope) < 4:
        raise RuntimeError("Trajectory is too short for an envelope-tail analysis")

    envelope_tail = envelope[int(0.7 * len(envelope)) :]
    sample_tail = h[int(0.7 * len(h)) :]

    ad_final = float(np.mean(envelope_tail))
    tail_cv = float(
        np.std(envelope_tail) / max(abs(ad_final), 1e-30)
    )

    ad_range = float(
        (sample_tail.max() - sample_tail.min()) / (2.0 * D)
    )
    ad_rms = float(np.sqrt(2.0) * np.std(sample_tail) / D)

    f_dominant = float(
        dominant_freq(
            h=h,
            dt=DT,
            fn=FN,
        )
    )

    fft_tail_n = len(h[int(0.6 * len(h)) :])
    fft_bin = float(1.0 / (fft_tail_n * DT))

    return {
        "ad_final": ad_final,
        "tail_cv": tail_cv,
        "ad_range": ad_range,
        "ad_rms": ad_rms,
        "f_dominant": f_dominant,
        "fft_bin": fft_bin,
    }


rows = []

for arm in ARMS:
    for seed_label, seed_ratio, expected_seed_ad in SEEDS:
        path = ROOT / (
            f"gate3_attractor_Ur{UR}_{arm}_{seed_label}.npz"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            h = np.asarray(data["h"], dtype=np.float64)
            D = float(scalar(data, "D"))
            stored_seed_ad = float(scalar(data, "ad_seed"))

            stored_arm = (
                str(scalar(data, "gate3_arm"))
                if "gate3_arm" in data.files
                else arm
            )
            stored_label = (
                str(scalar(data, "seed_label"))
                if "seed_label" in data.files
                else seed_label
            )

        if stored_arm != arm:
            raise ValueError(
                f"{path}: arm metadata={stored_arm}, expected={arm}"
            )

        if stored_label != seed_label:
            raise ValueError(
                f"{path}: seed label={stored_label}, expected={seed_label}"
            )

        if not np.isclose(
            stored_seed_ad,
            expected_seed_ad,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{path}: stored seed={stored_seed_ad}, "
                f"expected={expected_seed_ad}"
            )

        metrics = trajectory_metrics(h, D)

        amp_error = float(
            abs(metrics["ad_final"] - CFD_AD) / CFD_AD
        )
        freq_error = float(
            abs(metrics["f_dominant"] - CFD_F)
        )

        amplitude_pass = bool(amp_error <= AMP_TOL)
        stationary_pass = bool(
            metrics["tail_cv"] <= STATIONARITY_TOL
        )
        frequency_pass = bool(
            np.isfinite(metrics["f_dominant"])
            and freq_error
            <= FREQ_BIN_TOL * metrics["fft_bin"] + 1e-12
        )

        if seed_ratio < 1.0:
            direction = "from_below"
            direction_pass = bool(
                metrics["ad_final"] > stored_seed_ad
            )
        else:
            direction = "from_above"
            direction_pass = bool(
                metrics["ad_final"] < stored_seed_ad
            )

        individual_pass = bool(
            amplitude_pass
            and stationary_pass
            and frequency_pass
            and direction_pass
        )

        rows.append(
            {
                "arm": arm,
                "seed_label": seed_label,
                "direction": direction,
                "seed_ratio": seed_ratio,
                "seed_ad": stored_seed_ad,
                **metrics,
                "cfd_ad": CFD_AD,
                "cfd_f": CFD_F,
                "amp_error_frac": amp_error,
                "freq_error_hz": freq_error,
                "amplitude_pass": amplitude_pass,
                "stationary_pass": stationary_pass,
                "frequency_pass": frequency_pass,
                "direction_pass": direction_pass,
                "individual_pass": individual_pass,
                "path": str(path),
            }
        )


print(
    f"{'arm':18s} "
    f"{'seed':>10s} "
    f"{'A/D init':>9s} "
    f"{'A/D final':>10s} "
    f"{'err':>8s} "
    f"{'CV':>8s} "
    f"{'f_dom':>9s} "
    f"{'|df|':>9s} "
    f"{'bin':>9s} "
    f"{'dir':>6s} "
    f"{'status':>8s}"
)

for row in rows:
    print(
        f"{row['arm']:18s} "
        f"{row['seed_label']:>10s} "
        f"{row['seed_ad']:9.4f} "
        f"{row['ad_final']:10.4f} "
        f"{100 * row['amp_error_frac']:7.2f}% "
        f"{100 * row['tail_cv']:7.2f}% "
        f"{row['f_dominant']:9.5f} "
        f"{row['freq_error_hz']:9.5f} "
        f"{row['fft_bin']:9.5f} "
        f"{'OK' if row['direction_pass'] else 'FAIL':>6s} "
        f"{'PASS' if row['individual_pass'] else 'FAIL':>8s}"
    )


arm_summaries = []

print("\nArm-level Gate 3 decision")
print("-------------------------")

for arm in ARMS:
    arm_rows = [row for row in rows if row["arm"] == arm]
    final_ad = np.asarray(
        [row["ad_final"] for row in arm_rows],
        dtype=np.float64,
    )
    final_f = np.asarray(
        [row["f_dominant"] for row in arm_rows],
        dtype=np.float64,
    )

    endpoint_spread = float(np.ptp(final_ad))
    endpoint_spread_frac = float(endpoint_spread / CFD_AD)

    all_individual_pass = all(
        row["individual_pass"] for row in arm_rows
    )
    has_below = any(
        row["direction"] == "from_below" for row in arm_rows
    )
    has_above = any(
        row["direction"] == "from_above" for row in arm_rows
    )

    common_endpoint_pass = bool(
        endpoint_spread_frac <= MAX_ENDPOINT_SPREAD_FRAC
    )

    gate3_pass = bool(
        all_individual_pass
        and common_endpoint_pass
        and has_below
        and has_above
    )

    summary = {
        "arm": arm,
        "n_runs": len(arm_rows),
        "mean_final_ad": float(np.mean(final_ad)),
        "std_final_ad": float(np.std(final_ad)),
        "min_final_ad": float(np.min(final_ad)),
        "max_final_ad": float(np.max(final_ad)),
        "endpoint_spread": endpoint_spread,
        "endpoint_spread_frac_of_cfd": endpoint_spread_frac,
        "mean_final_frequency": float(np.mean(final_f)),
        "all_individual_pass": all_individual_pass,
        "common_endpoint_pass": common_endpoint_pass,
        "has_below": has_below,
        "has_above": has_above,
        "gate3_pass": gate3_pass,
    }
    arm_summaries.append(summary)

    print(
        f"{arm:18s} "
        f"mean A/D={summary['mean_final_ad']:.5f}  "
        f"spread={100 * endpoint_spread_frac:.2f}% CFD  "
        f"mean f={summary['mean_final_frequency']:.5f} Hz  "
        f"status={'PASS' if gate3_pass else 'FAIL'}"
    )


csv_path = ROOT / "gate3_Ur5.5_trajectory_summary.csv"

fieldnames = list(rows[0].keys())
with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


json_path = ROOT / "gate3_Ur5.5_summary.json"

json_path.write_text(
    json.dumps(
        {
            "Ur": UR,
            "fn": FN,
            "dt": DT,
            "cfd_target_ad": CFD_AD,
            "cfd_target_f": CFD_F,
            "criteria": {
                "amplitude_relative_tolerance": AMP_TOL,
                "stationarity_cv_tolerance": STATIONARITY_TOL,
                "frequency_fft_bins_tolerance": FREQ_BIN_TOL,
                "common_endpoint_spread_fraction_of_cfd":
                    MAX_ENDPOINT_SPREAD_FRAC,
            },
            "trajectories": rows,
            "arms": arm_summaries,
        },
        indent=2,
    )
)


# One convergence figure per arm.
for arm in ARMS:
    fig, ax = plt.subplots(figsize=(10, 5))

    for seed_label, _, _ in SEEDS:
        path = ROOT / (
            f"gate3_attractor_Ur{UR}_{arm}_{seed_label}.npz"
        )

        with np.load(path, allow_pickle=False) as data:
            h = np.asarray(data["h"], dtype=np.float64)
            D = float(scalar(data, "D"))

        env_t, env_ad = amplitude_envelope(
            h=h,
            D=D,
            dt=DT,
            fn=FN,
        )

        ax.plot(
            env_t,
            env_ad,
            linewidth=1.2,
            label=seed_label,
        )

    ax.axhline(
        CFD_AD,
        linestyle="--",
        linewidth=1.4,
        label=f"CFD target A/D={CFD_AD:.4f}",
    )

    ax.axhspan(
        CFD_AD * (1.0 - AMP_TOL),
        CFD_AD * (1.0 + AMP_TOL),
        alpha=0.12,
        label="±10% amplitude band",
    )

    ax.set_xlabel("Free-run time [s]")
    ax.set_ylabel("Envelope A/D")
    ax.set_title(
        f"Gate 3 two-sided attractor test — Ur={UR} — {arm}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figure_path = ROOT / (
        f"gate3_attractor_Ur{UR}_{arm}.png"
    )
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure: {figure_path}")


print(f"\nSaved trajectory table: {csv_path}")
print(f"Saved Gate 3 receipt:   {json_path}")