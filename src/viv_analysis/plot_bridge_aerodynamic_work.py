"""
Thesis figure for sec:bridge_aerodynamic_work -- aerodynamic work performed
by the surrogate during CLOSED-LOOP (coupled) inference, at Ur=6.7385
(U=16 m/s, the bridge deck's critical velocity).

REWRITE NOTE: the original version of this figure computed the
"fluctuating" force as F_L'(t) - mean(F_L'(t)) using the mean over the
WHOLE ~680s post-handoff record. That is wrong for this case: the coupled
CL saturates to a near-constant value within ~3.4s of handoff and stays
there for virtually the entire record (see the CL-saturation/attractor-
collapse finding), so the full-record mean is essentially equal to the
saturated plateau value itself, not a meaningful "DC level" for the early,
still-oscillatory part of the trajectory. Subtracting it from the first
few periods does not isolate an oscillatory component -- it just replots
the ramp-up transient into saturation, offset by a constant close to its
own endpoint (verified directly: this produced the "-6000 to 0 N/m,
one-sided, not oscillating around zero" artifact this rewrite fixes).
Confirmed by a domain expert review of the resulting figure -- not a
hypothetical concern.

This version avoids ANY global-record mean or single fixed analysis
window. Three genuinely DC-robust diagnostics instead:

  1. Sliding-window cross-spectral phase of F_L' relative to hdot at f_n
     (open_loop_energy_diagnostics.phase_difference_rad, which itself
     mean-removes only over whatever window it's given -- 2-period
     windows here, stepped by 1 period, so the mean removed is always
     LOCAL to that window). This reproduces, on the CURRENT checkpoint
     and dataset, both the ~6.5-7s exciting-to-damping flip and the
     ~177 degree post-flip antiphase value referenced in earlier
     diagnostic work (rollout_training.py's docstring) -- verified
     directly here, not assumed: periods 0-2 (t=0-6.2s) give -53.6 deg;
     periods 2-4 onward (t>=6.2s) give +173 to +180 deg for several
     periods running.

  2. Zero-phase Butterworth band-pass (0.5fn-1.5fn, same structural-band
     convention diagnose.py already uses elsewhere in this project)
     applied to the RAW force and velocity traces. A band-pass filter
     has no DC component by construction, so this sidesteps the global-
     vs-local-mean question entirely rather than answering it with a
     different mean.

  3. Cycle-resolved work: W_f and W_d integrated over each individual
     structural period using the RAW (non-detrended) signals. This is
     robust to any DC/quasi-steady offset in F_L' for a different reason
     than band-passing -- over one closed period, displacement returns
     close to its own starting value, so a locally-constant force
     component contributes close to zero net work regardless of its
     magnitude (∫ F̄·ḣ dt = F̄·Δh_cycle ≈ 0 when Δh_cycle ≈ 0). Replaces
     the old continuous cumulative-integral panel, which used the same
     flawed global mean as the old top panel.

c_exc is now computed from the band-pass-filtered force/velocity (method
2), not the old global-mean-removed ones -- documented explicitly since
Eq. bridge_excitation_coefficient's overbar notation is otherwise
ambiguous about which convention is meant.

rho, B, fn from viv_analysis.config; c from bridge_structural_params(); U
from Ur*fn*D -- the SAME convention coupled_inference.py itself uses to
convert Ur to a dimensional velocity, not re-derived independently.

Time axes are nondimensionalized by the structural period T_n=1/fn,
(t-t_h)/T_n, consistent with the rest of the chapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt

from viv_analysis.config import bridge_structural_params, config
from viv_analysis.open_loop_energy_diagnostics import phase_difference_rad
from viv_analysis.plot_style import (
    MODEL_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN, apply_thesis_style,
)

DEFAULT_NPZ = (
    "results/gru_bridge_nd_context_noacc_final22_coupled_eval/"
    "coupled_bridge_Ur6.7385_gru_bridge_nd_context_noacc_forc-v1_additive_"
    "noise-none_nd_scale1_s1_seed0_handoff_2000_muNone.npz"
)

STRUCTURAL_BAND_FRAC = (0.5, 1.5)  # x fn, matches diagnose.py's convention


def sliding_phase(t: np.ndarray, F_L: np.ndarray, h_dot: np.ndarray, fn: float,
                   window_periods: float = 2.0, stride_periods: float = 1.0) -> dict:
    """phase(F_L rel. h_dot) at f_n over successive windows (each mean-
    removed LOCALLY by phase_difference_rad, never over the full record).
    Returns arrays keyed by window CENTER time (relative to t[0])."""
    Tn = 1.0 / fn
    win_s, stride_s = window_periods * Tn, stride_periods * Tn
    t_rel = t - t[0]
    t_max = t_rel[-1]

    centers, phases_deg, n_samples = [], [], []
    start = 0.0
    while start + win_s <= t_max:
        m = (t_rel >= start) & (t_rel < start + win_s)
        if m.sum() > 10:
            phi = phase_difference_rad(t[m], F_L[m], h_dot[m], fn)
            centers.append(start + win_s / 2.0)
            phases_deg.append(float(np.degrees(phi)))
            n_samples.append(int(m.sum()))
        start += stride_s
    phases_deg = np.array(phases_deg)
    # phase_difference_rad wraps to (-180, 180] deg, where +180 and -180 are
    # the SAME antiphase condition -- plotting the signed value produces an
    # artificial jump whenever noise pushes the estimate across that
    # branch cut. phase_lag_deg = |phase_deg| in [0, 180] is single-valued:
    # 0 deg = force and velocity in phase (aerodynamic excitation), 180 deg
    # = antiphase (aerodynamic damping). This is the quantity plotted.
    return {"t_center": np.array(centers), "phase_deg": phases_deg,
            "phase_lag_deg": np.abs(phases_deg), "n_samples": np.array(n_samples)}


def cycle_resolved_work(t: np.ndarray, F_L: np.ndarray, h_dot: np.ndarray, c: float,
                        fn: float) -> dict:
    """W_f, W_d integrated over each individual structural period, using
    the RAW (non-detrended) signals -- see module docstring for why this
    is DC-robust without needing any mean-removal choice."""
    Tn = 1.0 / fn
    t_rel = t - t[0]
    n_cycles = int(t_rel[-1] / Tn)

    ends, W_f, W_d = [], [], []

    for k in range(n_cycles):
        m = (t_rel >= k * Tn) & (t_rel < (k + 1) * Tn)
        if m.sum() < 2:
            continue

        t_k, F_k, hd_k = t[m], F_L[m], h_dot[m]
        ends.append((k + 1.0) * Tn)
        W_f.append(float(trapezoid(F_k * hd_k, t_k)))
        W_d.append(float(trapezoid(c * hd_k**2, t_k)))

    return {
        "t_end": np.asarray(ends),
        "W_f": np.asarray(W_f),
        "W_d": np.asarray(W_d),
    }


def compute_aerodynamic_work(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())

    t, h_dot, cl = d["t"], d["h_dot"], d["cl"]
    Ur = float(d["Ur"])
    D = float(d["D"])

    t_h_receipt = float(receipt["handoff_time_s"])
    assert abs(float(t[0]) - t_h_receipt) < 1e-6, (
        f"{npz_path.name}: t[0]={t[0]!r} != receipt handoff_time_s="
        f"{t_h_receipt!r} -- the whole-trajectory-is-post-handoff assumption "
        f"this figure relies on does not hold for this npz; integrate from "
        f"the correct handoff index instead of t[0].")

    rho = config["bridge_rho"]
    B = config["bridge_B_ref"]
    fn = config["bridge_fn_hz"]
    c = bridge_structural_params()["c"]
    U = Ur * fn * D
    dt = float(np.median(np.diff(t)))

    F_L = 0.5 * rho * U**2 * B * cl  # raw, full record -- NOT mean-removed

    # -- (1) sliding-window cross-spectral phase --
    phase = sliding_phase(t, F_L, h_dot, fn)

    # -- (2) zero-phase band-pass around fn -- DC-free by construction --
    fs = 1.0 / dt
    lo, hi = STRUCTURAL_BAND_FRAC[0] * fn, STRUCTURAL_BAND_FRAC[1] * fn
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    F_L_bp = sosfiltfilt(sos, F_L)
    h_dot_bp = sosfiltfilt(sos, h_dot)
    c_exc = float(np.mean(F_L_bp * h_dot_bp) / np.mean(h_dot_bp**2))
    c_exc_over_c = c_exc / c

    # -- (3) cycle-resolved work, raw signals --
    cyc = cycle_resolved_work(t, F_L, h_dot, c, fn)

    return {
        "t": t, "t_rel": t - t[0], "F_L": F_L, "F_L_bp": F_L_bp,
        "h_dot": h_dot, "h_dot_bp": h_dot_bp,
        "phase": phase, "cycle": cyc,
        "c_exc": c_exc, "c_exc_over_c": c_exc_over_c,
        "Ur": Ur, "U": U, "D": D, "B": B, "rho": rho, "c": c, "fn": fn,
    }


def plot_aerodynamic_work(result: dict, out_path_stem: Path,
                          bandpass_window_periods: tuple[float, float] = (0.0, 10.0),
                          phase_window_periods: tuple[float, float] = (0.0, 30.0)):
    apply_thesis_style()
    import matplotlib.pyplot as plt

    Tn = 1.0 / result["fn"]
    t_star = result["t_rel"] / Tn

    fig, (ax_phase, ax_bp, ax_cyc) = plt.subplots(
        3, 1, figsize=(TEXT_WIDTH_IN, 7.5),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )

    # (a) sliding-window phase vs time, windowed to periods where the
    # band-passed oscillation amplitude is still well above the residual
    # noise floor -- amplitude decays ~600 N/m (cycle 0.5) to ~5-8 N/m by
    # period 30 as CL saturates, past which the phase estimate is on a
    # signal indistinguishable from noise and becomes uninformative.
    ph = result["phase"]
    lo_ph, hi_ph = phase_window_periods
    ph_mask = (ph["t_center"] / Tn >= lo_ph) & (ph["t_center"] / Tn <= hi_ph)
    ax_phase.plot(ph["t_center"][ph_mask] / Tn, ph["phase_lag_deg"][ph_mask], color=MODEL_COLOR,
                  marker="o", ms=3, lw=1.05)
    ax_phase.axhline(0, color="gray", lw=0.6, ls=":")
    ax_phase.axhline(90, color="gray", lw=0.6, ls="--")
    ax_phase.axhline(180, color="gray", lw=0.6, ls=":")
    trans = ax_phase.get_yaxis_transform()
    ax_phase.text(0.995, 2, "aerodynamic excitation", transform=trans, ha="right",
                  va="bottom", fontsize=7, style="italic", color="dimgray")
    ax_phase.text(0.995, 92, "zero coherent work", transform=trans, ha="right",
                  va="bottom", fontsize=7, style="italic", color="dimgray")
    ax_phase.text(0.995, 183, "aerodynamic damping", transform=trans, ha="right",
                  va="bottom", fontsize=7, style="italic", color="dimgray")
    ax_phase.set_ylabel(r"absolute phase difference [deg]")
    ax_phase.set_xlim(lo_ph, hi_ph)
    ax_phase.set_ylim(-10, 195)
    ax_phase.set_yticks([0, 45, 90, 135, 180])
    ax_phase.annotate("(a)", xy=(-0.1, 1.0), xycoords="axes fraction",
                       fontsize=10, fontweight="bold", va="top")

    # (b) band-pass-filtered force/velocity, RMS-normalized (each by its own
    # RMS over the displayed window) so both sit on ONE dimensionless axis --
    # the point of this panel is the phase relationship, not the dimensional
    # magnitudes (already reported via c_exc and the cycle-resolved work).
    lo_p, hi_p = bandpass_window_periods
    mask = (t_star >= lo_p) & (t_star <= hi_p)
    F_bp_win = result["F_L_bp"][mask]
    hdot_bp_win = result["h_dot_bp"][mask]
    F_norm = F_bp_win / np.sqrt(np.mean(F_bp_win**2))
    hdot_norm = hdot_bp_win / np.sqrt(np.mean(hdot_bp_win**2))
    ax_bp.plot(t_star[mask], F_norm, color=MODEL_COLOR, lw=1.0,
              label=r"$\widetilde{F}_L'$ (band-passed, RMS-normalized)")
    ax_bp.plot(t_star[mask], hdot_norm, color=SECONDARY_COLOR, lw=1.0,
              ls=(0, (4, 2)), label=r"$\dot h$ (band-passed, RMS-normalized)")
    ax_bp.axhline(0, color="black", lw=0.5)
    ax_bp.set_ylabel("Normalised amplitude [-]")
    ax_bp.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1,
                frameon=False, borderaxespad=0.0)
    ax_bp.annotate("(b)", xy=(-0.1, 1.0), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", va="top")

    # (c) cumulative net work W_net(t) = cumsum(W_f) - cumsum(W_d), built
    # from the per-cycle W_f/W_d (each individually DC-robust -- see module
    # docstring), so the running sum is too. Includes cycle 0 (the handoff
    # impulse);
    cyc = result["cycle"]
    W_net = np.cumsum(cyc["W_f"] - cyc["W_d"])

    # Cumulative work is zero at handoff. Each subsequent value belongs at
    # the end of the corresponding complete structural period.
    work_time_star = np.concatenate(([0.0], cyc["t_end"] / Tn))
    W_net_plot = np.concatenate(([0.0], W_net))

    ax_cyc.plot(work_time_star, W_net_plot,
                color=MODEL_COLOR, lw=1.2)
    ax_cyc.axhline(0.0, color="black", lw=0.6)

    peak_idx = int(np.argmax(W_net_plot))
    ax_cyc.plot(work_time_star[peak_idx], W_net_plot[peak_idx],
                marker="o", ms=3.5, color=MODEL_COLOR)

    ax_cyc.annotate(
        "maximum following handoff",
        xy=(work_time_star[peak_idx], W_net_plot[peak_idx]),
        xytext=(16, -10),
        textcoords="offset points",
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    ax_cyc.text(
    -0.10, 1.0, "(c)",
    transform=ax_cyc.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
    clip_on=False,
    )
    ax_cyc.set_ylabel(r"$W_{\mathrm{net}}=W_f-W_d$ [J/m]")
    ax_cyc.set_xlabel(r"$(t-t_{\mathrm{h}})/T_n$")

    fig.align_ylabels([ax_phase, ax_bp, ax_cyc])
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_path", default=DEFAULT_NPZ)
    p.add_argument("--out_dir", default=None,
                    help="Default: <npz's own model dir>/thesis_figures/")
    p.add_argument("--bandpass_window_periods", type=float, nargs=2, default=(0.0, 10.0),
                    help="Structural-period range shown in panel (b), e.g. 0 10 "
                         "to span both the pre-flip and post-flip regimes.")
    p.add_argument("--phase_window_periods", type=float, nargs=2, default=(0.0, 30.0),
                    help="Structural-period range shown in panel (a). Default 0-30: "
                         "the band-passed oscillation amplitude decays ~600 N/m to "
                         "~5-8 N/m over this range as CL saturates, and phase "
                         "estimates beyond it are on a signal at the residual noise "
                         "floor rather than a resolvable oscillation.")
    args = p.parse_args()

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent / "thesis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_aerodynamic_work(npz_path)
    ur_tag = f"{result['Ur']:g}".replace(".", "p")
    stem = out_dir / f"aerodynamic_work_Ur{ur_tag}"
    pdf, png = plot_aerodynamic_work(result, stem,
                                     bandpass_window_periods=tuple(args.bandpass_window_periods),
                                     phase_window_periods=tuple(args.phase_window_periods))

    ph = result["phase"]
    Tn = 1.0 / result["fn"]
    # Identify the flip window: first sliding-window center where the phase
    # LAG (|phase|, single-valued on [0,180], no branch-cut ambiguity) settles above 150 deg.
    post_flip = ph["phase_lag_deg"] > 150
    flip_t_star = float(ph["t_center"][post_flip][0] / Tn) if post_flip.any() else float("nan")

    lo_ph, hi_ph = tuple(args.phase_window_periods)
    cyc = result["cycle"]
    W_f0, W_d0 = cyc["W_f"][0], cyc["W_d"][0]
    W_net = np.cumsum(cyc["W_f"]) - np.cumsum(cyc["W_d"])
    W_net_plot = np.concatenate(([0.0], W_net))
    work_time_star = np.concatenate(([0.0], cyc["t_end"] / Tn))
    peak_idx = int(np.argmax(W_net_plot))

    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"c_exc = {result['c_exc']:.6g} N*s/m^2   c = {result['c']:.6g} N*s/m^2   "
          f"c_exc/c = {result['c_exc_over_c']:.6g}")
    print(f"Phase lag reaches antiphase by (t-t_h)/Tn = {flip_t_star:.2f}  "
          f"(t-t_h = {flip_t_star*Tn:.2f}s)")
    print(f"Cycle 0 (handoff transient): W_f={W_f0:.4g} J/m, W_d={W_d0:.4g} J/m")
    print(
        f"Cumulative net work: peak {W_net_plot[peak_idx]:.4g} J/m at "
        f"(t-t_h)/Tn={work_time_star[peak_idx]:.2f}, "
        f"final {W_net_plot[-1]:.4g} J/m"
    )
    print(f"Cycle-resolved work (cycles 1+): W_f range "
          f"[{cyc['W_f'][1:].min():.4g}, {cyc['W_f'][1:].max():.4g}] J/m, "
          f"W_d range [{cyc['W_d'][1:].min():.4g}, {cyc['W_d'][1:].max():.4g}] J/m")


if __name__ == "__main__":
    main()
