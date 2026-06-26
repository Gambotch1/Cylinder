# diagnose_off_manifold.py
#
# DAY-1 DIAGNOSTIC — is the coupled divergence an off-manifold (extrapolation)
# failure, or in-distribution error compounding?
#
# It re-runs the coupled GRU->Newmark loop standalone (mirroring _rollout_loss_k:
# full-window re-forward, "current" force timing), and at every step measures how
# far the current dynamical state sits from the states the GRU was trained on.
#
# Distance metric: nearest-neighbour distance to the TRAINING STATE CLOUD in
# scaled (disp, vel, acc) space. NOT Mahalanobis, NOT marginal z-scores — the
# training manifold is a limit-cycle RING, and a Gaussian/marginal view under-
# detects departures from it (e.g. the collapse fixed point: every marginal looks
# normal, but vel~=0 AND acc~=0 never co-occur on the ring). Marginal max|z| is
# logged too, precisely so you can watch it FAIL on the collapse case.
#
# FAITHFULNESS GATE: the script prints its own steady-state A/D and compares it to
# your production coupled_inference number for the same (Ur, offset). If they
# disagree, the standalone loop is NOT reproducing production and the diagnosis is
# untrustworthy — it warns loudly. Known Ur=6.0 references:
#     offset 1500 -> 0.4137   offset 2000 -> 0.0016   offset 2500 -> 0.4554
#
# Usage (matches coupled_inference args):
#   python -m viv_analysis.diagnose_off_manifold \
#       --Ur 6.0 --handoff_offset 2000 \
#       --checkpoint gru_rollout_k10.pt \
#       --model_subdir gru_rollout_cylinder_re_1000 \
#       --cfd_dataset cylinder_re_1000 --total_time 500
#
# Run it on a COLLAPSE case (offset 2000) and a SAWTOOTH case (offset 1500/2500)
# and compare the two distance traces — that contrast is the Day-1 result.

import argparse
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from viv_analysis.config import config
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.train_gru import split_cases
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.coupled_inference import Newmark_beta, warmup_history
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, parse_ur_label

CLOUD_CAP = 200_000     # subsample training cloud for KD-tree speed
# Faithfulness reference is now supplied per-run via --ad_reference. The old
# hardcoded dict held the BROKEN baseline model's A/D and produced false MISMATCH
# warnings on every other checkpoint (e.g. the noise-trained model).


def structural_params():
    D, fn, rho, M_star, zeta = 0.4, 0.2, 1.0, 2.0, 0.007
    m = M_star * rho * (np.pi * D ** 2 / 4.0)
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n ** 2
    c = 2.0 * m * omega_n * zeta
    return dict(D=D, fn=fn, rho=rho, m=m, c=c, k=k)


def load_model(artifact_dir, baseline_dir, checkpoint, device):
    ck = torch.load(artifact_dir / checkpoint, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = VIV_GRU(input_size=4,
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    dropout=config["dropout"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_artifacts(artifact_dir, baseline_dir):
    # scalers from the MODEL dir; ur_stats from the BASELINE dir (project rule)
    def _pick(name, primary, secondary):
        p = primary / name
        return p if p.exists() else secondary / name
    with open(_pick("x_scaler.pkl", artifact_dir, baseline_dir), "rb") as f:
        x_scaler = pickle.load(f)
    with open(_pick("y_scaler.pkl", artifact_dir, baseline_dir), "rb") as f:
        y_scaler = pickle.load(f)
    with open(_pick("ur_stats.pkl", baseline_dir, artifact_dir), "rb") as f:
        ur_stats = pickle.load(f)
    return x_scaler, y_scaler, ur_stats


def build_training_cloud(raw_df, train_cases, release_time, x_scaler):
    """Scaled (disp,vel,acc) points the GRU was trained on: post-release, train cases."""
    pts = []
    for case in sorted(train_cases):
        cdf = raw_df[raw_df["case"] == case].sort_values(["time", "step"])
        if cdf.empty:
            continue
        times = cdf["time"].to_numpy(dtype=np.float32)
        rel_idx = int(np.searchsorted(times, float(release_time[case])))
        phys = cdf[["disp", "vel", "acc"]].to_numpy(dtype=np.float32)[rel_idx:]
        if len(phys):
            pts.append(x_scaler.transform(phys).astype(np.float32))
    cloud = np.vstack(pts)
    if len(cloud) > CLOUD_CAP:
        rng = np.random.default_rng(0)
        cloud = cloud[rng.choice(len(cloud), CLOUD_CAP, replace=False)]
    return cloud


def run_coupled_with_logging(model, init_history, init_state, U, sp,
                             x_scaler, y_scaler, n_steps, dt, device,
                             window_conv="current"):
    """
    Re-runs the coupled loop, logging the scaled state each step.

    window_conv selects which state is pushed back into the window:
      "current" — the PRE-Newmark state h[i] (matches production run_coupled_viv,
                  coupled_inference.py:437, which pushes "the state that drove this
                  step"). This is the faithful setting.
      "next"    — the POST-Newmark state (the earlier, off-by-one standalone behavior).
    On a collapsed solution both agree; on a sustained limit cycle they differ by a
    half-step phase, which shifts A/D. Distance and h are logged for the pushed state
    (what the model actually conditions on — matches production's OOD basis).
    """
    xm = torch.tensor(x_scaler.mean_,  dtype=torch.float32, device=device)
    xs = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    y_mean = float(np.asarray(y_scaler.mean_)[0])
    y_scale = float(np.asarray(y_scaler.scale_)[0])

    window = torch.tensor(init_history, dtype=torch.float32,
                          device=device).unsqueeze(0)          # [1, seq_len, 4]
    h = torch.tensor(init_state["h"], dtype=torch.float32, device=device)
    hd = torch.tensor(init_state["h_dot"], dtype=torch.float32, device=device)
    hdd = torch.tensor(init_state["h_ddot"], dtype=torch.float32, device=device)

    states_scaled = np.empty((n_steps, 3), dtype=np.float32)
    h_phys = np.empty(n_steps, dtype=np.float32)
    cl_phys_log = np.empty(n_steps, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_steps):
            pred, _ = model(window)
            cl_phys = pred.squeeze() * y_scale + y_mean
            F = 0.5 * sp["rho"] * U ** 2 * sp["D"] * cl_phys
            # current (pre-Newmark) state, scaled — this is what production pushes
            cur = ((h - xm[0]) / xs[0], (hd - xm[1]) / xs[1], (hdd - xm[2]) / xs[2])
            h_cur = float(h)

            h, hd, hdd = Newmark_beta(F, h, hd, hdd, dt,
                                      sp["m"], sp["c"], sp["k"])
            nxt = ((h - xm[0]) / xs[0], (hd - xm[1]) / xs[1], (hdd - xm[2]) / xs[2])

            push = cur if window_conv == "current" else nxt
            states_scaled[i] = tuple(float(v) for v in push)
            h_phys[i] = h_cur if window_conv == "current" else float(h)
            cl_phys_log[i] = float(cl_phys)

            new_row = window[:, -1, :].clone()
            new_row[:, 0], new_row[:, 1], new_row[:, 2] = push
            window = torch.cat([window[:, 1:, :], new_row.unsqueeze(1)], dim=1)

    return states_scaled, h_phys, cl_phys_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ur", type=float, default=6.0)
    ap.add_argument("--handoff_offset", type=int, default=2000)
    ap.add_argument("--total_time", type=float, default=500.0)
    ap.add_argument("--checkpoint", type=str, default="gru_rollout_k10.pt")
    ap.add_argument("--model_subdir", type=str, default="gru_rollout_cylinder_re_1000")
    ap.add_argument("--cfd_dataset", type=str, default="cylinder_re_1000")
    ap.add_argument("--window_conv", choices=["current", "next"], default="current",
                    help="Window-feed convention; 'current' matches production run_coupled_viv.")
    ap.add_argument("--noise_std", type=float, default=0.0,
                    help="Input-noise sigma the model was TRAINED with (scaled units). >0 "
                         "thickens the in-manifold threshold to the noised training "
                         "distribution, so a noise-trained model is judged against the "
                         "manifold it actually learned, not the thin clean-CFD ring.")
    ap.add_argument("--ad_reference", type=float, default=None,
                    help="Optional known production A/D for THIS model at this (Ur, offset); "
                         "if given, prints a faithfulness MATCH/MISMATCH check.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 0.005
    sp = structural_params()
    seq_len = 1000

    artifact_dir = PROJECT_ROOT / "results" / args.model_subdir
    baseline_dir = PROJECT_ROOT / "results" / f"gru_{args.cfd_dataset}"

    x_scaler, y_scaler, ur_stats = load_artifacts(artifact_dir, baseline_dir)
    ur_mean, ur_std = ur_stats["mean"], ur_stats["std"]
    model = load_model(artifact_dir, baseline_dir, args.checkpoint, device)
    print(f"ur_stats mean={ur_mean:.4f} std={ur_std:.4f}  "
          f"y_scaler mean={float(y_scaler.mean_[0]):.6f} scale={float(y_scaler.scale_[0]):.6f}")

    # ── CFD data + kinematics ──────────────────────────────────────────────
    raw_df = merge_dataframes(dataset=args.cfd_dataset)
    struct = {"m": sp["m"], "c": sp["c"], "k": sp["k"],
              "cylinder_mass": sp["m"], "c_struct": sp["c"], "k_struct": sp["k"]}
    raw_df = compute_kinematics(raw_df, dataset=args.cfd_dataset, structural_params=struct)

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    cfg = dict(config); cfg["cylinder1000_t_release"] = config["cylinder1000_t_release"]
    train_cases, _, _, release_time = split_cases(all_cases, args.cfd_dataset, cfg)

    # ── training manifold cloud + calibrated threshold ─────────────────────
    print("Building training manifold cloud...")
    cloud = build_training_cloud(raw_df, train_cases, release_time, x_scaler)
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(cloud)
    rng = np.random.default_rng(1)
    samp = cloud[rng.choice(len(cloud), min(20000, len(cloud)), replace=False)]
    if args.noise_std > 0.0:
        # The model was trained on inputs perturbed by N(0, sigma): the manifold it
        # learned is the clean ring THICKENED by sigma. Calibrate "normal" distance to
        # how far a sigma-perturbed training point lands from the clean cloud — i.e.
        # judge the model against the manifold it actually saw, not the thin clean ring.
        perturbed = (samp + rng.normal(0.0, args.noise_std, size=samp.shape)).astype(np.float32)
        d_ref, _ = nn.kneighbors(perturbed, n_neighbors=1)
        inman = d_ref[:, 0]
        cal = f"noised (sigma={args.noise_std})"
    else:
        # Clean model: distance to nearest distinct training neighbour.
        d_self, _ = nn.kneighbors(samp, n_neighbors=2)
        inman = d_self[:, 1]
        cal = "clean"
    thr95, thr99 = np.percentile(inman, [95, 99])
    print(f"Cloud: {len(cloud):,} pts.  Threshold calibration: {cal}.  "
          f"In-manifold NN-dist p95={thr95:.3f} p99={thr99:.3f}")

    # ── warm-start exactly as production ───────────────────────────────────
    case_df = raw_df[raw_df["case"] == format_ur_label(args.Ur)].copy()
    if case_df.empty:
        raise SystemExit(f"No CFD case for Ur={args.Ur}")
    t_release = 400.0 / args.Ur
    init_history, init_state, t_handoff, handoff_idx = warmup_history(
        cfd_case_df=case_df, release_t=t_release, seq_len=seq_len,
        input_cols=["disp", "vel", "acc"], x_scaler=x_scaler,
        use_ur_context=True, ur_value=args.Ur, ur_stats=(ur_mean, ur_std),
        handoff_offset_steps=args.handoff_offset,
    )
    U = args.Ur * sp["fn"] * sp["D"]
    n_steps = int((args.total_time - t_handoff) / dt)
    print(f"Ur={args.Ur}  U={U:.4f}  handoff t={t_handoff:.2f}s  n_steps={n_steps}")

    # ── run + log ──────────────────────────────────────────────────────────
    states, h_phys, cl_phys = run_coupled_with_logging(
        model, init_history, init_state, U, sp, x_scaler, y_scaler, n_steps, dt, device,
        window_conv=args.window_conv)
    print(f"window_conv = {args.window_conv}")
    t = t_handoff + dt * np.arange(n_steps)

    # distances (batched)
    nn_dist, _ = nn.kneighbors(states, n_neighbors=1)
    nn_dist = nn_dist[:, 0]
    marg_maxz = np.abs(states).max(axis=1)

    # ── FAITHFULNESS GATE: A/D vs production ───────────────────────────────
    ss = int(0.7 * n_steps)
    ad = (h_phys[ss:].max() - h_phys[ss:].min()) / (2 * sp["D"])
    print("\n" + "=" * 60)
    print(f"Steady-state A/D (this script) = {ad:.4f}")
    if args.ad_reference is not None:
        ok = abs(ad - args.ad_reference) < 0.02
        print(f"Reference (this model) = {args.ad_reference:.4f}  -> "
              f"{'MATCH, loop is faithful' if ok else '*** MISMATCH — check window_conv / loop ***'}")
    else:
        print("No --ad_reference given; sanity-check A/D against a production run for THIS model.")
    print("=" * 60)

    # ── verdict text ───────────────────────────────────────────────────────
    disp_train_max = float(np.abs(cloud[:, 0]).max())  # scaled
    off = nn_dist > thr99
    frac_off = off.mean()
    t_cross = t[np.argmax(off)] if off.any() else None
    # amplitude divergence: |h/D| exceeds 1.5x the CFD training envelope (~0.26)
    div = np.abs(h_phys / sp["D"]) > 0.40
    t_div = t[np.argmax(div)] if div.any() else None
    print(f"\nCoupled NN-dist: median={np.median(nn_dist):.3f}  p90={np.percentile(nn_dist, 90):.3f}  "
          f"max={nn_dist.max():.3f}   (in-manifold p99={thr99:.3f})")
    print(f"Fraction of steps off-manifold (> p99): {frac_off:.1%}")
    if t_cross is not None:
        print(f"First off-manifold crossing: t={t_cross:.1f}s")
    if t_div is not None:
        print(f"First amplitude divergence (|h/D|>0.40): t={t_div:.1f}s")
    if t_cross is not None and t_div is not None:
        rel = ("PRECEDES" if t_cross < t_div - 1 else
               "COINCIDES with" if abs(t_cross - t_div) <= 1 else "FOLLOWS")
        print(f"=> off-manifold crossing {rel} amplitude divergence.")
    print("Interpretation: with the threshold calibrated to the model's own training "
          "distribution, a stabilized surrogate sits mostly under p99 (low median, brief "
          "excursions at the cycle extremes). A trace that pins far above p99 and stays "
          "there — with amplitude divergence — is genuine off-manifold runaway.")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ax[0].plot(t, h_phys / sp["D"], lw=0.7, color="tab:blue")
    ax[0].axhline(0.26, color="grey", ls=":", lw=1, label="CFD envelope ~0.26")
    ax[0].axhline(-0.26, color="grey", ls=":", lw=1)
    ax[0].set_ylabel("h/D (coupled)"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Off-manifold diagnostic — Ur={args.Ur}, offset={args.handoff_offset}, A/D={ad:.4f}")

    ax[1].plot(t, nn_dist, lw=0.7, color="tab:red")
    ax[1].axhline(thr99, color="black", ls="--", lw=1, label=f"in-manifold p99={thr99:.2f}")
    ax[1].axhline(thr95, color="grey", ls=":", lw=1, label=f"p95={thr95:.2f}")
    ax[1].set_ylabel("NN-dist to train cloud"); ax[1].legend(loc="upper right"); ax[1].grid(alpha=0.3)

    ax[2].plot(t, marg_maxz, lw=0.7, color="tab:green")
    ax[2].axhline(3.0, color="black", ls="--", lw=1, label="|z|=3 (marginal)")
    ax[2].set_ylabel("max |z| (marginal)"); ax[2].set_xlabel("Time [s]")
    ax[2].legend(loc="upper right"); ax[2].grid(alpha=0.3)

    out_png = PROJECT_ROOT / "results" / (
        f"offmanifold_Ur{args.Ur}_offset{args.handoff_offset}_{args.window_conv}_{args.checkpoint}.png")
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"\nSaved plot: {out_png}")

    out_csv = PROJECT_ROOT / "results" / (
        f"offmanifold_Ur{args.Ur}_offset{args.handoff_offset}_{args.window_conv}_{args.checkpoint}.csv")
    np.savetxt(out_csv, np.column_stack([t, h_phys, cl_phys, nn_dist, marg_maxz]),
               delimiter=",", header="time,h,cl,nn_dist,marg_maxz", comments="")
    print(f"Saved log:  {out_csv}")


if __name__ == "__main__":
    main()