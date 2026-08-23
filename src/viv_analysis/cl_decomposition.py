#!/usr/bin/env python3
"""
CL decomposition: GRU (cl_det) + VdP-like closure (cl_vdp) + residual forcing (e).
===================================================================================
run_coupled_viv logs each additive component of the lift it feeds to Newmark_beta
(coupled_inference.py: CL[i] = cl_det + cl_vdp + e[i]) and every coupled npz already
carries them (cl_det, cl_vdp, e alongside the summed cl). This is a plotting/reporting
job over that existing data -- no rerun needed.

Shows:
  - a short steady-state window with all four traces overlaid, so the reader can see
    how much of the total CL waveform each source actually drives
  - RMS of each component as a fraction of the total CL's RMS (a magnitude ballpark,
    not a strict variance decomposition -- the components are not statistically
    independent, so RMS shares need not sum to 1)
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="coupled_*.npz with cl_det/cl_vdp/e")
    ap.add_argument("--window_s", type=float, default=15.0,
                    help="length of the zoomed steady-state window (from the end of the run)")
    ap.add_argument("--label", default=None, help="title label (default: derived from npz filename)")
    ap.add_argument("--out", default=None, help="output png (default: derived from npz filename)")
    a = ap.parse_args()

    d = np.load(a.npz)
    t, cl, cl_det, cl_vdp, e = (np.asarray(d[k], float) for k in ("t", "cl", "cl_det", "cl_vdp", "e"))
    if not np.allclose(cl, cl_det + cl_vdp + e, atol=1e-4):
        print("[warn] cl != cl_det + cl_vdp + e within 1e-4 -- decomposition may not be exhaustive for this npz")

    label = a.label or a.npz.split("/")[-1].replace(".npz", "")
    out = a.out or a.npz.replace(".npz", "_decomposition.png")

    def stats(x):
        return dict(rms=float(np.sqrt(np.mean(x**2))), peak=float(np.abs(x).max()))

    s_det, s_vdp, s_e, s_cl = stats(cl_det), stats(cl_vdp), stats(e), stats(cl)
    print(f"{'component':<10}{'rms':>10}{'peak':>10}{'rms/rms(cl)':>14}")
    print("-" * 44)
    for name, s in [("cl_det (GRU)", s_det), ("cl_vdp (VdP)", s_vdp), ("e (resid.)", s_e), ("cl (total)", s_cl)]:
        frac = f"{s['rms']/s_cl['rms']:.3f}" if name != "cl (total)" else "1.000"
        print(f"{name:<10}{s['rms']:>10.4f}{s['peak']:>10.4f}{frac:>14}")

    t0 = t[-1] - a.window_s
    m = t >= t0

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.2, 1]})

    ax[0].plot(t[m], cl[m], color="k", lw=1.6, label="cl (total)")
    ax[0].plot(t[m], cl_det[m], color="tab:blue", lw=1.1, label="cl_det (GRU)")
    ax[0].plot(t[m], cl_vdp[m], color="tab:green", lw=1.1, label="cl_vdp (VdP closure)")
    ax[0].plot(t[m], e[m], color="tab:red", lw=1.0, label="e (residual forcing)")
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("$C_L$ contribution")
    ax[0].set_title(f"CL decomposition (steady-state window) — {label}")
    ax[0].legend(loc="upper right", ncol=2); ax[0].grid(alpha=0.3)

    names = ["cl_det\n(GRU)", "cl_vdp\n(VdP)", "e\n(residual)", "cl\n(total)"]
    rms_vals = [s_det["rms"], s_vdp["rms"], s_e["rms"], s_cl["rms"]]
    colors = ["tab:blue", "tab:green", "tab:red", "k"]
    bars = ax[1].bar(names, rms_vals, color=colors, alpha=0.85)
    for b, v in zip(bars, rms_vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("RMS($C_L$ component)")
    ax[1].set_title("Magnitude of each contribution (full run)")
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
