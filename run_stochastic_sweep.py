"""
Harness: collect A/D from coupled_*.npz files and plot the three-trace
decision figure.

Usage:
    python run_stochastic_sweep.py coupled_*.npz --out ad_vs_ur.png
"""

import argparse
import json
import numpy as np


def amplitude_over_D(h, D, kind="rms"):
    """One consistent A/D estimator applied to ALL traces (CFD, model, null).
    Uses the steady tail (last 40%) to skip the transient."""
    h = np.asarray(h, float)
    tail = h[int(0.6 * len(h)):]
    if kind == "rms":
        return float(np.std(tail) / D)
    return float((tail.max() - tail.min()) / 2.0 / D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="coupled_*.npz from the sweep")
    ap.add_argument("--out", default="ad_vs_ur.png")
    ap.add_argument("--kind", default="rms", choices=["rms", "pp"])
    args = ap.parse_args()

    rows = {}   # config -> list of (Ur, A/D)
    cfd = {}    # Ur -> A/D from CFD warm-start

    for f in sorted(args.files):
        d = np.load(f)
        Ur = float(d["Ur"])
        D = float(d["D"])
        ad = amplitude_over_D(d["h"], D, args.kind)
        cfd[Ur] = amplitude_over_D(d["h_cfd"], D, args.kind)
        cfg = "GRU+noise"
        if "gruoff" in f:
            cfg = "noise only (null)"
        elif "white" in f:
            cfg = "GRU+white"
        rows.setdefault(cfg, []).append((Ur, ad))

    print(json.dumps(
        {"CFD": cfd,
         **{k: dict(sorted(v)) for k, v in rows.items()}},
        indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    urs = sorted(cfd)
    ax.plot(urs, [cfd[u] for u in urs], "k-o", lw=2, label="CFD (truth)")
    styles = {"GRU+noise": "tab:green", "noise only (null)": "tab:red",
              "GRU+white": "tab:orange"}
    for cfg, pts in rows.items():
        pts = sorted(pts)
        u = [p[0] for p in pts]
        a = [p[1] for p in pts]
        ax.plot(u, a, "--s", color=styles.get(cfg, "gray"), label=cfg)
    ax.axvspan(6.7, 7.3, color="green", alpha=0.07, label="lock-in band (Hallak)")
    ax.set_xlabel("$U_r$")
    ax.set_ylabel(f"A/D ({args.kind})")
    ax.set_title("Stochastic-closure probe: does GRU+measured forcing "
                 "reproduce the lock-in hump?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")
    print("\nREAD: GRU+noise tracks CFD hump AND sits above noise-only "
          "-> motion-induced feedback is real (thesis result).\n"
          "      GRU+noise == noise-only -> pure resonance, GRU decorative.")


if __name__ == "__main__":
    main()
