#!/usr/bin/env python3
"""
TF-residual spectral diagnostic for surrogate FSI closure.
=========================================================

Question it answers
-------------------
The teacher-forced (TF) surrogate reproduces CL from TRUE kinematics with
R^2 ~ 0.999. Yet the closed loop collapses. Two competing explanations:

  (H1) EXPOSURE BIAS. The lift IS reproduced given motion across the whole
       band; the closed-loop failure is dynamical (drift to a fixed-point
       attractor). -> scheduled sampling / Flipped Classroom / BPTT-through-
       Newmark are the correct next methods.

  (H2) SPECTRAL SMOOTHING of dynamically-essential broadband lift. The model
       low-passes the (small-variance) broadband / shedding-influenced part
       of CL that R^2 is blind to but that sustains a lightly-damped resonant
       response. -> no exposure-bias schedule restores it; stochastic /
       residual closure is the path. (cf. Abbas & Kavrakov, CompStruct 2020.)

This script computes the evidence that discriminates them:
  * PSD overlay  : CL_true, CL_TF, residual           (where is the miss?)
  * Attenuation  : PSD_TF / PSD_true vs frequency      (does the model low-pass?)
  * Coherence    : gamma^2(CL_true, velocity) vs freq  (is lift motion-determined?)
  * Band metrics : variance fractions + DC bias in/above the structural band
  * Energy proxy : <CL * v> sign (direction of aero work; see caveat in output)

It is signal-processing only (numpy + scipy + matplotlib). It does NOT load
your model -- feed it arrays you already compute (see __main__).

Run a KNOWN-exposure-bias control too (the cylinder): there the residual should
be white, PSD_TF should overlay PSD_true, and coherence should stay high. If it
does, the method is trustworthy and the BRIDGE contrast is meaningful.
"""

import argparse
import json
import numpy as np
from scipy.signal import welch, coherence

EPS = 1e-30


def _common_nperseg(n, fs, fn, min_cycles=8):
    """Welch segment length: power of two, long enough to resolve fn with a few
    cycles per segment, but <= n/4 so we still get averaging."""
    want = int(min_cycles * fs / max(fn, EPS))          # samples for min_cycles of fn
    cap = max(256, n // 4)
    nper = min(max(want, 256), cap)
    # round down to power of two for clean FFT
    nper = 1 << int(np.floor(np.log2(nper)))
    return max(nper, 256)


def _band(f, lo, hi):
    return (f >= lo) & (f < hi)


def _var_fraction(f, P, mask):
    tot = np.trapezoid(P, f)
    return float(np.trapezoid(P[mask], f[mask]) / (tot + EPS))


def analyze(cl_true, cl_tf, vel, dt, fn=0.32, fmax_metric=None,
            skip_s=0.0, label="case"):
    """Core analysis. Arrays are 1-D, same length, uniform dt (seconds)."""
    cl_true = np.asarray(cl_true, float).ravel()
    cl_tf = np.asarray(cl_tf, float).ravel()
    vel = np.asarray(vel, float).ravel()
    n = min(len(cl_true), len(cl_tf), len(vel))
    cl_true, cl_tf, vel = cl_true[:n], cl_tf[:n], vel[:n]

    fs = 1.0 / dt
    nyq = fs / 2.0
    if fmax_metric is None:
        fmax_metric = min(10.0, 0.9 * nyq)

    # drop warm-up / transient if requested
    k0 = int(skip_s * fs)
    cl_true, cl_tf, vel = cl_true[k0:], cl_tf[k0:], vel[k0:]
    n = len(cl_true)
    record_s = n * dt

    resid = cl_true - cl_tf
    dc_bias = float(np.mean(resid))             # pure offset: shifts equilibrium, not cycle energy
    resid_ac = resid - dc_bias

    nper = _common_nperseg(n, fs, fn)
    if record_s < 10.0 / fn:
        print(f"  [warn] record is {record_s:.1f}s (~{record_s*fn:.1f} cycles of fn). "
              f"Run TF over the FULL trajectory for a clean low-frequency spectrum.")

    wk = dict(fs=fs, nperseg=nper, detrend="constant", window="hann")
    f, P_true = welch(cl_true, **wk)
    _, P_tf = welch(cl_tf, **wk)
    _, P_res = welch(resid, **wk)                # AC handled by detrend='constant'
    fc, coh_motion = coherence(cl_true, vel, fs=fs, nperseg=nper)
    _, coh_resid = coherence(resid, vel, fs=fs, nperseg=nper)

    # bands
    s_lo, s_hi = 0.5 * fn, 1.5 * fn              # structural / lock-in band
    b_lo, b_hi = 1.5 * fn, fmax_metric           # broadband / shedding region
    m_struct = _band(f, s_lo, s_hi)
    m_broad = _band(f, b_lo, b_hi)
    mc_struct = _band(fc, s_lo, s_hi)
    mc_broad = _band(fc, b_lo, b_hi)

    ratio = P_tf / (P_true + EPS)               # <1 means TF is attenuated vs true
    atten_struct = float(np.median(ratio[m_struct])) if m_struct.any() else np.nan
    atten_broad = float(np.median(ratio[m_broad])) if m_broad.any() else np.nan

    out = dict(
        label=label,
        n=n, record_s=record_s, fs=fs, nperseg=nper,
        resid_var_frac=float(np.var(resid_ac) / (np.var(cl_true - np.mean(cl_true)) + EPS)),
        dc_bias=dc_bias,
        cl_true_var_in_struct=_var_fraction(f, P_true, m_struct),
        cl_true_var_in_broad=_var_fraction(f, P_true, m_broad),
        resid_var_in_struct=_var_fraction(f, P_res, m_struct),
        resid_var_in_broad=_var_fraction(f, P_res, m_broad),
        atten_struct=atten_struct,         # PSD_TF/PSD_true in lock-in band
        atten_broad=atten_broad,           # PSD_TF/PSD_true in broadband region
        coh_motion_struct=float(np.mean(coh_motion[mc_struct])) if mc_struct.any() else np.nan,
        coh_motion_broad=float(np.mean(coh_motion[mc_broad])) if mc_broad.any() else np.nan,
        coh_resid_struct=float(np.mean(coh_resid[mc_struct])) if mc_struct.any() else np.nan,
        work_true=float(np.mean(cl_true * vel)),   # sign ~ direction of aero work (see caveat)
        work_tf=float(np.mean(cl_tf * vel)),
    )
    spectra = dict(f=f, P_true=P_true, P_tf=P_tf, P_res=P_res, ratio=ratio,
                   fc=fc, coh_motion=coh_motion, coh_resid=coh_resid,
                   fn=fn, fmax=fmax_metric, bands=(s_lo, s_hi, b_lo, b_hi))
    return out, spectra


def verdict(out):
    a_b = out["atten_broad"]
    a_s = out["atten_struct"]
    c_b = out["coh_motion_broad"]
    lines = []
    lines.append("INTERPRETATION")
    lines.append("-" * 60)
    lines.append(f"  residual variance / CL variance : {out['resid_var_frac']*100:6.3f}%   "
                 f"(= 1 - R^2, expect tiny)")
    lines.append(f"  residual DC bias                : {out['dc_bias']:+.4f}   "
                 f"(offset only; shifts equilibrium, not cycle energy)")
    lines.append(f"  PSD_TF/PSD_true  lock-in band   : {a_s:5.2f}   (1.0 = faithful)")
    lines.append(f"  PSD_TF/PSD_true  broadband      : {a_b:5.2f}   (<<1 = model low-passes)")
    lines.append(f"  coherence(CL,vel) lock-in band  : {out['coh_motion_struct']:5.2f}   "
                 f"(~1 = lift is motion-determined)")
    lines.append(f"  coherence(CL,vel) broadband     : {c_b:5.2f}")
    lines.append("")
    # decision tree
    low_pass = (a_b < 0.5)
    broad_incoherent = (c_b < 0.5)
    if low_pass:
        lines.append("  => MODEL LOW-PASSES the lift: broadband content is attenuated.")
        if broad_incoherent:
            lines.append("     Broadband lift is also weakly motion-coherent -> not cleanly")
            lines.append("     learnable from kinematics. This is the H2 / SMOOTHING regime.")
        else:
            lines.append("     Broadband lift is motion-coherent but still attenuated by the")
            lines.append("     model -> a one-step-objective artifact; a spectral/residual loss")
            lines.append("     or residual prediction may recover it.")
        lines.append("     => Exposure-bias schedules (scheduled sampling / Flipped Classroom)")
        lines.append("        will NOT by themselves restore the lost band. Pursue STOCHASTIC /")
        lines.append("        RESIDUAL CLOSURE, or report the amplitude deficit (Abbas framing).")
    else:
        lines.append("  => NO spectral attenuation: TF reproduces CL across the band, yet the")
        lines.append("     loop collapsed -> the failure is DYNAMICAL (attractor / exposure bias).")
        lines.append("     => Scheduled sampling / Flipped Classroom / BPTT-through-Newmark ARE")
        lines.append("        the right methods to read and implement.")
    lines.append("")
    lines.append("  NOTE: <CL*v> sign indicates aero-work direction but is ~tautologically")
    lines.append(f"        positive on true steady data (true={out['work_true']:+.4f}, "
                 f"tf={out['work_tf']:+.4f}); use it only as a consistency check, not proof.")
    return "\n".join(lines)


def make_plots(spectra, out, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, P_true, P_tf, P_res = (spectra[k] for k in ("f", "P_true", "P_tf", "P_res"))
    ratio, fc = spectra["ratio"], spectra["fc"]
    coh_m, coh_r = spectra["coh_motion"], spectra["coh_resid"]
    fn, fmax = spectra["fn"], spectra["fmax"]
    s_lo, s_hi, b_lo, b_hi = spectra["bands"]

    fig, ax = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax[0].semilogy(f, P_true + EPS, color="k", lw=1.4, label="CFD CL (true)")
    ax[0].semilogy(f, P_tf + EPS, color="tab:blue", lw=1.1, label="GRU CL (teacher forcing)")
    ax[0].semilogy(f, P_res + EPS, color="tab:red", lw=1.0, label="residual (true - TF)")
    ax[0].axvline(fn, color="green", ls="--", lw=1, label=f"f_n = {fn} Hz")
    ax[0].axvspan(s_lo, s_hi, color="green", alpha=0.07)
    ax[0].set_ylabel("PSD"); ax[0].set_title(f"TF-residual spectral diagnostic - {out['label']}")
    ax[0].legend(fontsize=9); ax[0].grid(True, which="both", alpha=0.3)

    ax[1].plot(f, ratio, color="tab:purple", lw=1.3)
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].axhline(0.5, color="tab:red", ls=":", lw=1, label="0.5 (low-pass threshold)")
    ax[1].axvline(fn, color="green", ls="--", lw=1)
    ax[1].axvspan(s_lo, s_hi, color="green", alpha=0.07)
    ax[1].set_ylabel("PSD_TF / PSD_true"); ax[1].set_ylim(0, 1.6)
    ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)

    ax[2].plot(fc, coh_m, color="k", lw=1.3, label="coh(CL_true, vel)")
    ax[2].plot(fc, coh_r, color="tab:red", lw=1.0, label="coh(residual, vel)")
    ax[2].axvline(fn, color="green", ls="--", lw=1)
    ax[2].axvspan(s_lo, s_hi, color="green", alpha=0.07)
    ax[2].set_ylabel("coherence $\\gamma^2$"); ax[2].set_xlabel("Frequency [Hz]")
    ax[2].set_ylim(0, 1.05); ax[2].set_xlim(0, fmax)
    ax[2].legend(fontsize=9); ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    print(f"  saved figure -> {path_png}")


def main():
    ap = argparse.ArgumentParser(description="TF-residual spectral diagnostic")
    ap.add_argument("npz", help=".npz with cl_true, cl_tf, vel, dt (and optional fn)")
    ap.add_argument("--fn", type=float, default=0.32, help="structural frequency [Hz]")
    ap.add_argument("--skip_s", type=float, default=0.0, help="seconds to drop at start (transient)")
    ap.add_argument("--fmax", type=float, default=None, help="upper freq for metrics/plot [Hz]")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    d = np.load(f"results/{args.npz}")
    dt = float(d["dt"])
    label = args.label or args.npz.replace(".npz", "")
    out, spectra = analyze(d["cl_true"], d["cl_tf"], d["vel"], dt,
                           fn=args.fn, fmax_metric=args.fmax,
                           skip_s=args.skip_s, label=label)
    print(json.dumps({k: v for k, v in out.items()}, indent=2, default=float))
    print()
    print(verdict(out))
    png = args.out or (args.npz.replace(".npz", "") + "_tf_residual_spectrum.png")
    make_plots(spectra, out, png)


if __name__ == "__main__":
    main()