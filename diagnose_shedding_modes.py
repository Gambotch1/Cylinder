#!/usr/bin/env python3
"""
Bimodal-shedding coherence decomposition (Hallak 2013 framing).
===============================================================

Hallak et al. (2013), Rio-Niteroi deck, report BIMODAL vortex shedding:
  * upper mode  : corner shedding, Strouhal St ~ 0.15-0.16  -> f = St*U/D
  * lower mode  : unsteady shear-layer reattachment (lower frequency)
At lock-in (u0 ~ 58 km/h ~ 16 m/s) the upper mode is captured by the deck
motion at f_n = 0.32 Hz (single merged peak); off lock-in the two coexist and
beat.

This module reports, PER MODE, three numbers:
  * energy fraction : how much of the true CL variance is in that band
                      (GUARD: coherence over a near-empty band is meaningless)
  * coherence(CL,vel): how motion-determined the lift is IN that band
  * attenuation     : PSD_TF/PSD_true in that band (does the model reproduce it)

The argument it makes quantitative: at lock-in, the mode captured by the motion
should show HIGH coherence; the mode acting as exogenous forcing should show
LOW coherence despite carrying real energy. That split IS the "half the
sustaining lift is exogenous" claim, resolved by mode.

Feed it the same npz you built for diagnose_tf_residual_spectrum.py
(cl_true, cl_tf, vel, dt), plus the physical wind speed U for the case.
numpy + scipy + matplotlib.
"""

import argparse
import json
import numpy as np
from scipy.signal import welch, coherence

EPS = 1e-30
D_HEIGHT = 7.42          # deck height (Hallak reference length for St)
ST_UPPER = 0.15          # Hallak upper (corner-shedding) Strouhal number
FN = 0.32                # structural / lock-in frequency [Hz]


def _nperseg(n, fs, f_lo, min_cycles=8):
    want = int(min_cycles * fs / max(f_lo, EPS))
    nper = min(max(want, 256), max(256, n // 4))
    return max(1 << int(np.floor(np.log2(nper))), 256)


def _band_stats(f, P_true, P_tf, fc, coh, lo, hi):
    m = (f >= lo) & (f < hi)
    mc = (fc >= lo) & (fc < hi)
    tot = np.trapezoid(P_true, f) + EPS
    return dict(
        f_lo=float(lo), f_hi=float(hi),
        energy_frac=float(np.trapezoid(P_true[m], f[m]) / tot) if m.any() else 0.0,
        coherence=float(np.mean(coh[mc])) if mc.any() else float("nan"),
        attenuation=float(np.median((P_tf[m] + EPS) / (P_true[m] + EPS)))
        if m.any() else float("nan"),
    )


def analyze_modes(cl_true, cl_tf, vel, dt, U, skip_s=5.0,
                  st_upper=ST_UPPER, fn=FN, D=D_HEIGHT, f_lower_hz=None,
                  rel_halfwidth=0.25, label="case"):
    cl_true = np.asarray(cl_true, float).ravel()
    cl_tf = np.asarray(cl_tf, float).ravel()
    vel = np.asarray(vel, float).ravel()
    nmin = min(len(cl_true), len(cl_tf), len(vel))
    cl_true, cl_tf, vel = cl_true[:nmin], cl_tf[:nmin], vel[:nmin]

    fs = 1.0 / dt
    k0 = int(skip_s * fs)
    cl_true, cl_tf, vel = cl_true[k0:], cl_tf[k0:], vel[k0:]
    n = len(cl_true)

    # Hallak mode centers. Upper (corner) shedding scales with U:
    f_upper = st_upper * U / D
    # Lower (reattachment) mode: use measured value if given, else Hallak's
    # observation that it sits well below the upper mode (~0.5x here as default).
    f_lower = f_lower_hz if f_lower_hz is not None else 0.5 * f_upper

    nper = _nperseg(n, fs, min(fn, f_lower))
    wk = dict(fs=fs, nperseg=nper, detrend="constant", window="hann")
    f, P_true = welch(cl_true, **wk)
    _, P_tf = welch(cl_tf, **wk)
    fc, coh = coherence(cl_true, vel, fs=fs, nperseg=nper)

    def band(center, name):
        lo = center * (1 - rel_halfwidth)
        hi = center * (1 + rel_halfwidth)
        d = _band_stats(f, P_true, P_tf, fc, coh, lo, hi)
        d["name"] = name
        d["center"] = float(center)
        return d

    modes = {
        "structural_fn":   band(fn, "structural / lock-in (f_n=0.32)"),
        "upper_shedding":  band(f_upper, f"upper corner-shedding (St={st_upper})"),
        "lower_shedding":  band(f_lower, "lower reattachment mode"),
    }
    out = dict(label=label, U=float(U), fs=float(fs), n=int(n),
               f_upper=float(f_upper), f_lower=float(f_lower), modes=modes)
    spectra = dict(f=f, P_true=P_true, P_tf=P_tf, fc=fc, coh=coh,
                   fn=fn, f_upper=f_upper, f_lower=f_lower, rel=rel_halfwidth)
    return out, spectra


def report(out):
    L = [f"Bimodal-shedding decomposition — {out['label']}  (U={out['U']} m/s)",
         f"  upper corner-shedding f = {out['f_upper']:.3f} Hz | "
         f"lower reattachment f = {out['f_lower']:.3f} Hz | f_n = 0.32 Hz",
         "-" * 72,
         f"  {'mode':<34}{'E-frac':>9}{'coh':>8}{'atten':>8}   read"]
    for key in ("structural_fn", "upper_shedding", "lower_shedding"):
        m = out["modes"][key]
        ef, co, at = m["energy_frac"], m["coherence"], m["attenuation"]
        if ef < 0.03:
            flag = "band ~empty; ignore coh"
        elif co >= 0.7:
            flag = "MOTION-DETERMINED"
        elif co <= 0.5:
            flag = "EXOGENOUS (not motion)"
        else:
            flag = "mixed"
        L.append(f"  {m['name']:<34}{ef*100:>7.1f}%{co:>8.2f}{at:>8.2f}   {flag}")
    L += ["-" * 72,
          "  READ: at lock-in, a HIGH-coherence mode carrying energy = captured",
          "  by the motion (reproducible in closed loop); a LOW-coherence mode",
          "  carrying energy = exogenous forcing the deterministic surrogate",
          "  cannot generate. The latter is what stochastic closure must supply.",
          "  Energy fraction <3% => coherence is computed on a near-empty band;",
          "  do not interpret it (this is what happened at Ur=4.63)."]
    return "\n".join(L)


def make_plot(spectra, out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, P_true, P_tf = spectra["f"], spectra["P_true"], spectra["P_tf"]
    fc, coh = spectra["fc"], spectra["coh"]
    fn, fu, fl, rel = (spectra[k] for k in ("fn", "f_upper", "f_lower", "rel"))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].semilogy(f, P_true + EPS, "k", lw=1.4, label="CFD CL (true)")
    ax[0].semilogy(f, P_tf + EPS, "tab:blue", lw=1.0, label="GRU CL (TF)")
    for c, col, name in [(fn, "green", "$f_n$=0.32"),
                         (fu, "purple", "upper shed"),
                         (fl, "orange", "lower shed")]:
        ax[0].axvspan(c * (1 - rel), c * (1 + rel), color=col, alpha=0.12)
        ax[0].axvline(c, color=col, ls="--", lw=1, label=name)
    ax[0].set_ylabel("PSD of $C_l$"); ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title(f"Bimodal-shedding coherence — {out['label']} (U={out['U']} m/s)")
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].plot(fc, coh, "k", lw=1.3, label="coh(CL, vel)")
    for c, col in [(fn, "green"), (fu, "purple"), (fl, "orange")]:
        ax[1].axvspan(c * (1 - rel), c * (1 + rel), color=col, alpha=0.12)
        ax[1].axvline(c, color=col, ls="--", lw=1)
    ax[1].axhline(0.7, color="gray", ls=":", lw=1)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel(r"coherence $\gamma^2$")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_xlim(0, max(0.5, 1.5 * fn)); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--U", type=float, required=True, help="wind speed [m/s]")
    ap.add_argument("--f_lower", type=float, default=None,
                    help="measured lower-mode freq [Hz]; else 0.5*f_upper")
    ap.add_argument("--st_upper", type=float, default=ST_UPPER)
    ap.add_argument("--skip_s", type=float, default=5.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(f"results/{a.npz}")
    label = a.label or a.npz.replace(".npz", "")
    out, spectra = analyze_modes(d["cl_true"], d["cl_tf"], d["vel"],
                                 float(d["dt"]), a.U, skip_s=a.skip_s,
                                 st_upper=a.st_upper, f_lower_hz=a.f_lower,
                                 label=label)
    print(json.dumps(out, indent=2, default=float))
    print()
    print(report(out))
    make_plot(spectra, out, a.out or a.npz.replace(".npz", "") + "_modes.png")


if __name__ == "__main__":
    main()