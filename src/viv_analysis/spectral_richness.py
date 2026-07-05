#!/usr/bin/env python3
"""
CL spectral-richness comparison: isolate the GRU's contribution in v3.
=====================================================================
Compares the LIFT (CL) spectrum across CFD / v3-on / VdP-null(s). Displacement
is structure-filtered (clean tone in every mode) and does NOT discriminate;
the aerodynamic content lives in CL.

Claim it tests: a pure VdP closure produces a sterile tone (+ odd harmonics);
the GRU corrects the CL waveform -- harmonic structure and phase -- toward CFD.
The dominant FREQUENCY is structural (~0.32 Hz) in all modes, so richness, not
peak location, is the discriminator.

Metrics per signal:
  f_dom          dominant frequency [Hz]
  flatness       spectral flatness (geo/arith mean of PSD), 0=pure tone .. 1=white
  oob_frac       fraction of energy OUTSIDE the fundamental peak band (richness)
  h2, h3         energy at 2f, 3f relative to fundamental (harmonic signature)
"""
import argparse, json
import numpy as np
from scipy.signal import welch

def richness(cl, dt, fn, fmax=None):
    cl = np.asarray(cl, float); cl = cl - cl.mean()
    fs = 1.0/dt; fmax = fmax or min(5.0, 0.45*fs)
    nper = min(len(cl)//8, 1<<14)
    f, P = welch(cl, fs=fs, nperseg=max(nper,256), detrend="constant")
    band = f <= fmax; f, P = f[band], P[band]
    m = (f>0.05*fn) & (f<3.0*fn)
    f_dom = float(f[m][np.argmax(P[m])]) if m.any() else float("nan")
    # fundamental band = +-15% around f_dom
    fb = (f>0.85*f_dom) & (f<1.15*f_dom)
    tot = np.trapezoid(P, f)+1e-30
    oob = 1.0 - np.trapezoid(P[fb], f[fb])/tot
    Pp = np.clip(P, 1e-30, None)
    flat = float(np.exp(np.mean(np.log(Pp)))/ (np.mean(Pp)+1e-30))
    def band_energy(fc):
        b=(f>0.9*fc)&(f<1.1*fc); return float(np.trapezoid(P[b], f[b])) if b.any() else 0.0
    e1=band_energy(f_dom)+1e-30
    return dict(f=f, P=P, f_dom=f_dom, flatness=flat, oob_frac=float(oob),
                h2=band_energy(2*f_dom)/e1, h3=band_energy(3*f_dom)/e1)

def load_cl(path):
    d = np.load(path)
    if "cl_true" in d:  return np.asarray(d["cl_true"], float), float(d["dt"])   # tf_residual = CFD
    return np.asarray(d["cl"], float), float(np.median(np.diff(d["t"])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfd", required=True, help="tf_residual npz (has cl_true) OR coupled npz")
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.npz for each mode")
    ap.add_argument("--fn", type=float, default=0.32)
    ap.add_argument("--out", default="cl_richness.png")
    a = ap.parse_args()

    cl_cfd, dt = load_cl(a.cfd)
    series = {"CFD": richness(cl_cfd, dt, a.fn)}
    for spec in a.runs:
        lbl, path = spec.split("=", 1)
        cl, dtr = load_cl(path)
        series[lbl] = richness(cl, dtr, a.fn)

    print(f"{'signal':<18}{'f_dom':>7}{'flatness':>10}{'oob_frac':>10}{'h2':>7}{'h3':>7}")
    print("-"*60)
    for lbl, r in series.items():
        print(f"{lbl:<18}{r['f_dom']:>7.3f}{r['flatness']:>10.4f}"
              f"{r['oob_frac']:>10.3f}{r['h2']:>7.3f}{r['h3']:>7.3f}")
    # closeness to CFD richness (oob_frac) — the GRU should move v3-on toward CFD
    cfd_oob = series["CFD"]["oob_frac"]
    print("\nrichness gap to CFD (|oob_frac - CFD|, smaller = more CFD-like):")
    for lbl, r in series.items():
        if lbl!="CFD":
            print(f"  {lbl:<16}{abs(r['oob_frac']-cfd_oob):.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl, r in series.items():
        ax.semilogy(r["f"], r["P"]+1e-30, lw=1.4 if lbl=="CFD" else 1.0,
                    color="k" if lbl=="CFD" else None, label=lbl)
    ax.axvline(a.fn, color="green", ls="--", lw=1)
    ax.set_xlim(0, 5*a.fn); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD of $C_L$")
    ax.set_title("Lift spectrum: does the GRU restore CFD's aerodynamic richness?")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print(f"\nsaved -> {a.out}")

# ---- self-test: sterile VdP-like vs rich CFD-like ----
def _selftest():
    dt=0.002; fn=0.32; w=2*np.pi*fn; n=120000; t=np.arange(n)*dt
    rng=np.random.default_rng(0)
    vdp = np.sin(w*t) + 0.15*np.sin(3*w*t)                          # tone + odd harmonic
    from scipy.signal import butter, sosfiltfilt
    broad = sosfiltfilt(butter(2,1.0,fs=1/dt,output='sos'), rng.standard_normal(n))
    cfd = np.sin(w*t) + 0.25*np.sin(2*w*t+0.7) + 0.6*broad/broad.std()  # rich
    rv=richness(vdp,dt,fn); rc=richness(cfd,dt,fn)
    print("self-test  (VdP sterile vs CFD-like rich)")
    print(f"  VdP  flatness={rv['flatness']:.4f}  oob={rv['oob_frac']:.3f}")
    print(f"  CFD  flatness={rc['flatness']:.4f}  oob={rc['oob_frac']:.3f}")
    assert rc['oob_frac']>rv['oob_frac'] and rc['flatness']>rv['flatness']
    print("  PASS: rich signal has higher oob_frac + flatness than sterile VdP")

if __name__ == "__main__":
    import sys
    if len(sys.argv)==1: _selftest()
    else: main()