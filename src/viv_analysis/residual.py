#!/usr/bin/env python3
"""
Residual-forcing generator for the stochastic-closure probe.
===========================================================

Purpose
-------
We measured (TF-residual diagnostic) that ~50% of the lock-in-band lift is
incoherent with the deck's motion -> exogenous vortex shedding that a
deterministic kinematics->lift surrogate cannot generate. This module
synthesizes that exogenous component as an additive forcing term e(t) in CL
units, to inject into the coupled loop:

    CL_total(t) = CL_GRU(kinematics)  +  e(t)
    F_aero(t)   = 0.5 * rho * U^2 * B * CL_total(t)

e(t) is calibrated to the MEASURED residual (CL_true - CL_TF), so its magnitude
is not a free knob -- it is a property of the CFD, injected at scale 1.0.

Three modes (the last two are CONTROLS, not the model):
  'surrogate' : phase-randomized surrogate of the residual. Same PSD, same
                variance, INDEPENDENT realization (decorrelated from the true
                forcing). This is the probe.
  'replay'    : the actual residual time series. Tautology / upper bound -- if
                surrogate sustains but replay does too at the same amplitude,
                fine; if ONLY replay works, the forcing is more structured than
                its spectrum (informative, but not a usable model).
  'white'     : Gaussian white noise at the same variance. SPECTRUM-AGNOSTIC
                control. If white reproduces the response as well as surrogate,
                then spectral shape does not matter and the result is just
                resonance-band energy -> NOT a prediction.

Why phase randomization: it produces a new realization with the residual's
exact power spectrum (hence autocorrelation and variance) but new random
phases, so it is statistically matched yet causally independent of the true
forcing. That independence is what makes "matched forcing -> matched response"
a meaningful claim rather than a replay.

numpy only.
"""

import numpy as np


def _phase_randomized(x, rng):
    """One phase-randomized surrogate block, same length and PSD as x."""
    x = np.asarray(x, float)
    x = x - x.mean()
    N = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=mag.shape)
    ph[0] = 0.0                      # DC real
    if N % 2 == 0:
        ph[-1] = 0.0                 # Nyquist real for even N
    Xr = mag * np.exp(1j * ph)
    s = np.fft.irfft(Xr, n=N)
    return s


def make_forcing(resid, n_out, mode="surrogate", scale=1.0, seed=0,
                 strip_dc=True):
    """
    Build an additive CL-forcing vector e[0:n_out].

    resid : measured residual (CL_true - CL_TF), 1-D.
    n_out : number of coupled steps needed.
    mode  : 'surrogate' | 'replay' | 'white'.
    scale : multiplies the forcing std. Keep 1.0 for the principled probe;
            sweeping it is the 'am I tuning to fit?' test (you should NOT
            need to retune per case if the mechanism is real).
    """
    resid = np.asarray(resid, float).ravel()
    if strip_dc:
        resid = resid - resid.mean()
    target_std = resid.std()
    rng = np.random.default_rng(seed)

    if mode == "replay":
        # tile the actual residual to length n_out
        reps = int(np.ceil(n_out / len(resid)))
        e = np.tile(resid, reps)[:n_out]

    elif mode == "white":
        e = rng.standard_normal(n_out)

    elif mode == "surrogate":
        blocks = []
        got = 0
        while got < n_out:
            b = _phase_randomized(resid, rng)
            blocks.append(b)
            got += len(b)
        e = np.concatenate(blocks)[:n_out]

    else:
        raise ValueError(f"unknown mode {mode!r}")

    # set variance exactly to (scale * measured std); DC is handled by the
    # structural equilibrium, not the forcing, so e is zero-mean here.
    e = e - e.mean()
    e = e * (scale * target_std / (e.std() + 1e-30))
    return e


# --------------------------------------------------------------------------- #
# self-test: verify the surrogate preserves PSD + variance and is reproducible
# --------------------------------------------------------------------------- #
def _selftest():
    from scipy.signal import welch
    rng = np.random.default_rng(1)
    fs = 500.0
    n = 120000
    # synthetic "residual": broadband floor + a peak near 0.32 Hz
    t = np.arange(n) / fs
    white = rng.standard_normal(n)
    # cheap colored signal: low-pass white + a narrowband tone with noise
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, 2.0, fs=fs, output="sos")
    colored = sosfiltfilt(sos, white)
    tone = 0.4 * np.sin(2 * np.pi * 0.32 * t + rng.uniform(0, 6.28)) \
        * (1 + 0.3 * rng.standard_normal(n))
    resid = colored + tone
    resid -= resid.mean()

    e1 = make_forcing(resid, n, mode="surrogate", seed=7)
    e2 = make_forcing(resid, n, mode="surrogate", seed=7)
    e3 = make_forcing(resid, n, mode="surrogate", seed=8)

    f, Pr = welch(resid, fs=fs, nperseg=8192)
    _, Pe = welch(e1, fs=fs, nperseg=8192)

    band = (f > 0.05) & (f < 10)
    psd_corr = np.corrcoef(np.log(Pr[band] + 1e-30),
                           np.log(Pe[band] + 1e-30))[0, 1]
    print("self-test")
    print(f"  variance  resid={resid.var():.5f}  surrogate={e1.var():.5f}  "
          f"ratio={e1.var()/resid.var():.3f}")
    print(f"  log-PSD correlation (resid vs surrogate), 0.05-10 Hz = {psd_corr:.4f}")
    print(f"  reproducible under same seed : {np.allclose(e1, e2)}")
    print(f"  different under diff seed     : {not np.allclose(e1, e3)}")
    # power at the resonance band should match (this is what drives amplitude)
    rb = (f > 0.16) & (f < 0.48)
    print(f"  resonance-band power  resid={np.trapezoid(Pr[rb], f[rb]):.5e}  "
          f"surrogate={np.trapezoid(Pe[rb], f[rb]):.5e}")
    assert psd_corr > 0.9, "PSD not preserved"
    assert abs(e1.var() / resid.var() - 1) < 0.05, "variance not preserved"
    assert np.allclose(e1, e2) and not np.allclose(e1, e3)
    print("  PASS")


if __name__ == "__main__":
    _selftest()