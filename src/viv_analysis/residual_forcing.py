"""
Generate a stochastic forcing vector e[0:n_steps] from a measured TF residual.

Modes
-----
surrogate : phase-randomized surrogate — same power spectrum, independent phase realization
replay    : direct replay of the residual (tiled to fill n_steps)
white     : white noise matched to residual std (spectral-shape null control)
none      : zeros (handled at call site; this function is not called for "none")
"""

import numpy as np


def make_forcing(resid, n_steps, mode="surrogate", scale=1.0, seed=0):
    """
    Parameters
    ----------
    resid   : 1-D array — cl_true - cl_tf for one CFD case
    n_steps : length of the output vector
    mode    : "surrogate" | "replay" | "white"
    scale   : scalar multiplier applied to the whole vector
    seed    : RNG seed (for surrogate and white modes)

    Returns
    -------
    e : float32 array of shape (n_steps,)
    """
    resid = np.asarray(resid, dtype=np.float64)
    n_resid = len(resid)

    rng = np.random.default_rng(seed)

    if mode == "surrogate":
        # Phase-randomised surrogate: preserve amplitude spectrum, scramble phases.
        # Each tile gets an independent phase draw so tiling seams are invisible.
        tiles_needed = int(np.ceil(n_steps / n_resid))
        pieces = []
        for _ in range(tiles_needed):
            fft = np.fft.rfft(resid)
            n_rfft = len(fft)
            phases = rng.uniform(0.0, 2.0 * np.pi, n_rfft)
            phases[0] = 0.0  # preserve DC
            if n_resid % 2 == 0:
                phases[-1] = 0.0  # preserve real-valued Nyquist bin
            fft_surr = np.abs(fft) * np.exp(1j * phases)
            pieces.append(np.fft.irfft(fft_surr, n=n_resid))
        e = np.concatenate(pieces)[:n_steps]

    elif mode == "replay":
        tiles_needed = int(np.ceil(n_steps / n_resid))
        e = np.tile(resid, tiles_needed)[:n_steps]

    elif mode == "white":
        # Same variance as the residual so energy comparison is fair.
        e = rng.normal(0.0, float(resid.std()), n_steps)

    else:
        raise ValueError(f"make_forcing: unknown mode '{mode}'. "
                         "Use 'surrogate', 'replay', or 'white'.")

    return (scale * e).astype(np.float32)
