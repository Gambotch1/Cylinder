#!/usr/bin/env python3
"""
Post-hoc diagnostics for the v3_coherent closure, read entirely from data
already saved by coupled_inference.py (t, h, h_dot, cl, cl_det, cl_vdp,
D, Ur, a_ref_used_m, mu_value) -- no simulation code is touched or rerun.

Context: v3_coherent's mu and a_ref are measured from each TARGET case's
own CFD growth transient / steady-state amplitude (self_excitation.py's
measure_mu_from_cfd / measure_cfd_amplitude), not predicted or generalized
across cases. It is therefore relabeled here as an exploratory,
case-calibrated amplitude regulator rather than an independently
predictive closure -- see CLOSURE_LABELS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from viv_analysis.closed_loop_metrics import cycles_from_displacement

# Human-facing relabeling for logs/tables/manifests. Internal CLI/API
# identifiers (--forcing_mode v1_additive/v2_multiplicative/v3_coherent,
# filenames already on disk, etc.) are NOT renamed -- this is a
# presentation-layer label only, per the demotion request.
CLOSURE_LABELS = {
    "v1_additive": "Raw causal GRU (primary baseline)",
    "v2_multiplicative": "Exploratory amplitude-gained residual forcing",
    "v3_coherent": "Exploratory CFD-calibrated amplitude regulator",
}


def compute_v3_time_resolved(npz_path: str, fn: float, rho: float, B: float,
                             c: float) -> dict:
    """Time-resolved A_env, A_env/A_ref, cl_gru/cl_vdp/cl_total, and
    instantaneous + cumulative work for the GRU, v3, and structural-
    dissipation contributions, for one v3_coherent coupled_*.npz.

    Purely derived from already-saved arrays (cl_det, cl_vdp, h, h_dot) --
    does not change or reevaluate the closure itself.
    """
    d = np.load(npz_path, allow_pickle=True)
    for required in ("t", "h", "h_dot", "cl_det", "cl_vdp", "cl", "Ur", "D",
                     "a_ref_used_m", "mu_value"):
        if required not in d.files:
            raise ValueError(f"{npz_path} is missing '{required}' -- not a "
                            f"v3_coherent run, or predates cl_det/cl_vdp logging.")

    t, h, h_dot = d["t"], d["h"], d["h_dot"]
    cl_det, cl_vdp, cl_total = d["cl_det"], d["cl_vdp"], d["cl"]
    Ur, D = float(d["Ur"]), float(d["D"])
    a_ref = float(d["a_ref_used_m"])
    mu = float(d["mu_value"])
    U = Ur * fn * D
    omega_n = 2.0 * np.pi * fn

    a_env = np.sqrt(h**2 + (h_dot / omega_n) ** 2)
    a_env_over_aref = a_env / a_ref if a_ref else np.full_like(a_env, np.nan)

    q = 0.5 * rho * U**2 * B
    power_gru = (q * cl_det) * h_dot
    power_vdp = (q * cl_vdp) * h_dot
    power_diss = c * h_dot**2

    def cumtrapz(t, y):
        out = np.zeros_like(y)
        if len(t) > 1:
            out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
        return out

    return dict(
        t=t, D=D, Ur=Ur, a_ref=a_ref, mu_vdp=mu,
        a_env=a_env, a_env_over_aref=a_env_over_aref,
        cl_gru=cl_det, cl_vdp=cl_vdp, cl_total=cl_total,
        power_gru=power_gru, power_vdp=power_vdp, power_diss=power_diss,
        cum_work_gru=cumtrapz(t, power_gru),
        cum_work_vdp=cumtrapz(t, power_vdp),
        cum_work_diss=cumtrapz(t, power_diss),
    )


def compute_cycle_integrated_work(npz_path: str, fn: float, rho: float, B: float,
                                  c: float) -> pd.DataFrame:
    """Per-cycle (zero-crossing-of-h_dot) integrated work from the GRU, v3,
    and structural-dissipation contributions, plus mean/max A_env/A_ref for
    that cycle -- the basis for checking whether negative v3 cycle-work
    lines up with A_env > A_ref (the closure's own designed stabilizing
    regime) or is unrelated to amplitude (a sign inconsistency /
    implementation detail elsewhere)."""
    sig = compute_v3_time_resolved(npz_path, fn, rho, B, c)
    t, h = sig["t"], np.load(npz_path, allow_pickle=True)["h"]
    cycles = cycles_from_displacement(t, h, D=sig["D"])

    rows = []
    for i0, i1 in cycles:
        t_c = t[i0:i1 + 1]
        rows.append(dict(
            t_start=float(t_c[0]), t_end=float(t_c[-1]), period_s=float(t_c[-1] - t_c[0]),
            work_gru=float(np.trapezoid(sig["power_gru"][i0:i1 + 1], t_c)),
            work_vdp=float(np.trapezoid(sig["power_vdp"][i0:i1 + 1], t_c)),
            work_diss=float(np.trapezoid(sig["power_diss"][i0:i1 + 1], t_c)),
            a_env_over_aref_mean=float(np.mean(sig["a_env_over_aref"][i0:i1 + 1])),
            a_env_over_aref_max=float(np.max(sig["a_env_over_aref"][i0:i1 + 1])),
        ))
    return pd.DataFrame(rows)


def diagnose_negative_v3_work(npz_path: str, fn: float, rho: float, B: float,
                              c: float) -> dict:
    """Splits cycles by sign of work_vdp and reports whether the
    A_env>A_ref regime (the closure's designed stabilizing condition;
    cl_vdp's (1-(A_env/A_ref)^2) factor flips negative there) explains
    negative-work cycles, vs. a residual not explained by amplitude alone."""
    cyc = compute_cycle_integrated_work(npz_path, fn, rho, B, c)
    if cyc.empty:
        return {"n_cycles": 0}
    neg = cyc[cyc["work_vdp"] < 0]
    pos = cyc[cyc["work_vdp"] >= 0]
    return {
        "n_cycles": len(cyc),
        "n_negative_work_cycles": len(neg),
        "n_positive_work_cycles": len(pos),
        "negative_cycles_with_a_env_gt_a_ref": int((neg["a_env_over_aref_mean"] > 1.0).sum()),
        "negative_cycles_with_a_env_le_a_ref": int((neg["a_env_over_aref_mean"] <= 1.0).sum()),
        "mean_a_env_over_aref_negative_cycles": float(neg["a_env_over_aref_mean"].mean()) if len(neg) else float("nan"),
        "mean_a_env_over_aref_positive_cycles": float(pos["a_env_over_aref_mean"].mean()) if len(pos) else float("nan"),
        "cycle_table": cyc,
    }
