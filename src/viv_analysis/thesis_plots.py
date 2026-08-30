#!/usr/bin/env python3
"""
Thesis-quality figure functions -- deliberately separate from the training-
diagnostic plotting in train_gru.py (plot_tf_result) and coupled_inference.py
so that changing thesis formatting can never change what a training run
itself prints/saves.

Shared conventions across every function here:
  - CFD_STYLE / MODEL_STYLE / ERROR_STYLE (plot_style.py) for all traces --
    never an ad hoc color= call, so "CFD" and "GRU prediction" always look
    the same across every figure in the thesis.
  - No text baked into the figure beyond axis labels/legend (no title, no
    R^2 annotation) -- metrics belong in the caption, generated alongside
    each figure as a plain-text LaTeX-ready snippet (see build_caption).
  - Both PDF (primary thesis format) and PNG (quick preview) are written.
  - A fixed, documented interval choice (not "wherever the error is
    largest") for any zoomed panel, so the figure can't be read as
    selectively cropped.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from viv_analysis.plot_style import (
    CFD_COLOR, CFD_STYLE, ERROR_STYLE, GRID_COLOR, MODEL_COLOR, MODEL_STYLE,
    ORANGE_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN, apply_thesis_style,
)
from viv_analysis.utils import segment_by_time_gaps

HANDOFF_COLOR = "#777777"


def break_at_gaps(t: np.ndarray, *ys: np.ndarray):
    """Insert a NaN row at every internal time-axis discontinuity
    (segment_by_time_gaps) so matplotlib breaks the line there instead of
    drawing a straight interpolation across a real gap (e.g. a
    discontinuous CFD restart -- see bridge Ur=6.9491's ~77.5s report-file
    gap) -- a plotted straight line across such a gap has no physical
    meaning and reads as a real (and badly wrong) prediction/reference if
    left in. No-op (returns the inputs unchanged) when there is no gap."""
    t = np.asarray(t)
    segs = segment_by_time_gaps(t)
    if len(segs) <= 1:
        return (t, *ys)
    break_idxs = [seg_end for seg_start, seg_end in segs[:-1]]  # insert before these
    t_out = np.insert(t.astype(float), break_idxs, np.nan)
    ys_out = tuple(np.insert(np.asarray(y).astype(float), break_idxs, np.nan) for y in ys)
    return (t_out, *ys_out)


def ur_tag(ur: float) -> str:
    """Filesystem-safe Ur identifier for output filenames, e.g. 5.5 ->
    'Ur5p50' (fixed 2-decimal precision so 5.5 and 5.50 never collide, and
    no literal '.' in a filename)."""
    return f"Ur{ur:.2f}".replace(".", "p")


def downsample_for_display(x: np.ndarray, y: np.ndarray, max_points: int = 8000):
    """Min/max-binning decimation for DISPLAY ONLY -- every metric (R^2,
    amplitude, ...) must always be computed on the full-resolution arrays
    BEFORE calling this. Unlike naive uniform-stride slicing, min/max
    binning cannot alias away an oscillation peak: each bin contributes
    both its local min and max (in chronological order), so peak-to-peak
    envelopes are preserved even though the point count drops by ~10-20x
    for a typical smooth VIV trajectory."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    n_bins = max(1, max_points // 2)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        seg_x, seg_y = x[lo:hi], y[lo:hi]
        i_min, i_max = int(np.argmin(seg_y)), int(np.argmax(seg_y))
        order = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        xs.extend(seg_x[list(order)]); ys.extend(seg_y[list(order)])
    return np.array(xs), np.array(ys)


def build_caption(condition_label: str, r2: float, zoom_duration: float,
                  quantity: str = "lift-coefficient") -> str:
    """LaTeX-ready caption text (not rendered in the figure -- meant to be
    pasted into the thesis alongside the PDF)."""
    return (
        f"Open-loop {quantity} prediction for {condition_label}. "
        f"The upper panel shows the complete analysed interval, while the "
        f"lower panel enlarges the first {zoom_duration:g} seconds. "
        f"The model achieved $R^2={r2:.4f}$."
    )


def plot_tf_result_thesis(
    cl_pred: np.ndarray,
    cl_true: np.ndarray,
    times: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    physical_params_note: str | None = None,
    zoom_duration: float = 20.0,
    include_residual: bool = False,
    fn: float | None = None,
    font_scale: float = 1.0,
    linewidth_scale: float = 1.0,
) -> dict:
    """Two-panel (optionally three) open-loop teacher-forcing figure:
    upper = full analysed interval, lower = a FIXED zoom of the first
    `zoom_duration` seconds after the interval start (never chosen to
    showcase the largest error -- see module docstring). No in-figure
    title or R^2 annotation; metrics go into the returned/saved caption.

    The upper (full-interval) panel uses SOLID lines for both traces, not
    MODEL_STYLE's dashed pattern -- these test cases run for hundreds of
    oscillation cycles, and a dashed line at that cycle density produces a
    moire/beat artifact against its own dash period (verified directly,
    same finding as plot_coupled_thesis; not a downsampling issue). The
    lower (20s) zoom panel keeps MODEL_STYLE's actual dash pattern, where
    only a handful of cycles are shown and the dash renders cleanly --
    exactly where the CFD-vs-model distinction needs to be visible.

    case_label: filesystem-safe identifier used for the output filename
        (e.g. "Ur5.5" or "bridge_Ur6.7385").
    condition_label: human-readable condition description for the caption
        text (e.g. "$U_r=5.5$" for cylinder, or
        "$U=16\\,\\mathrm{m/s}$ ($U_r=6.7385$)" for bridge, per the
        bridge-chapter convention of leading with physical wind speed).
        Defaults to case_label if not given.
    physical_params_note: fixed physical parameters common to every case in
        the chapter (e.g. "$Re=200$, $m^*=10$, $\\zeta=0.01$") -- reported
        once in the caption rather than repeated in every figure's legend.
    fn: if given, the x-axis is nondimensionalized as (t-t_0)/T_n (T_n=
        1/fn) instead of raw seconds -- opt-in, same convention as
        plot_coupled_thesis/plot_open_loop_representative.
    font_scale: multiplies axes/tick/legend font sizes by this factor
        without changing figsize -- same convention and rationale as
        plot_open_loop_representative's font_scale (a figure generated at
        full \\textwidth proportions but displayed smaller, e.g. two-up at
        0.48\\textwidth in an appendix, needs deliberately oversized text
        so it reads normally once LaTeX shrinks the page). Pass
        1/display_fraction, e.g. 1/0.48.
    linewidth_scale: multiplies both CFD_STYLE's and MODEL_STYLE's
        linewidth by this factor (both traces, so their relative
        thickness is preserved) -- for a case like the cylinder open-loop
        figures where the CFD/GRU traces overlap almost exactly and a
        thinner line reads more cleanly.
    """
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    cfd_kwargs = {**CFD_STYLE, "linestyle": "-",
                  "linewidth": CFD_STYLE["linewidth"] * linewidth_scale}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-",
                    "linewidth": MODEL_STYLE["linewidth"] * linewidth_scale}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(times) - times[0]
    cl_pred = np.asarray(cl_pred)
    cl_true = np.asarray(cl_true)
    residual = cl_pred - cl_true

    if fn is not None:
        Tn = 1.0 / fn
        t_disp, xlabel = t / Tn, r"$(t-t_0)/T_n$"
    else:
        t_disp, xlabel = t, r"Time after evaluation start, $t-t_0$ [s]"

    n_rows = 3 if include_residual else 2
    height_ratios = [1.6, 1.0, 0.7] if include_residual else [1.6, 1.0]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(TEXT_WIDTH_IN, 4.6 if not include_residual else 5.6),
        gridspec_kw={"height_ratios": height_ratios}, constrained_layout=True,
    )
    ax_full, ax_zoom = axes[0], axes[1]

    # break_at_gaps only affects the FULL panel's drawn line (a straight
    # interpolation across a real CFD-restart gap has no physical meaning);
    # the underlying t/cl_pred/cl_true arrays used for the zoom/residual/R^2
    # below are left untouched.
    t_full, cl_true_full, cl_pred_full = break_at_gaps(t_disp, cl_true, cl_pred)
    ax_full.plot(t_full, cl_true_full, **cfd_kwargs)
    ax_full.plot(t_full, cl_pred_full, **model_kwargs)
    ax_full.set_ylabel(r"$C_L$")
    ax_full.grid(True, which="major")
    # Anchored ABOVE the axes (bbox_to_anchor y=1.0 with loc="lower left"
    # corner of the legend box at that point), not "upper right" inside the
    # data area -- at font_scale>1 (e.g. the 0.48\textwidth appendix
    # figures) an in-axes legend box scales up right along with the text
    # and can cover a large fraction of the panel's own data (verified
    # directly: ~1/4 of the panel at font_scale=1/0.48).
    ax_full.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    zoom_mask = t <= zoom_duration
    ax_zoom.plot(t_disp[zoom_mask], cl_true[zoom_mask], **{**cfd_kwargs, "label": "_nolegend_"})
    ax_zoom.plot(t_disp[zoom_mask], cl_pred[zoom_mask], **{**model_kwargs, "label": "_nolegend_"})
    ax_zoom.set_ylabel(r"$C_L$")
    ax_zoom.grid(True, which="major")
    if not include_residual:
        ax_zoom.set_xlabel(xlabel)

    if include_residual:
        ax_res = axes[2]
        ax_res.plot(t_disp[zoom_mask], residual[zoom_mask], **ERROR_STYLE)
        ax_res.axhline(0.0, color=CFD_STYLE["color"], linewidth=0.6)
        ax_res.set_ylabel(r"Residual, $e_{C_L}=\widehat{C}_L-C_L^{\mathrm{CFD}}$")
        ax_res.set_xlabel(xlabel)
        ax_res.grid(True, which="major")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"open_loop_{case_label}.{suffix}")
    plt.close(fig)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(cl_true, cl_pred))
    caption = build_caption(condition_label or case_label, r2, zoom_duration)
    if physical_params_note:
        caption += f" Physical parameters: {physical_params_note}."
    caption_path = output_dir / f"open_loop_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(r2=r2, caption=caption,
               pdf_path=output_dir / f"open_loop_{case_label}.pdf",
               png_path=output_dir / f"open_loop_{case_label}.png",
               caption_path=caption_path)


def plot_coupled_thesis(
    t: np.ndarray, h: np.ndarray, CL: np.ndarray, D: float,
    CFD_t: np.ndarray, CFD_h: np.ndarray, CFD_cl: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    t_handoff: float | None = None,
    h_mode: str = "raw",
    dataset_note: str | None = None,
    max_display_points: int = 8000,
    fn: float | None = None,
    U: float | None = None,
    font_scale: float = 1.0,
) -> dict:
    """Closed-loop (coupled GRU-structural) time-series figure: normalized
    displacement h/D and lift coefficient C_L, shared time axis, one shared
    legend (upper panel only -- the lower panel's traces are the same two
    series and would only duplicate it), handoff marked with a thin grey
    dotted line on both panels.

    h_mode: "raw" plots h/D and CFD_h/D directly. "mean_removed" instead
        plots (h-mean(h))/D using EACH trace's own post-handoff mean (CFD
        and GRU means computed independently) -- for the bridge chapter,
        where a failed closed-loop trajectory can settle to an INCORRECT
        STATIC deflection rather than a small residual LCO; conflating that
        static offset into the same "amplitude" axis as a genuinely
        oscillating case would misrepresent it. Only the mean used for
        display is affected -- metrics (e.g. amplitude) must be computed
        upstream on the raw arrays, not recovered from this plot.

    Both raw trajectories are decimated for DISPLAY ONLY via
    downsample_for_display (see its docstring) -- these traces are often
    10^5-10^6 samples, and writing all of them as PDF vector paths is slow
    and produces very large files without adding visible information at
    thesis page size.

    fn: if given (and U is not), the x-axis is nondimensionalized as
        (t-t_handoff)/T_n (T_n=1/fn, handoff at 0) instead of raw seconds --
        opt-in so the cylinder chapter's existing figures (which don't pass
        fn) render exactly as before.
    U: if given, the x-axis is nondimensionalized as the convective/reduced
        time t*=t*U/D instead of raw seconds (no handoff shift -- t* runs
        from the CFD case's own t=0, matching the t*=tU/D convention
        directly, not a handoff-relative one). Takes precedence over fn
        when both are given.
    font_scale: multiplies axes/tick/legend font sizes by this factor
        without changing figsize -- same convention as plot_open_loop_
        representative/plot_tf_result_thesis's font_scale, for appendix
        figures displayed narrower than full \\textwidth. Pass
        1/display_fraction.
    """
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t, h, CL = np.asarray(t), np.asarray(h), np.asarray(CL)
    CFD_t, CFD_h, CFD_cl = np.asarray(CFD_t), np.asarray(CFD_h), np.asarray(CFD_cl)

    if h_mode == "mean_removed":
        h_disp = h - np.mean(h)
        CFD_h_disp = CFD_h - np.mean(CFD_h)
        h_ylabel = r"$(h-\bar{h})/D$"
    elif h_mode == "raw":
        h_disp, CFD_h_disp = h, CFD_h
        h_ylabel = r"$h/D$"
    else:
        raise ValueError(f"h_mode must be 'raw' or 'mean_removed', got {h_mode!r}")

    if U is not None:
        t_plot, CFD_t_plot = t * U / D, CFD_t * U / D
        t_handoff_plot = t_handoff * U / D if t_handoff is not None else None
        time_xlabel = r"$t^*=tU/D$"
    elif fn is not None:
        Tn = 1.0 / fn
        t0 = t_handoff if t_handoff is not None else 0.0
        t_plot, CFD_t_plot = (t - t0) / Tn, (CFD_t - t0) / Tn
        t_handoff_plot = 0.0 if t_handoff is not None else None
        time_xlabel = r"$(t-t_{\mathrm{h}})/T_n$"
    else:
        t_plot, CFD_t_plot = t, CFD_t
        t_handoff_plot = t_handoff
        time_xlabel = r"Time, $t$ [s]"

    t_d, h_d = downsample_for_display(t_plot, h_disp / D, max_display_points)
    CFD_t_d, CFD_h_d = downsample_for_display(CFD_t_plot, CFD_h_disp / D, max_display_points)
    t_d2, CL_d = downsample_for_display(t_plot, CL, max_display_points)
    CFD_t_d2, CFD_cl_d = downsample_for_display(CFD_t_plot, CFD_cl, max_display_points)

    # linestyle="-" (not MODEL_STYLE's "--") for BOTH traces here, by design:
    # a sustained VIV response panel spans hundreds of oscillation cycles
    # compressed into one page width. Verified directly (full-resolution AND
    # decimated) that a dashed line in that regime produces a moire/beat
    # artifact against its own dash period, independent of the display
    # decimation above -- this is not a downsampling bug, it's dashing
    # being the wrong choice at this cycle density. CFD-vs-model is still
    # unambiguous here from color + the model line's higher zorder drawing
    # over CFD. MODEL_STYLE/CFD_STYLE's dashed/solid distinction is kept
    # as-is for lower-cycle-density figures (open-loop zoom, amplitude
    # response) where it renders cleanly.
    cfd_kwargs = {**CFD_STYLE, "linestyle": "-", "linewidth": 0.6}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-", "linewidth": 0.6, "label": "GRU-coupled response"}

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.5), sharex=True,
                             constrained_layout=True)

    axes[0].plot(CFD_t_d, CFD_h_d, **cfd_kwargs)
    axes[0].plot(t_d, h_d, **model_kwargs)
    axes[0].set_ylabel(h_ylabel)
    # Anchored ABOVE the axes, not inside as an opaque box -- the traces
    # fill nearly the entire panel height at a sustained LCO, so ANY
    # in-axes legend (opaque or not) sits on top of dense data; this also
    # avoids the font_scale-at-reduced-display-width issue where a larger-
    # text in-axes legend box covers a growing fraction of the panel (see
    # plot_tf_result_thesis's identical fix).
    axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    axes[1].plot(CFD_t_d2, CFD_cl_d, **{**cfd_kwargs, "label": "_nolegend_"})
    axes[1].plot(t_d2, CL_d, **{**model_kwargs, "label": "_nolegend_"})
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(time_xlabel)

    for ax in axes:
        ax.grid(True, which="major")
        if t_handoff_plot is not None:
            ax.axvline(t_handoff_plot, color=HANDOFF_COLOR, linestyle=":", linewidth=0.9)

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"closed_loop_{case_label}.{suffix}")
    plt.close(fig)

    caption = (
        f"Closed-loop (coupled GRU-structural) response for {condition_label or case_label}"
        + (f" ({dataset_note})" if dataset_note else "") + ". "
        + ("Displacement is shown with the post-handoff mean removed. "
           if h_mode == "mean_removed" else "")
        + ("The handoff from CFD-history warm-start to fully coupled rollout is marked "
           "with a dotted vertical line. " if t_handoff is not None else "")
        + "CFD reference in black, GRU-coupled response in blue."
    )
    caption_path = output_dir / f"closed_loop_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(caption=caption,
               pdf_path=output_dir / f"closed_loop_{case_label}.pdf",
               png_path=output_dir / f"closed_loop_{case_label}.png",
               caption_path=caption_path)


AMPLITUDE_DEFINITION = (
    r"$A^*=A/D$, defined as half the peak-to-peak envelope of the steady-state "
    r"response (final 30\% of the simulated trajectory), computed identically "
    r"for CFD and every model series."
)


def plot_amplitude_response_thesis(
    df,
    model_column: str,
    output_dir: Path,
    model_label: str = "GRU-coupled response",
    cfd_column: str = "CFD",
    ur_column: str = "Ur",
    literature_columns: dict | None = None,
    dataset_note: str | None = None,
    out_name: str = "amplitude_response",
) -> dict:
    """Closed-loop amplitude-response curve (A/D vs Ur): CFD reference
    (black circles, solid), GRU model (blue open squares, dashed), and
    optional named literature/other-model series (each an explicit column,
    never auto-discovered -- a sweep CSV can carry diagnostic columns
    (frequency, RMS, stability labels, ...) that must never silently end up
    plotted as if they were amplitude series).

    literature_columns: {column_name: {"label": str, "color": "orange"|"green"}}
        -- open markers, using ORANGE_COLOR/SECONDARY_COLOR by name.
    """
    apply_thesis_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(ur_column)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.7), constrained_layout=True)

    ax.plot(df[ur_column], df[cfd_column], color=CFD_COLOR, marker="o",
           linestyle="-", linewidth=1.2, markersize=4.5, label="CFD reference")
    ax.plot(df[ur_column], df[model_column], color=MODEL_COLOR, marker="s",
           markerfacecolor="white", markeredgecolor=MODEL_COLOR,
           linestyle="--", linewidth=1.1, markersize=4.5, label=model_label)

    _lit_color_by_name = {"orange": ORANGE_COLOR, "green": SECONDARY_COLOR}
    for col, spec in (literature_columns or {}).items():
        if col not in df.columns:
            raise KeyError(f"literature_columns references missing column '{col}'")
        color = _lit_color_by_name.get(spec.get("color", "orange"), ORANGE_COLOR)
        ax.plot(df[ur_column], df[col], color=color, marker="^",
               markerfacecolor="white", markeredgecolor=color,
               linestyle="-.", linewidth=1.0, markersize=4.5, label=spec["label"])

    ax.set_xlabel(r"Reduced velocity, $U_r$")
    ax.set_ylabel(r"Normalized amplitude, $A^*=A/D$")
    ax.grid(True, which="major")
    ax.legend(loc="best")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}.{suffix}")
    plt.close(fig)

    caption = (
        f"Closed-loop amplitude response"
        + (f" ({dataset_note})" if dataset_note else "") + ". "
        + AMPLITUDE_DEFINITION
    )
    caption_path = output_dir / f"{out_name}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(caption=caption,
               pdf_path=output_dir / f"{out_name}.pdf",
               png_path=output_dir / f"{out_name}.png",
               caption_path=caption_path)


# Reference-quality-aware amplitude response (bridge). Marker SHAPE (never
# color -- CFD/model color must stay identical to every other figure in the
# thesis, per this module's header) encodes the train/val/test partition
# (circle/square/triangle), so the split is visible directly on the
# response curve, not just in a caption. Every scored point is drawn
# FILLED -- reference-quality validity (settled_lco vs everything else) is
# deliberately NOT a second marker-fill channel here; it belongs in the
# accompanying results table (case-by-case reference_status breakdown),
# which is where a reader needs it to actually interpret a specific
# number, not as an open/filled distinction on the trend curve.
# insufficient_duration/numerically_suspect still carry no meaningful
# amplitude at all and get no marker -- see REFERENCE_STATUS_EXCLUDED.
PARTITION_MARKERS = {
    "train": {"marker": "o", "markersize": 5.0},
    "val": {"marker": "s", "markersize": 4.5},
    "test": {"marker": "^", "markersize": 5.0},
}
PARTITION_LABELS = {"train": "Training case", "val": "Validation case", "test": "Test case"}
# insufficient_duration / numerically_suspect: reference_quality's own
# _EVAL_PROCEDURE_BY_STATUS assigns these NO evaluation procedure at all
# ("none -- excluded ..."), so there is no meaningful A* value to plot for
# them -- they are marked separately, never given a y-coordinate.
REFERENCE_STATUS_EXCLUDED = ("insufficient_duration", "numerically_suspect")


def plot_amplitude_response_status_aware_thesis(
    df,
    model_column: str,
    status_column: str,
    partition_column: str,
    output_dir: Path,
    model_label: str = "GRU-coupled response",
    cfd_column: str = "CFD",
    ur_column: str = "Ur",
    dataset_note: str | None = None,
    out_name: str = "amplitude_response_status_aware",
) -> dict:
    """Closed-loop amplitude-response curve (A/D vs Ur) with both the
    train/val/test partition and the CFD reference-quality classification
    made visible, instead of implying (as a plain amplitude comparison
    would) that every case is an equally valid, equally in-scope
    steady-amplitude test. See reference_quality.py's module docstring
    and classify_reference_status for the underlying reference-quality
    criteria.

    status_column: a column in df giving each row's reference_status
    (e.g. merged in from reference_quality.build_reference_status_table()).
    partition_column: a column in df giving each row's "train"/"val"/"test"
    label (e.g. merged in from the model's own metrics_gru.json case_split).
    """
    apply_thesis_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(ur_column).reset_index(drop=True)
    excluded_mask = df[status_column].isin(REFERENCE_STATUS_EXCLUDED)
    scored = df[~excluded_mask]
    is_settled = scored[status_column] == "settled_lco"  # kept for the n_settled count only

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.9), constrained_layout=True)

    # Continuous trend line through scored Ur values only (no marker here --
    # per-partition marker shape is drawn separately below, layered on top).
    ax.plot(scored[ur_column], scored[cfd_column], color=CFD_COLOR,
           linestyle="-", linewidth=1.1, zorder=2, label="_nolegend_")
    ax.plot(scored[ur_column], scored[model_column], color=MODEL_COLOR,
           linestyle="--", linewidth=1.0, zorder=2, label="_nolegend_")

    for partition, style in PARTITION_MARKERS.items():
        sub = scored[scored[partition_column] == partition]
        if sub.empty:
            continue
        ax.plot(sub[ur_column], sub[cfd_column], color=CFD_COLOR, linestyle="none",
               markerfacecolor=CFD_COLOR, markeredgecolor=CFD_COLOR, zorder=3, **style)
        # markerfacecolor="white" (not MODEL_COLOR) matches plot_amplitude_
        # response_thesis's own model-marker convention elsewhere in this
        # file -- a crisp white-faced, colored-edge marker, not a second
        # "hollow means X" signal (that channel was removed; this is just
        # this file's standing style for the model series).
        ax.plot(sub[ur_column], sub[model_column], color=MODEL_COLOR, linestyle="none",
               markerfacecolor="white", markeredgecolor=MODEL_COLOR, zorder=3, **style)

    # Excluded cases carry no amplitude value -- a short tick at the foot of
    # the axes marks their Ur position without giving them a false y-value.
    if excluded_mask.any():
        for ur in df.loc[excluded_mask, ur_column]:
            ax.axvline(ur, color=GRID_COLOR, linestyle=":", linewidth=0.9,
                      ymax=0.045, zorder=1)

    legend_handles = [
        Line2D([0], [0], color=CFD_COLOR, linestyle="-", label="CFD reference"),
        Line2D([0], [0], color=MODEL_COLOR, linestyle="--", label=model_label),
    ]
    for partition, style in PARTITION_MARKERS.items():
        if (df[partition_column] == partition).any():
            legend_handles.append(Line2D([0], [0], color="0.3", linestyle="none",
                                        markerfacecolor="0.3", markeredgecolor="0.3",
                                        label=PARTITION_LABELS[partition], **style))
    # if excluded_mask.any():
    #     legend_handles.append(Line2D(
    #         [0], [0], color=GRID_COLOR, linestyle=":",
    #         label=f"Excluded ({int(excluded_mask.sum())} case(s): insufficient "
    #               f"duration / numerically suspect)"))

    ax.set_xlabel(r"Reduced velocity, $U_r$")
    ax.set_ylabel(r"Normalized amplitude, $A^*=A/D$")
    ax.grid(True, which="major")
    ax.legend(handles=legend_handles, loc="best", fontsize=6.5)

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}.{suffix}")
    plt.close(fig)

    n_settled = int(is_settled.sum())
    n_total = len(df)
    caption = (
        f"Closed-loop amplitude response, partition-aware"
        + (f" ({dataset_note})" if dataset_note else "") + ". "
        + AMPLITUDE_DEFINITION + " "
        + f"Marker shape indicates the train/validation/test partition "
          f"(circle/square/triangle). Only {n_settled} of {n_total} cases "
          f"have a settled limit-cycle CFD reference, the sole category "
          f"for which this steady-amplitude comparison is strictly valid; "
          f"see the accompanying results table for the case-by-case "
          f"reference-quality breakdown (reference_quality.classify_"
          f"reference_status). Cases marked by a dotted vertical tick "
          f"(insufficient recording duration or a numerically suspect "
          f"reference) have no meaningful steady-state amplitude and are "
          f"omitted from both curves."
    )
    caption_path = output_dir / f"{out_name}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(caption=caption,
               pdf_path=output_dir / f"{out_name}.pdf",
               png_path=output_dir / f"{out_name}.png",
               caption_path=caption_path,
               n_settled_lco=n_settled, n_total=n_total)


def plot_learning_curve_thesis(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    output_dir: Path,
    case_label: str,
    condition_label: str | None = None,
    epoch_start: int = 1,
    out_name: str = "learning_curve",
) -> dict:
    """Training/validation loss curve with the selected (minimum-val-loss)
    checkpoint marked. Log y-axis (late-epoch behaviour is compressed on a
    linear scale), epochs numbered from epoch_start (default 1, not 0), no
    in-figure title -- model description belongs in the caption."""
    apply_thesis_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    epochs = np.arange(epoch_start, epoch_start + len(train_losses))
    best_idx = int(np.argmin(val_losses))
    best_epoch = int(epochs[best_idx])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2), constrained_layout=True)
    ax.plot(epochs, train_losses, **{**CFD_STYLE, "label": "Training"})
    ax.plot(epochs, val_losses, **{**MODEL_STYLE, "label": "Validation"})
    ax.axvline(best_epoch, color="0.45", linestyle=":", linewidth=0.9)
    ax.plot(best_epoch, val_losses[best_idx], "o", color=MODEL_COLOR, markersize=4,
           zorder=4, label="_nolegend_")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean-squared error of standardised $C_L$")
    ax.grid(True, which="major")
    ax.legend(loc="best")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}_{case_label}.{suffix}")
    plt.close(fig)

    caption = (
        f"Training and validation losses for {condition_label or case_label}. "
        f"The marker identifies the checkpoint with the minimum validation loss "
        f"(epoch {best_epoch}), which was retained for subsequent evaluation."
    )
    caption_path = output_dir / f"{out_name}_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(best_epoch=best_epoch, caption=caption,
               pdf_path=output_dir / f"{out_name}_{case_label}.pdf",
               png_path=output_dir / f"{out_name}_{case_label}.png",
               caption_path=caption_path)


def plot_open_loop_representative(
    cl_pred: np.ndarray,
    cl_true: np.ndarray,
    times: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    physical_params_note: str | None = None,
    zoom_duration: float = 20.0,
    out_name: str = "open_loop_representative",
    width_in: float | None = None,
    font_scale: float = 1.0,
    fn: float | None = None,
    linewidth_scale: float = 1.0,
) -> dict:
    """The 'preferred layout' main-text figure for ONE representative test
    case: full history, fixed 20s zoom, and the residual over that SAME
    20s window (not the full record -- the full residual history is
    scientifically useful but belongs in the appendix, not occupying a
    third of a main-text figure). Same solid-full/dashed-zoom convention
    as plot_tf_result_thesis, same reasoning (see its docstring).

    width_in: the figure's actual generated width in inches. Defaults to
    TEXT_WIDTH_IN (a full-\\textwidth figure); leave at default when the
    figure keeps its normal proportions and only the TEXT needs to read
    larger at a smaller display width (see font_scale below) -- pass an
    explicit smaller width only if the figure itself should be physically
    generated at that size (rarer; changes line/marker proportions too).

    font_scale: multiplies axes.labelsize/xtick.labelsize/ytick.labelsize/
    legend.fontsize (set by apply_thesis_style) by this factor, WITHOUT
    changing figsize -- for a figure whose canvas stays full-\\textwidth-
    proportioned but will be displayed smaller (e.g. inside a 0.49\\textwidth
    subfigure), so \\includegraphics scales the whole page down by ~0.49x
    and the deliberately oversized text lands back at a normal readable
    size. Pass 1/display_fraction (e.g. 1/0.49 for a 0.49\\textwidth
    subfigure) to land as close as possible to the original design's
    per-point legibility. Only this function's own rcParams are touched,
    and only for the duration of this call -- every other thesis_plots
    figure function calls apply_thesis_style() itself first, which resets
    these to their normal (font_scale=1) values.

    fn: if given, the x-axis is nondimensionalized as (t-t_0)/T_n (T_n=
        1/fn) instead of raw seconds -- opt-in, same convention as
        plot_coupled_thesis's fn parameter; zoom_duration stays in raw
        seconds (it only selects how much data to show, independent of
        display units).
    linewidth_scale: multiplies both CFD_STYLE's and MODEL_STYLE's
        linewidth by this factor (both traces, preserving their relative
        thickness) -- see plot_tf_result_thesis's identical parameter.
    """
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    cfd_kwargs = {**CFD_STYLE, "linestyle": "-",
                  "linewidth": CFD_STYLE["linewidth"] * linewidth_scale}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-",
                    "linewidth": MODEL_STYLE["linewidth"] * linewidth_scale}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width_in = width_in or TEXT_WIDTH_IN
    height_in = 6.2 * (width_in / TEXT_WIDTH_IN)

    t = np.asarray(times) - times[0]
    cl_pred = np.asarray(cl_pred)
    cl_true = np.asarray(cl_true)
    residual = cl_pred - cl_true
    zoom_mask = t <= zoom_duration  # selection stays in raw seconds regardless of display units

    if fn is not None:
        Tn = 1.0 / fn
        t_disp, xlabel = t / Tn, r"$(t-t_0)/T_n$"
    else:
        t_disp, xlabel = t, r"Time after evaluation start, $t-t_0$ [s]"

    fig, (ax_full, ax_zoom, ax_res) = plt.subplots(
        3, 1, figsize=(width_in, height_in),
        gridspec_kw={"height_ratios": [1.6, 1.0, 0.7]}, constrained_layout=True,
    )

    t_full, cl_true_full, cl_pred_full = break_at_gaps(t_disp, cl_true, cl_pred)
    ax_full.plot(t_full, cl_true_full, **cfd_kwargs)
    ax_full.plot(t_full, cl_pred_full, **model_kwargs)
    ax_full.set_ylabel(r"$C_L$")
    ax_full.grid(True, which="major")
    # See plot_tf_result_thesis's identical legend for why this sits above
    # the axes rather than inside the data area at "upper right".
    ax_full.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    ax_zoom.plot(t_disp[zoom_mask], cl_true[zoom_mask], **{**cfd_kwargs, "label": "_nolegend_"})
    ax_zoom.plot(t_disp[zoom_mask], cl_pred[zoom_mask], **{**model_kwargs, "label": "_nolegend_"})
    ax_zoom.set_ylabel(r"$C_L$")
    ax_zoom.grid(True, which="major")

    ax_res.plot(t_disp[zoom_mask], residual[zoom_mask], **ERROR_STYLE)
    ax_res.axhline(0.0, color=CFD_STYLE["color"], linewidth=0.6)
    # Just "$e_{C_L}$", not the full "Residual, e_CL = ... - ..." spelled-
    # out definition -- that full formula belongs in the caption/text
    # (already given there), not squeezed into a y-axis label where it
    # crowds out the plot at any display width smaller than full-textwidth.
    ax_res.set_ylabel(r"$e_{C_L}$")
    ax_res.set_xlabel(xlabel)
    ax_res.grid(True, which="major")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}_{case_label}.{suffix}")
    plt.close(fig)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(cl_true, cl_pred))
    caption = (
        f"Open-loop lift-coefficient prediction for {condition_label or case_label}, "
        f"the representative test case for the main text. The upper panel shows the "
        f"complete analysed interval, the middle panel enlarges the first "
        f"{zoom_duration:g} seconds, and the lower panel shows the residual "
        f"$e_{{C_L}}$ over that same interval. The model achieved $R^2={r2:.4f}$."
    )
    if physical_params_note:
        caption += f" Physical parameters: {physical_params_note}."
    caption_path = output_dir / f"{out_name}_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(r2=r2, caption=caption,
               pdf_path=output_dir / f"{out_name}_{case_label}.pdf",
               png_path=output_dir / f"{out_name}_{case_label}.png",
               caption_path=caption_path)


def build_metrics_table(rows: list[dict], output_dir: Path, out_name: str = "test_case_metrics",
                        caption: str | None = None) -> dict:
    """LaTeX booktabs-style table (R^2, RMSE, MAE per test case) for the
    main text, alongside a plain CSV. rows: [{"case": "$U_r=3.5$", "r2":
    ..., "rmse": ..., "mae": ...}, ...]."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Test case & $R^2$ & RMSE & MAE \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['case']} & {r['r2']:.4f} & {r['rmse']:.4f} & {r['mae']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines) + "\n"

    tex_path = output_dir / f"{out_name}.tex"
    tex_path.write_text(tex)

    import pandas as pd
    csv_path = output_dir / f"{out_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    caption_path = None
    if caption:
        caption_path = output_dir / f"{out_name}.caption.txt"
        caption_path.write_text(caption + "\n")

    return dict(tex_path=tex_path, csv_path=csv_path, caption_path=caption_path)
