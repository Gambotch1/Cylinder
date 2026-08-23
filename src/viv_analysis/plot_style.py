# src/viv_analysis/plot_style.py
"""Shared matplotlib style for thesis-consistent figures.

Self-contained rcParams -- no scienceplots, no LaTeX/usetex dependency.
"Latin Modern Roman" / "CMU Serif" are the thesis's preferred fonts but are
not installed as matplotlib-discoverable fonts on this system (checked
directly) -- listing them in a font.serif fallback chain only produces
findfont fallback-search warnings without changing the rendered result, so
they are not listed at all. Instead this uses matplotlib's own BUNDLED STIX
fonts for both body text and math (font.family="STIXGeneral",
mathtext.fontset="stix") -- a dependency-free choice where text and math
are guaranteed to visually match each other (the previous "cm" mathtext
config paired with a DejaVu Serif text fallback did not: two visibly
different type families in the same figure). This also avoids the usetex
round-trip probe an even older version needed: lmodern's font metrics
(rm-lmr10.tfm), dvipng, and ghostscript were an ongoing source of
on-compute-node failures that plain mathtext sidesteps entirely.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito-derived semantic colours -- colorblind-safe, and named so every
# figure uses the same color for the same meaning (CFD is always
# CFD_COLOR, never "black" in one script and "gray"/"k" in another).
CFD_COLOR       = "#222222"
MODEL_COLOR     = "#0072B2"
ERROR_COLOR     = "#D55E00"
SECONDARY_COLOR = "#009E73"
PURPLE_COLOR    = "#CC79A7"
ORANGE_COLOR    = "#E69F00"
GRID_COLOR      = "#D9D9D9"

TEXT_WIDTH_CM = 16.5
TEXT_WIDTH_IN = TEXT_WIDTH_CM / 2.54

# Complete per-role line styles -- rcParams alone does not make "the CFD
# line" black and "the model line" blue; every plotting function must
# explicitly unpack one of these (**CFD_STYLE / **MODEL_STYLE /
# **ERROR_STYLE) rather than passing color= ad hoc. Distinct linestyles
# (solid vs dashed) keep CFD-vs-model readable in grayscale printouts, not
# just in color.
CFD_STYLE = {
    "color": CFD_COLOR,
    "linestyle": "-",
    "linewidth": 1.15,
    "label": "CFD reference",
    "zorder": 2,
}

MODEL_STYLE = {
    "color": MODEL_COLOR,
    "linestyle": (0, (4, 2)),  # explicit long-dash pattern -- more visibly
                               # distinct from CFD_STYLE's solid line than
                               # matplotlib's default "--" at this linewidth
    "linewidth": 1.05,
    "label": "GRU prediction",
    "zorder": 3,
}

ERROR_STYLE = {
    "color": ERROR_COLOR,
    "linestyle": "-",
    "linewidth": 0.75,
    "zorder": 2,
}


def apply_thesis_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "figure.figsize": (TEXT_WIDTH_IN, 3.8),
        "figure.dpi": 120,
        "savefig.dpi": 600,

        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        "lines.linewidth": 1.15,
        "axes.linewidth": 0.75,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,

        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,

        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })
