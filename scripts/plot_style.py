#!/usr/bin/env python3
r"""Shared matplotlib style for the publication figures.

Every plot script calls use() once, which loads scripts/revtex.mplstyle.
The point sizes there are chosen for a figure that is included at its
natural size in a two-column RevTeX article, i.e.

    \includegraphics[width=\columnwidth]{figs/....pdf}

with no scale factor: a figure drawn at matplotlib's default 6.4 in and
then squeezed into a 3.4 in column has all of its text shrunk by the same
factor, which is what makes default-styled figures unreadable in print.
The default figure size in the style file is therefore \columnwidth, and
figsize() builds the wider sizes (\textwidth, multi-panel) from the same
measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style

STYLE_FILE = Path(__file__).with_name("revtex.mplstyle")

# RevTeX 4-2, twocolumn, letterpaper: \columnwidth = 246 pt and
# \textwidth = 510 pt, in TeX points of 1/72.27 in.
PT_PER_INCH = 72.27
COLUMN_WIDTH = 246.0 / PT_PER_INCH   # 3.404 in, one column
TEXT_WIDTH = 510.0 / PT_PER_INCH     # 7.057 in, both columns

# Height/width of a single panel. Shallower than the golden ratio, which
# stacks better in a two-column layout.
DEFAULT_RATIO = 0.75


def use() -> None:
    """Apply the RevTeX style to the global rcParams."""
    matplotlib.style.use(STYLE_FILE)


def figsize(width: float = 1.0, ratio: float = DEFAULT_RATIO,
            nrows: int = 1, ncols: int = 1) -> tuple[float, float]:
    """Figure size in inches for a figure spanning `width` columns.

    width=1 is \\columnwidth and width=2 is \\textwidth (slightly more than
    two columns, as the column separation is part of it). The height is
    ratio times the width of a single panel, so panels stay the same shape
    as the single-panel default when nrows/ncols grow."""
    w = COLUMN_WIDTH if width == 1 else width / 2.0 * TEXT_WIDTH
    return (w, nrows * ratio * w / ncols)


# A deviation panel is a fifth of the height of the plot it belongs to: enough
# to read a scatter about zero off, little enough that the plot keeps the size
# it has without one.
PANEL_HEIGHT_RATIOS = (5, 1)

# The gap between the plot and its panel, in units of the mean height of the
# two: enough that they read as two axes rather than one, little enough that
# the panel still belongs to the plot above it. It is applied after
# tight_layout rather than to the gridspec, which tight_layout refuses to
# touch once it carries a spacing of its own.
PANEL_HSPACE = 0.1


def deviation_panel(**kwargs):
    """A figure holding a plot and, underneath it, a small panel for the
    deviation of the same points from whatever they are compared against.

    Returns (fig, ax, rax). The two axes share the x axis, and the figure
    keeps the single-panel size of the style file:
    the
    panel comes out of the height of the plot rather than on top of it, so a
    figure with one is included in the paper exactly like a figure without.
    The gap between the two is PANEL_HSPACE, to be set with
    fig.subplots_adjust(hspace=...) after the figure has been laid out."""
    fig, (ax, rax) = plt.subplots(2, 1, sharex=True,
                                  height_ratios=list(PANEL_HEIGHT_RATIOS),
                                  **kwargs)
    return fig, ax, rax


def center_panel(rax) -> None:
    """Center a deviation panel on zero, labelled at +-one round step.

    Zero is the line such a panel is read against, so it belongs in the middle
    of it: the limits the points asked for are widened to the larger of the
    two rather than replaced, so nothing drops out. The step is picked by the
    same locator as everywhere else but placed by hand, because the view has
    to end above it rather than on it: a label at the top of the panel lands
    next to the lowest label of the plot."""
    half = max(abs(v) for v in rax.get_ylim())
    ticks = plt.MaxNLocator(nbins=4, symmetric=True).tick_values(-half, half)
    step = max((t for t in ticks if 0.0 < t < half), default=0.5*half)
    rax.set_yticks([-step, step])
    rax.set_ylim(-max(1.45*step, 1.05*half), max(1.45*step, 1.05*half))


def tidy_panel_ticks(ax, rax) -> None:
    """Clear the labels the plot and its deviation panel would put at the
    ends of the gap between them, and the panel's label at zero.

    The bottom of the plot is barely above the top of the panel, so labels at
    either end of that gap sit almost on top of each other; whichever axis has
    ticks to spare gives way. Zero goes
    as well: it is already drawn across the panel as a line, and the panel is
    a sixth of the figure high. Call this once both axes have their final
    limits and their final size, since it is those that say which ticks there
    are and which of them are at the gap. (Pruning the locators instead does
    not do it: a locator prunes its outermost tick before the view clips it,
    which may be one that was never visible.)"""
    for axis, edge in ((ax, "bottom"), (rax, "top")):
        lo, hi = axis.get_ylim()
        margin = 0.02*(hi - lo)
        keep = []
        for t in axis.get_yticks():
            if t < lo or t > hi:
                continue
            if edge == "bottom" and t < lo + margin:
                continue
            if edge == "top" and (t > hi - margin or t == 0.0):
                continue
            keep.append(t)
        axis.set_yticks(keep)
