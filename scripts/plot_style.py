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
